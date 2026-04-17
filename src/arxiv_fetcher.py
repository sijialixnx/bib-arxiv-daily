from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import logging
from urllib.parse import urlencode

import feedparser

from models import ArxivFetchStats, CandidatePaper
from utils import chunked, clean_text, extract_arxiv_id


LOGGER = logging.getLogger(__name__)
ARXIV_EXPORT_API_URL = "https://export.arxiv.org/api/query"
RSS_FALLBACK_WINDOW_HOURS = 24


class ArxivFetcher:
    def __init__(
        self,
        categories: tuple[str, ...],
        max_candidates: int,
        feedparser_module=feedparser,
        arxiv_module=None,
        now_fn: Callable[[], datetime] | None = None,
    ):
        self.categories = categories
        self.max_candidates = max_candidates
        self._feedparser = feedparser_module
        self._arxiv_module = arxiv_module
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def _get_arxiv_module(self):
        if self._arxiv_module is None:
            import arxiv

            self._arxiv_module = arxiv
        return self._arxiv_module

    def fetch_new_papers(self) -> tuple[list[CandidatePaper], ArxivFetchStats]:
        if not self.categories:
            raise ValueError("At least one arXiv category must be configured")

        query = "+".join(self.categories)
        feed_url = f"https://rss.arxiv.org/atom/{query}"
        feed = self._feedparser.parse(feed_url)
        title = getattr(getattr(feed, "feed", None), "title", "")
        if isinstance(title, str) and "Feed error for query" in title:
            raise ValueError(f"Invalid arXiv RSS query: {query}")

        paper_ids = []
        for entry in getattr(feed, "entries", []):
            if entry.get("arxiv_announce_type", "new") != "new":
                continue
            entry_id = clean_text(entry.get("id"))
            if not entry_id:
                continue
            paper_ids.append(entry_id.removeprefix("oai:arXiv.org:"))

        unique_ids = list(dict.fromkeys(paper_ids))[: self.max_candidates]
        LOGGER.info("Fetched %s new arXiv ids from RSS feed", len(unique_ids))
        if not unique_ids:
            window_end = self._normalize_utc(self._now_fn())
            window_start = window_end - timedelta(hours=RSS_FALLBACK_WINDOW_HOURS)
            fallback_candidates = self._fetch_recent_papers_via_api(window_start=window_start, window_end=window_end)
            return fallback_candidates, ArxivFetchStats(
                rss_new_count=len(paper_ids),
                rss_unique_count=len(unique_ids),
                fetched_candidate_count=len(fallback_candidates),
                query_mode="rss_new",
                fallback_used=True,
                fallback_window_hours=RSS_FALLBACK_WINDOW_HOURS,
                fallback_candidate_count=len(fallback_candidates),
            )

        arxiv_module = self._get_arxiv_module()
        client = arxiv_module.Client(num_retries=5, delay_seconds=6)
        candidates: list[CandidatePaper] = []

        for batch in chunked(unique_ids, 5):
            search = arxiv_module.Search(id_list=batch)
            for result in client.results(search):
                candidates.append(self._convert_result(result))
            time.sleep(5) 
        candidates.sort(key=self._published_sort_key, reverse=True)
        fetch_stats = ArxivFetchStats(
            rss_new_count=len(paper_ids),
            rss_unique_count=len(unique_ids),
            fetched_candidate_count=len(candidates[: self.max_candidates]),
            query_mode="rss_new",
        )
        return candidates[: self.max_candidates], fetch_stats

    def fetch_recent_papers(self, days: int) -> tuple[list[CandidatePaper], ArxivFetchStats]:
        if days < 1:
            raise ValueError("lookback days must be at least 1")

        window_end = self._normalize_utc(self._now_fn())
        window_start = window_end - timedelta(days=days)
        candidates = self._fetch_recent_papers_via_api(window_start=window_start, window_end=window_end)
        return candidates, ArxivFetchStats(
            rss_new_count=0,
            rss_unique_count=0,
            fetched_candidate_count=len(candidates),
            query_mode="lookback",
            lookback_days=days,
        )

    def _fetch_recent_papers_via_api(self, window_start: datetime, window_end: datetime) -> list[CandidatePaper]:
        arxiv_module = self._get_arxiv_module()
        query = self._build_recent_query(window_start, window_end)
        api_url = f"{ARXIV_EXPORT_API_URL}?{urlencode(self._build_recent_query_params(query))}"
        LOGGER.info("Querying arXiv export API over explicit time window: %s", api_url)

        client = arxiv_module.Client(num_retries=5, delay_seconds=3)
        sort_criterion = getattr(getattr(arxiv_module, "SortCriterion", None), "SubmittedDate", None)
        sort_order = getattr(getattr(arxiv_module, "SortOrder", None), "Descending", None)
        search_kwargs = dict(
            query=query,
            max_results=30,
        )
        if sort_criterion is not None:
            search_kwargs["sort_by"] = sort_criterion
        if sort_order is not None:
            search_kwargs["sort_order"] = sort_order
        search = arxiv_module.Search(**search_kwargs)
        candidates = [self._convert_result(result) for result in client.results(search)]
        candidates.sort(key=self._published_sort_key, reverse=True)
        return candidates[: self.max_candidates]

    def _build_recent_query(self, window_start: datetime, window_end: datetime) -> str:
        category_query = " OR ".join(f"cat:{category}" for category in self.categories)
        start_token = window_start.strftime("%Y%m%d%H%M%S")
        end_token = window_end.strftime("%Y%m%d%H%M%S")
        # [EN] Query by submittedDate over an explicit UTC window so manual weekly runs and RSS fallback both operate on a reproducible candidate set. / [CN] 按显式 UTC 时间窗口查询 submittedDate，这样手动周报和 RSS 回退都能基于可复现的候选集合工作。
        return f"submittedDate:[{start_token} TO {end_token}] AND ({category_query})"

    def _build_recent_query_params(self, query: str) -> dict[str, str | int]:
        return {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": self.max_candidates,
        }

    def _normalize_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _published_sort_key(self, paper: CandidatePaper) -> datetime:
        published = paper.published
        if published is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        return self._normalize_utc(published)

    def _convert_result(self, result) -> CandidatePaper:
        title = clean_text(getattr(result, "title", ""))
        abstract = clean_text(getattr(result, "summary", ""))
        authors = tuple(clean_text(author.name) for author in getattr(result, "authors", []) if clean_text(author.name))
        entry_id = clean_text(getattr(result, "entry_id", ""))
        pdf_url = clean_text(getattr(result, "pdf_url", "")) or None
        categories = tuple(getattr(result, "categories", []) or ())
        doi = clean_text(getattr(result, "doi", "")) or None
        published = getattr(result, "published", None)
        arxiv_id = extract_arxiv_id(entry_id, pdf_url)
        if arxiv_id is None:
            arxiv_id = entry_id.rsplit("/", maxsplit=1)[-1]

        return CandidatePaper(
            title=title,
            abstract=abstract,
            authors=authors,
            entry_id=entry_id or arxiv_id,
            pdf_url=pdf_url,
            published=published,
            categories=categories,
            doi=doi,
            arxiv_id=arxiv_id,
        )
