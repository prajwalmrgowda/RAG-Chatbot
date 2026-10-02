"""Allowlisted HTML source loading for the ingestion pipeline."""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, Tag

from src.config import (
    APPROVED_FOLLOW_ON_SOURCES,
    APPROVED_SOURCE_ROLES,
    HTTP_CONNECT_TIMEOUT_SECONDS,
    HTTP_MAX_REDIRECTS,
    HTTP_MAX_RETRIES,
    HTTP_READ_TIMEOUT_SECONDS,
    SUPPORTED_SCHEMES,
)
from src.models import SourceDocument


class SourceLoadError(RuntimeError):
    """Base class for an explicit, reportable source-loading failure."""


class ManifestError(SourceLoadError):
    """The source manifest is malformed or contains an unapproved source."""


class DisallowedSourceError(SourceLoadError):
    """A URL is not part of the fixed, reviewed corpus."""


class DisallowedRedirectError(SourceLoadError):
    """A source redirected outside the reviewed corpus."""


class FetchError(SourceLoadError):
    """A source could not be fetched successfully after bounded retries."""


class UnsupportedContentError(SourceLoadError):
    """A source uses a content type that the HTML loader cannot parse."""


class BlockedPageError(SourceLoadError):
    """A response appears to be a bot challenge or access-denied page."""


class EmptyPageError(SourceLoadError):
    """A response has no useful server-rendered content to index."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One reviewed manifest row."""

    url: str
    title: str
    scheme_name: str
    role: str


_EXPECTED_MANIFEST_FIELDS = ("url", "title", "scheme")
_RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_NOISE_NAME = re.compile(
    r"(^|[-_\s])(ad|ads|advert|banner|breadcrumb|cookie|footer|menu|"
    r"modal|nav|newsletter|popup|promo|sidebar|social)([-_\s]|$)",
    re.IGNORECASE,
)
_BLOCK_PAGE_MARKERS = (
    "access denied",
    "attention required",
    "checking your browser",
    "complete the security check",
    "enable javascript and cookies",
    "human verification",
    "unusual traffic",
    "verify you are human",
)


def normalize_url(url: str) -> str:
    """Return a stable HTTP(S) URL for allowlist and chunk-ID comparisons."""

    raw = url.strip()
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    if scheme not in {"http", "https"} or not hostname:
        raise DisallowedSourceError(f"Only absolute HTTP(S) URLs are allowed: {url!r}")
    if parts.username or parts.password:
        raise DisallowedSourceError("Source URLs must not contain credentials")

    port = parts.port
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _approved_source_catalog() -> dict[str, tuple[str, str, str]]:
    catalog: dict[str, tuple[str, str, str]] = {}
    for scheme in SUPPORTED_SCHEMES:
        catalog[normalize_url(scheme.seed_url)] = (
            scheme.canonical_name,
            scheme.canonical_name,
            APPROVED_SOURCE_ROLES[scheme.seed_url],
        )
    for source in APPROVED_FOLLOW_ON_SOURCES:
        catalog[normalize_url(source.url)] = (
            source.title,
            source.scope,
            source.role,
        )
    return catalog


def load_source_manifest(path: str | Path) -> tuple[SourceSpec, ...]:
    """Read and validate the fixed source manifest without fetching anything."""

    manifest_path = Path(path)
    try:
        handle = manifest_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"Could not read source manifest {manifest_path}: {exc}") from exc

    catalog = _approved_source_catalog()
    seen: set[str] = set()
    sources: list[SourceSpec] = []
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _EXPECTED_MANIFEST_FIELDS:
            raise ManifestError(
                "Source manifest columns must be exactly: url,title,scheme"
            )
        for line_number, row in enumerate(reader, start=2):
            if any(not (row.get(field) or "").strip() for field in _EXPECTED_MANIFEST_FIELDS):
                raise ManifestError(f"Blank manifest value on line {line_number}")
            normalized = normalize_url(row["url"])
            if normalized in seen:
                raise ManifestError(f"Duplicate source URL on line {line_number}: {row['url']}")
            approved = catalog.get(normalized)
            if approved is None:
                raise DisallowedSourceError(
                    f"Source on line {line_number} is not explicitly approved: {row['url']}"
                )
            approved_title, approved_scope, role = approved
            if row["title"].strip() != approved_title:
                raise ManifestError(
                    f"Title on line {line_number} does not match the approved source"
                )
            if row["scheme"].strip() != approved_scope:
                raise ManifestError(
                    f"Scheme/scope on line {line_number} does not match the approved source"
                )
            sources.append(
                SourceSpec(
                    url=normalized,
                    title=approved_title,
                    scheme_name=approved_scope,
                    role=role,
                )
            )
            seen.add(normalized)

    required_scheme_urls = {
        normalize_url(scheme.seed_url) for scheme in SUPPORTED_SCHEMES
    }
    missing = required_scheme_urls - seen
    if missing:
        raise ManifestError(
            "Manifest must include all five scheme pages; missing: "
            + ", ".join(sorted(missing))
        )
    return tuple(sources)


class PageLoader:
    """Fetch and clean explicitly approved HTML pages."""

    def __init__(
        self,
        approved_sources: Iterable[SourceSpec],
        *,
        client: httpx.Client | None = None,
        max_retries: int = HTTP_MAX_RETRIES,
        max_redirects: int = HTTP_MAX_REDIRECTS,
        retry_backoff_seconds: float = 0.25,
    ) -> None:
        sources = tuple(approved_sources)
        if not sources:
            raise ValueError("At least one approved source is required")
        if max_retries < 0 or max_redirects < 0:
            raise ValueError("Retry and redirect limits must be non-negative")

        approved_catalog = _approved_source_catalog()
        validated_sources: dict[str, SourceSpec] = {}
        for source in sources:
            normalized = normalize_url(source.url)
            approved = approved_catalog.get(normalized)
            if approved is None:
                raise DisallowedSourceError(
                    f"Source is not in the reviewed corpus: {source.url}"
                )
            approved_title, approved_scope, approved_role = approved
            if (
                source.title != approved_title
                or source.scheme_name != approved_scope
                or source.role != approved_role
            ):
                raise DisallowedSourceError(
                    f"Source metadata does not match the reviewed corpus: {source.url}"
                )
            validated_sources[normalized] = source

        if len(validated_sources) != len(sources):
            raise ValueError("Approved source URLs must be unique")
        self._sources = validated_sources
        self._client = client
        self._max_retries = max_retries
        self._max_redirects = max_redirects
        self._retry_backoff_seconds = retry_backoff_seconds

    def load(self, source: SourceSpec | str) -> SourceDocument:
        """Load one source; raise a typed failure rather than returning bad data."""

        requested_url = source.url if isinstance(source, SourceSpec) else source
        normalized = normalize_url(requested_url)
        spec = self._sources.get(normalized)
        if spec is None:
            raise DisallowedSourceError(f"Source is not in the active manifest: {requested_url}")
        if normalized.lower().endswith(".pdf"):
            raise UnsupportedContentError("PDF extraction is not implemented")

        if self._client is not None:
            response = self._fetch_with_redirect_validation(self._client, spec)
        else:
            timeout = httpx.Timeout(
                connect=HTTP_CONNECT_TIMEOUT_SECONDS,
                read=HTTP_READ_TIMEOUT_SECONDS,
                write=HTTP_READ_TIMEOUT_SECONDS,
                pool=HTTP_CONNECT_TIMEOUT_SECONDS,
            )
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; HDFCFactsRAG/1.0; "
                    "+local-educational-demo)"
                ),
                "Accept": "text/html,application/xhtml+xml",
            }
            with httpx.Client(
                timeout=timeout,
                headers=headers,
                follow_redirects=False,
            ) as client:
                response = self._fetch_with_redirect_validation(client, spec)

        content_type = response.headers.get("content-type", "").lower()
        if "pdf" in content_type or normalize_url(str(response.url)).lower().endswith(".pdf"):
            raise UnsupportedContentError("PDF extraction is not implemented")
        if content_type and not any(
            allowed in content_type for allowed in ("text/html", "application/xhtml+xml")
        ):
            raise UnsupportedContentError(
                f"Unsupported content type for {spec.url}: {content_type}"
            )

        page_title, cleaned_html = self._clean_html(response.text, spec)
        return SourceDocument(
            source_url=normalize_url(str(response.url)),
            page_title=page_title,
            scheme_name=spec.scheme_name,
            html=cleaned_html,
        )

    def load_all(self) -> tuple[SourceDocument, ...]:
        """Load all active sources in manifest order, failing on the first error."""

        return tuple(self.load(source) for source in self._sources.values())

    def _fetch_with_redirect_validation(
        self,
        client: httpx.Client,
        source: SourceSpec,
    ) -> httpx.Response:
        current_url = source.url
        redirects = 0
        while True:
            response = self._request_with_retries(client, current_url)
            if response.status_code not in _REDIRECT_STATUS_CODES:
                if response.status_code in {401, 403}:
                    raise BlockedPageError(
                        f"HTTP {response.status_code} access blocked for {current_url}"
                    )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise FetchError(
                        f"HTTP {response.status_code} while fetching {current_url}"
                    ) from exc
                return response

            location = response.headers.get("location")
            if not location:
                raise FetchError(f"Redirect from {current_url} has no Location header")
            redirects += 1
            if redirects > self._max_redirects:
                raise FetchError(f"Too many redirects while fetching {source.url}")
            destination = normalize_url(urljoin(current_url, location))
            destination_spec = self._sources.get(destination)
            if destination_spec is None:
                raise DisallowedRedirectError(
                    f"Redirect destination is not approved: {destination}"
                )
            if destination_spec.scheme_name != source.scheme_name:
                raise DisallowedRedirectError(
                    "Redirect destination belongs to a different scheme/scope"
                )
            current_url = destination

    def _request_with_retries(
        self,
        client: httpx.Client,
        url: str,
    ) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = client.get(url)
            except httpx.RequestError as exc:
                last_error = exc
            else:
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    return response
                last_error = FetchError(
                    f"Retryable HTTP {response.status_code} while fetching {url}"
                )
            if attempt < self._max_retries and self._retry_backoff_seconds:
                time.sleep(self._retry_backoff_seconds * (2**attempt))
        raise FetchError(
            f"Failed to fetch {url} after {self._max_retries + 1} attempts"
        ) from last_error

    @staticmethod
    def _clean_html(html: str, source: SourceSpec) -> tuple[str, str]:
        if not html.strip():
            raise EmptyPageError(f"Empty response for {source.url}")

        soup = BeautifulSoup(html, "html.parser")
        raw_text = " ".join(soup.stripped_strings).casefold()
        if any(marker in raw_text for marker in _BLOCK_PAGE_MARKERS):
            raise BlockedPageError(f"Blocked/challenge page returned for {source.url}")

        title_tag = soup.find("title")
        page_title = (
            " ".join(title_tag.stripped_strings).strip()
            if isinstance(title_tag, Tag)
            else source.title
        )
        if not page_title:
            page_title = source.title

        content_root = soup.find("main") or soup.find("article")
        if content_root is None:
            primary_heading = soup.find("h1")
            if isinstance(primary_heading, Tag):
                content_root = primary_heading.find_parent(
                    class_=re.compile(r"(^|[-_])layout-main($|[-_])", re.IGNORECASE)
                )

        root = content_root or soup.body or soup
        for tag in root.find_all(
            ["script", "style", "noscript", "svg", "canvas", "iframe", "form", "template"]
        ):
            tag.decompose()
        for tag in root.find_all(["nav", "footer", "aside"]):
            tag.decompose()
        for tag in root.find_all("header"):
            if tag.find("h1") is None:
                tag.decompose()
        for tag in list(root.find_all(True)):
            if not isinstance(tag, Tag) or tag.parent is None:
                continue
            identity = " ".join(
                [tag.get("id", ""), *tag.get("class", [])]
            ).strip()
            role = tag.get("role", "").casefold()
            if (identity and _NOISE_NAME.search(identity)) or role in {
                "banner",
                "complementary",
                "dialog",
                "navigation",
            }:
                tag.decompose()
                continue
            if tag.has_attr("hidden") or tag.get("aria-hidden", "").casefold() == "true":
                tag.decompose()

        meaningful_text = " ".join(root.stripped_strings).strip()
        if len(meaningful_text) < 80:
            if "javascript" in meaningful_text.casefold():
                raise BlockedPageError(
                    f"JavaScript-only shell returned for {source.url}"
                )
            raise EmptyPageError(f"No useful server-rendered content for {source.url}")

        for tag in root.find_all(True):
            tag.attrs = {}
        return page_title, str(root)


__all__ = [
    "BlockedPageError",
    "DisallowedRedirectError",
    "DisallowedSourceError",
    "EmptyPageError",
    "FetchError",
    "ManifestError",
    "PageLoader",
    "SourceLoadError",
    "SourceSpec",
    "UnsupportedContentError",
    "load_source_manifest",
    "normalize_url",
]
