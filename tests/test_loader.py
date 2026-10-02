from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from src.config import APPROVED_SOURCE_ROLES, SUPPORTED_SCHEMES
from src.loader import (
    BlockedPageError,
    DisallowedRedirectError,
    DisallowedSourceError,
    FetchError,
    PageLoader,
    SourceSpec,
    UnsupportedContentError,
    load_source_manifest,
)


FIXTURES = Path(__file__).parent / "fixtures"
SCHEME = SUPPORTED_SCHEMES[0]
SPEC = SourceSpec(
    url=SCHEME.seed_url,
    title=SCHEME.canonical_name,
    scheme_name=SCHEME.canonical_name,
    role=APPROVED_SOURCE_ROLES[SCHEME.seed_url],
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def test_manifest_contains_five_schemes_and_reviewed_follow_ons() -> None:
    sources = load_source_manifest("seed_sources.csv")

    assert len(sources) == 8
    assert {source.url for source in sources} == set(APPROVED_SOURCE_ROLES)
    assert all(source.role for source in sources)


def test_manifest_rejects_unapproved_url(tmp_path: Path) -> None:
    manifest = tmp_path / "sources.csv"
    manifest.write_text(
        "url,title,scheme\n"
        "https://example.com/fund,Unreviewed,Shared investor education and documents\n",
        encoding="utf-8",
    )

    with pytest.raises(DisallowedSourceError, match="not explicitly approved"):
        load_source_manifest(manifest)


def test_loader_cleans_noise_and_preserves_structures() -> None:
    html = (FIXTURES / "scheme_page.html").read_text(encoding="utf-8")
    client = _client(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=html,
            request=request,
        )
    )

    document = PageLoader([SPEC], client=client, retry_backoff_seconds=0).load(SPEC)

    assert document.source_url == SPEC.url
    assert document.scheme_name == SCHEME.canonical_name
    assert "Expense ratio" in document.html
    assert "Minimum SIP" in document.html
    assert "NIFTY 100 Total Return Index" in document.html
    assert "Sponsored content" not in document.html
    assert "window.tracker" not in document.html
    assert "<table>" in document.html
    assert "<h2>" in document.html


def test_loader_retries_then_reports_failed_request() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text="unavailable", request=request)

    client = _client(handler)
    loader = PageLoader([SPEC], client=client, max_retries=2, retry_backoff_seconds=0)

    with pytest.raises(FetchError, match="after 3 attempts"):
        loader.load(SPEC)
    assert attempts == 3


def test_loader_rejects_disallowed_redirect() -> None:
    client = _client(
        lambda request: httpx.Response(
            302,
            headers={"location": "https://evil.example/fund"},
            request=request,
        )
    )

    with pytest.raises(DisallowedRedirectError, match="not approved"):
        PageLoader([SPEC], client=client, retry_backoff_seconds=0).load(SPEC)


def test_loader_rejects_pdf_response() -> None:
    client = _client(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF-1.7 fixture",
            request=request,
        )
    )

    with pytest.raises(UnsupportedContentError, match="PDF extraction"):
        PageLoader([SPEC], client=client, retry_backoff_seconds=0).load(SPEC)


def test_loader_reports_http_403_as_blocked() -> None:
    client = _client(
        lambda request: httpx.Response(403, text="Request rejected", request=request)
    )

    with pytest.raises(BlockedPageError, match="HTTP 403"):
        PageLoader([SPEC], client=client, retry_backoff_seconds=0).load(SPEC)


@pytest.mark.parametrize("fixture", ["blocked_page.html", "javascript_shell.html"])
def test_loader_rejects_block_and_javascript_shell(fixture: str) -> None:
    html = (FIXTURES / fixture).read_text(encoding="utf-8")
    client = _client(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text=html,
            request=request,
        )
    )

    with pytest.raises(BlockedPageError):
        PageLoader([SPEC], client=client, retry_backoff_seconds=0).load(SPEC)


def test_loader_constructor_rejects_arbitrary_source() -> None:
    arbitrary = SourceSpec(
        url="https://example.com/fund",
        title="Example",
        scheme_name=SCHEME.canonical_name,
        role="Unreviewed",
    )

    with pytest.raises(DisallowedSourceError, match="reviewed corpus"):
        PageLoader([arbitrary])
