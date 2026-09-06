from pathlib import Path
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

from bs4 import BeautifulSoup
import pytest
import requests

import scraper
from linkedin_source_policy import ConsumedGrant, LinkedInCircuitOpen


def _disable_relist_tracking(monkeypatch):
    monkeypatch.setattr(scraper.config, "ENABLE_LINKEDIN_RELIST_TRACKING", False)


def test_extract_search_card_metadata():
    html = Path("tests/fixtures/linkedin_search_results.html").read_text()
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.find_all("li")
    results = scraper._extract_linkedin_search_cards(cards)
    assert results[0]["job_id"] == "4428124095"
    assert results[0]["posted_at"] == "2026-06-12"
    assert results[0]["posted_relative_text"] == "2 hours ago"


def test_extract_detail_metadata_applicants_salary_recruiter():
    html = Path("tests/fixtures/linkedin_job_detail_recruiter.html").read_text()
    soup = BeautifulSoup(html, "html.parser")
    details = scraper._extract_linkedin_detail_metadata(soup)
    assert details["applicant_count"] == 26
    assert details["applicant_count_text"] == "26 applicants"
    assert details["applicant_count_type"] == "exact"
    assert details["salary_text"] == "$120,000-$135,000 CAD"
    assert details["salary_min"] == 120000
    assert details["salary_max"] == 135000
    assert details["salary_currency"] == "CAD"
    assert details["recruiter_name"] == "Jane Smith"
    assert details["recruiter_profile_url"] == "https://www.linkedin.com/in/jane-smith-123456/"
    assert details["recruiter_identifier"] == "jane-smith-123456"


def test_fetch_linkedin_job_details_returns_content_and_metadata(monkeypatch):
    html = Path("tests/fixtures/linkedin_job_detail_recruiter.html").read_text()

    class Response:
        text = html

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr(scraper.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(scraper.time, "sleep", lambda _seconds: None)

    details, metadata = scraper._fetch_linkedin_job_details(
        "4428124095",
        {"posted_at": "2026-06-12", "posted_relative_text": "2 hours ago"},
    )

    assert details["job_id"] == "4428124095"
    assert details["posted_at"] == "2026-06-12"
    assert "applicant_count" not in details
    assert metadata["applicant_count"] == 26
    assert metadata["salary_min"] == 120000
    assert metadata["detail_metadata_checked_at"]


def test_fetch_linkedin_details_uses_durable_gate_without_legacy_limiter(monkeypatch):
    html = Path("tests/fixtures/linkedin_job_detail_recruiter.html").read_text()
    events = []

    class Response:
        status_code = 200
        text = html
        url = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/4428124095"

        @staticmethod
        def raise_for_status():
            return None

    class Gate:
        def acquire(self, kind, key, **_kwargs):
            events.append(("acquire", kind, key))
            return ConsumedGrant("grant-1", datetime.now(timezone.utc))

        def finish(self, grant, response_class, status):
            events.append(("finish", grant.grant_id, response_class, status))

    monkeypatch.setattr(scraper.requests, "get", lambda *_args, **_kwargs: Response())

    result = scraper._fetch_linkedin_job_details(
        "4428124095", durable_gate=Gate(), user_agent="ua"
    )

    assert result
    assert events == [
        ("acquire", "detail", "4428124095:0"),
        ("finish", "grant-1", "complete", 200),
    ]


def test_durable_detail_challenge_raises_source_circuit(monkeypatch):
    class Response:
        status_code = 403
        text = "blocked"
        url = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/123"

    class Gate:
        def acquire(self, *_args, **_kwargs):
            return ConsumedGrant("grant-1", datetime.now(timezone.utc))

        def open_circuit(self, *_args):
            return None

    monkeypatch.setattr(scraper.requests, "get", lambda *_args, **_kwargs: Response())

    with pytest.raises(LinkedInCircuitOpen):
        scraper._fetch_linkedin_job_details("123", durable_gate=Gate(), user_agent="ua")


def test_global_request_limiter_spaces_requests_from_one_clock(monkeypatch):
    clock = iter([10.0, 10.0, 11.0, 12.5])
    sleeps = []
    monkeypatch.setattr(scraper.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(scraper.time, "sleep", sleeps.append)
    monkeypatch.setattr(scraper.random, "uniform", lambda _low, _high: 0.5)
    limiter = scraper.LinkedInRequestLimiter(2_000, 1_000)

    assert limiter.wait() == 0
    assert limiter.wait() == 1.5
    assert sleeps == [1.5]
    assert limiter.request_count == 2
    assert limiter.total_wait_seconds == 1.5


def test_retry_after_supports_delta_seconds_and_http_date():
    class Response:
        headers = {"Retry-After": "42"}

    assert scraper._retry_after_seconds(Response()) == 42

    retry_at = datetime.now(timezone.utc) + timedelta(seconds=30)
    Response.headers = {"Retry-After": format_datetime(retry_at, usegmt=True)}
    assert 28 <= scraper._retry_after_seconds(Response()) <= 30


def test_linkedin_challenge_detection_covers_denial_status_and_body():
    assert scraper._linkedin_response_is_challenge(
        type("Response", (), {"status_code": 403, "text": ""})()
    )
    assert scraper._linkedin_response_is_challenge(
        type("Response", (), {"status_code": 200, "text": "<title>Security Verification</title>", "url": ""})()
    )
    assert not scraper._linkedin_response_is_challenge(
        type("Response", (), {"status_code": 200, "text": "Security verification engineer", "url": ""})()
    )


def test_search_request_failure_aborts_required_coverage(monkeypatch):
    monkeypatch.setattr(
        scraper.requests,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            requests.exceptions.ConnectionError("offline")
        ),
    )

    with pytest.raises(scraper.LinkedInRequestFailed, match="search request failed"):
        scraper._fetch_linkedin_job_ids("TPM", "Canada", max_start=0)


def test_detail_transient_http_failure_exhaustion_aborts_required_coverage(monkeypatch):
    class Response:
        status_code = 503
        text = "temporarily unavailable"
        url = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/123"
        headers = {}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

    monkeypatch.setattr(scraper.requests, "get", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(scraper.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(scraper.random, "uniform", lambda *_args: 0)
    monkeypatch.setattr(scraper.random, "choice", lambda values: values[0])

    with pytest.raises(scraper.LinkedInRequestFailed, match="HTTP 503 exhausted retries"):
        scraper._fetch_linkedin_job_details("123")


def test_detail_not_found_is_a_terminal_unavailable_result(monkeypatch):
    class Response:
        status_code = 404
        text = "not found"
        url = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/123"
        headers = {}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(response=self)

    requests_made = []
    monkeypatch.setattr(
        scraper.requests,
        "get",
        lambda *_args, **_kwargs: requests_made.append(True) or Response(),
    )
    monkeypatch.setattr(scraper.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(scraper.random, "choice", lambda values: values[0])

    result = scraper._fetch_linkedin_job_details("123")

    assert result is scraper.LINKEDIN_DETAIL_UNAVAILABLE
    assert not result
    assert len(requests_made) == 2


def test_process_query_marks_zero_cards_incomplete(monkeypatch):
    _disable_relist_tracking(monkeypatch)
    monkeypatch.setattr(scraper, "_fetch_linkedin_job_ids", lambda *_args, **_kwargs: [])

    jobs = scraper.process_linkedin_query("TPM", "Canada")

    assert jobs == []
    assert jobs.processing_complete is False
    assert jobs.incomplete_reason == "zero cards; empty result or parser/request failure"


def test_search_continues_after_duplicate_only_page_and_records_page_yield(monkeypatch):
    class Response:
        status_code = 200
        url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
        headers = {}

        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    pages = iter([
        Response('<li><div class="base-card" data-entity-urn="urn:li:jobPosting:1"></div></li>'),
        Response('<li><div class="base-card" data-entity-urn="urn:li:jobPosting:1"></div></li>'),
        Response('<li><div class="base-card" data-entity-urn="urn:li:jobPosting:2"></div></li>'),
    ])
    monkeypatch.setattr(scraper.requests, "get", lambda *_args, **_kwargs: next(pages))
    monkeypatch.setattr(scraper.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(scraper.random, "choice", lambda values: values[0])

    cards = scraper._fetch_linkedin_job_ids("TPM", "Canada", max_start=20, request_delay_ms=0)

    assert [card["job_id"] for card in cards] == ["1", "2"]
    assert scraper._linkedin_scrape_state.coverage == {
        "pages_attempted": 3,
        "pages_completed": 3,
        "page_coverage": [
            {"page": 1, "start": 0, "elements": 1, "cards": 1, "new_source_ids": 1, "result": "complete"},
            {"page": 2, "start": 10, "elements": 1, "cards": 1, "new_source_ids": 0, "result": "complete"},
            {"page": 3, "start": 20, "elements": 1, "cards": 1, "new_source_ids": 1, "result": "complete"},
        ],
    }


def test_search_rejects_nonempty_page_when_no_cards_parse(monkeypatch):
    class Response:
        status_code = 200
        text = "<li><div>unknown markup</div></li>"
        url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
        headers = {}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(scraper.requests, "get", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(scraper.random, "choice", lambda values: values[0])

    with pytest.raises(scraper.LinkedInRequestFailed, match="parser extracted zero cards"):
        scraper._fetch_linkedin_job_ids("TPM", "Canada", max_start=0)

    assert scraper._linkedin_scrape_state.coverage["page_coverage"] == [
        {
            "page": 1,
            "start": 0,
            "elements": 1,
            "cards": 0,
            "new_source_ids": 0,
            "result": "parser_failure",
        }
    ]


def test_search_records_terminal_no_results_page(monkeypatch):
    class Response:
        status_code = 200
        text = "<html></html>"
        url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
        headers = {}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(scraper.requests, "get", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(scraper.random, "choice", lambda values: values[0])

    assert scraper._fetch_linkedin_job_ids("TPM", "Canada", max_start=0) == []
    assert scraper._linkedin_scrape_state.coverage["page_coverage"][0]["result"] == "no_results"


def test_search_rejects_empty_response_body(monkeypatch):
    class Response:
        status_code = 200
        text = ""
        url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
        headers = {}

        def raise_for_status(self):
            return None

    monkeypatch.setattr(scraper.requests, "get", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(scraper.random, "choice", lambda values: values[0])

    with pytest.raises(scraper.LinkedInRequestFailed, match="empty response"):
        scraper._fetch_linkedin_job_ids("TPM", "Canada", max_start=0)

    assert scraper._linkedin_scrape_state.coverage["page_coverage"][0]["result"] == "empty_response"


def test_process_query_persists_page_coverage(monkeypatch):
    monkeypatch.setattr(scraper.config, "ENABLE_LINKEDIN_RELIST_TRACKING", True)
    finished = []

    def fetch(*_args, **_kwargs):
        scraper._linkedin_scrape_state.coverage = {
            "pages_attempted": 1,
            "pages_completed": 1,
            "page_coverage": [
                {"page": 1, "start": 0, "elements": 0, "cards": 0, "new_source_ids": 0, "result": "no_results"}
            ],
        }
        return []

    monkeypatch.setattr(scraper, "_fetch_linkedin_job_ids", fetch)
    monkeypatch.setattr(scraper.supabase_utils, "start_ingestion_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scraper.supabase_utils,
        "finish_ingestion_run",
        lambda _run_id, **metrics: finished.append(metrics),
    )

    scraper.process_linkedin_query("TPM", "Canada")

    assert finished[0]["page_coverage"] == [
        {"page": 1, "start": 0, "elements": 0, "cards": 0, "new_source_ids": 0, "result": "no_results"}
    ]


def test_parse_salary_supports_common_range_formats():
    assert scraper._parse_salary_fields("CAD $120K to $135K per year") == {
        "salary_text": "CAD $120K to $135K",
        "salary_min": 120000,
        "salary_max": 135000,
        "salary_currency": "CAD",
    }
    assert scraper._parse_salary_fields("Compensation: $65,000–$80,000") == {
        "salary_text": "$65,000–$80,000",
        "salary_min": 65000,
        "salary_max": 80000,
        "salary_currency": None,
    }


def test_parse_salary_rejects_date_ranges():
    assert scraper._parse_salary_fields("Experience delivering programs from 2020-2025") == {
        "salary_text": None,
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
    }


def test_parse_salary_rejects_project_budget_ranges():
    assert scraper._parse_salary_fields("Managed projects ranging from $5M-$20M in value") == {
        "salary_text": None,
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
    }


def test_phase1_posted_at_metadata_is_attached_to_detail_record(monkeypatch):
    _disable_relist_tracking(monkeypatch)
    cards = [{"job_id": "123", "posted_at": "2026-06-12", "posted_relative_text": "2 hours ago"}]

    monkeypatch.setattr(scraper, "_fetch_linkedin_job_ids", lambda query, location, posting_date_filter=None: cards)
    monkeypatch.setattr(scraper.supabase_utils, "get_existing_jobs_from_supabase", lambda: (set(), set()))
    monkeypatch.setattr(scraper.supabase_utils, "get_incomplete_linkedin_metadata_ids", lambda _ids: set())
    monkeypatch.setattr(scraper, "_fetch_linkedin_job_details", lambda job_id, search_card=None: ({
        "job_id": job_id,
        "description": "Real description",
        "company": "Acme",
        "job_title": "Technical Project Manager",
        "location": "Toronto, Ontario, Canada",
        "provider": "linkedin",
        "posted_at": search_card["posted_at"],
        "posted_relative_text": search_card["posted_relative_text"],
    }, {"applicant_count": 26}))

    results = scraper.process_linkedin_query("TPM", "Canada")
    assert results[0]["posted_at"] == "2026-06-12"
    assert results[0]["posted_relative_text"] == "2 hours ago"
    assert results[0]["applicant_count"] == 26


def test_process_linkedin_query_output_flows_through_linkedin_and_generic_savers(monkeypatch):
    _disable_relist_tracking(monkeypatch)
    cards = [{"job_id": "123", "posted_at": "2026-06-12", "posted_relative_text": "2 hours ago"}]
    content = {
        "job_id": "123",
        "description": "Real description",
        "company": "Acme",
        "job_title": "Technical Project Manager",
        "location": "Toronto, Ontario, Canada",
        "provider": "linkedin",
        "posted_at": "2026-06-12",
    }
    detail_metadata = {
        "applicant_count": 26,
        "salary_text": "$120,000 CAD",
        "detail_metadata_checked_at": "2026-06-12T10:00:00+00:00",
    }
    saved = []

    monkeypatch.setattr(scraper.config, "ENABLE_REPOST_DEDUP", False)
    monkeypatch.setattr(scraper, "_fetch_linkedin_job_ids", lambda query, location, posting_date_filter=None: cards)
    monkeypatch.setattr(scraper.supabase_utils, "get_existing_jobs_from_supabase", lambda: (set(), set()))
    monkeypatch.setattr(scraper.supabase_utils, "get_incomplete_linkedin_metadata_ids", lambda _ids: set())
    monkeypatch.setattr(
        scraper,
        "_fetch_linkedin_job_details",
        lambda job_id, search_card=None: (content, detail_metadata),
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "save_job_to_supabase",
        lambda job: saved.append(job) or job["job_id"],
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "upsert_job_archetype_membership",
        lambda _job_id, _job: None,
    )

    jobs = scraper.process_linkedin_query("TPM", "Canada")
    linkedin_saved_job_ids = scraper.supabase_utils.save_linkedin_jobs_canonicalized(jobs)
    generic_saved_job_ids = scraper.supabase_utils.save_jobs_canonicalized(jobs)

    assert len(jobs) == 1
    assert isinstance(jobs[0], dict)
    assert linkedin_saved_job_ids == ["123"]
    assert generic_saved_job_ids == ["123"]
    assert jobs[0]["applicant_count"] == 26
    assert "applicant_count" not in content
    assert detail_metadata == {
        "applicant_count": 26,
        "salary_text": "$120,000 CAD",
        "detail_metadata_checked_at": "2026-06-12T10:00:00+00:00",
    }
    assert len(saved) == 2
    assert all(job["salary_text"] == "$120,000 CAD" for job in saved)
    assert all(job["detail_metadata_checked_at"] == "2026-06-12T10:00:00+00:00" for job in saved)
    assert all(job["listing_instances"][0]["applicant_count"] == 26 for job in saved)


def test_process_linkedin_query_skips_ids_already_seen_as_latest_job_id(monkeypatch):
    _disable_relist_tracking(monkeypatch)
    cards = [{"job_id": "linkedin-live-99", "posted_at": "2026-06-12", "posted_relative_text": "2 hours ago"}]
    fetched_job_ids = []

    monkeypatch.setattr(scraper, "_fetch_linkedin_job_ids", lambda query, location, posting_date_filter=None: cards)
    monkeypatch.setattr(
        scraper.supabase_utils,
        "get_existing_jobs_from_supabase",
        lambda: ({"canonical-1", "linkedin-live-99"}, set()),
    )
    monkeypatch.setattr(scraper.supabase_utils, "get_incomplete_linkedin_metadata_ids", lambda _ids: set())
    monkeypatch.setattr(
        scraper,
        "_fetch_linkedin_job_details",
        lambda job_id, search_card=None: fetched_job_ids.append(job_id),
    )

    results = scraper.process_linkedin_query("TPM", "Canada")

    assert results == []
    assert fetched_job_ids == []


def test_existing_job_with_stale_metadata_is_refetched(monkeypatch):
    _disable_relist_tracking(monkeypatch)
    monkeypatch.setattr(
        scraper,
        "_fetch_linkedin_job_ids",
        lambda query, location, posting_date_filter=None: [{"job_id": "existing", "posted_at": "2026-06-12"}],
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "get_existing_jobs_from_supabase",
        lambda: ({"existing"}, set()),
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "get_incomplete_linkedin_metadata_ids",
        lambda _ids: {"existing"},
    )
    monkeypatch.setattr(
        scraper,
        "_fetch_linkedin_job_details",
        lambda job_id, search_card=None: ({"job_id": job_id, "description": "Updated detail"}, {}),
    )

    results = scraper.process_linkedin_query("project manager", "Canada")

    assert [job["job_id"] for job in results] == ["existing"]
