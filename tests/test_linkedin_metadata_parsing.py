from pathlib import Path

from bs4 import BeautifulSoup

import scraper


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
