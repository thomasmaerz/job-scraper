from pathlib import Path

from bs4 import BeautifulSoup

import scraper


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
    assert details["salary_text"] == "$120,000-$135,000 CAD"
    assert details["salary_min"] == 120000
    assert details["salary_max"] == 135000
    assert details["salary_currency"] == "CAD"
    assert details["recruiter_name"] == "Jane Smith"
    assert details["recruiter_profile_url"] == "https://www.linkedin.com/in/jane-smith-123456/"
    assert details["recruiter_identifier"] == "jane-smith-123456"


def test_phase1_posted_at_metadata_is_attached_to_detail_record(monkeypatch):
    cards = [{"job_id": "123", "posted_at": "2026-06-12", "posted_relative_text": "2 hours ago"}]

    monkeypatch.setattr(scraper, "_fetch_linkedin_job_ids", lambda q, l: cards)
    monkeypatch.setattr(scraper.supabase_utils, "get_existing_jobs_from_supabase", lambda: (set(), set()))
    monkeypatch.setattr(scraper, "_fetch_linkedin_job_details", lambda job_id, search_card=None: {
        "job_id": job_id,
        "description": "Real description",
        "company": "Acme",
        "job_title": "Technical Project Manager",
        "location": "Toronto, Ontario, Canada",
        "provider": "linkedin",
        "posted_at": search_card["posted_at"],
        "posted_relative_text": search_card["posted_relative_text"],
    })

    results = scraper.process_linkedin_query("TPM", "Canada")
    assert results[0]["posted_at"] == "2026-06-12"
    assert results[0]["posted_relative_text"] == "2 hours ago"
