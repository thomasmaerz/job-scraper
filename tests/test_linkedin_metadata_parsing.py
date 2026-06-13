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
