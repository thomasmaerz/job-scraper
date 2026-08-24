from datetime import datetime, timezone
from unittest.mock import patch

import scraper
import pytest


def test_linkedin_lookback_extends_from_last_success_with_overlap(monkeypatch):
    monkeypatch.setattr(scraper.config, "LINKEDIN_LOOKBACK_HOURS", 48)
    monkeypatch.setattr(scraper.config, "LINKEDIN_LOOKBACK_OVERLAP_HOURS", 6)
    monkeypatch.setattr(scraper.config, "LINKEDIN_MAX_LOOKBACK_HOURS", 168)

    hours = scraper.resolve_linkedin_lookback_hours(
        "2026-08-22T10:00:00+00:00",
        now=datetime(2026, 8, 24, 16, 0, tzinfo=timezone.utc),
    )

    assert hours == 60


def test_linkedin_lookback_uses_safe_default_without_watermark(monkeypatch):
    monkeypatch.setattr(scraper.config, "LINKEDIN_LOOKBACK_HOURS", 48)

    assert scraper.resolve_linkedin_lookback_hours(None) == 48


def test_linkedin_lookback_caps_long_outage(monkeypatch):
    monkeypatch.setattr(scraper.config, "LINKEDIN_LOOKBACK_HOURS", 48)
    monkeypatch.setattr(scraper.config, "LINKEDIN_LOOKBACK_OVERLAP_HOURS", 6)
    monkeypatch.setattr(scraper.config, "LINKEDIN_MAX_LOOKBACK_HOURS", 168)

    hours = scraper.resolve_linkedin_lookback_hours(
        "2026-08-01T10:00:00Z",
        now=datetime(2026, 8, 24, 16, 0, tzinfo=timezone.utc),
    )

    assert hours == 168


def test_process_linkedin_query_stamps_search_query_archetype_and_filter_profile():
    fake_details = {
        "job_id": "123",
        "job_title": "Technical Program Manager",
        "description": "Own software delivery and cross-functional execution.",
        "company": "Example Corp",
        "provider": "linkedin",
    }

    with patch.object(scraper, "_fetch_linkedin_job_ids", return_value=["123"]), \
         patch.object(scraper.supabase_utils, "get_existing_jobs_from_supabase", return_value=(set(), set())), \
         patch.object(scraper, "_fetch_linkedin_job_details", return_value=fake_details):
        jobs = scraper.process_linkedin_query(
            search_query="Technical Program Manager",
            location="Canada",
            limit=None,
            archetype="software_tpm",
            filter_profile="software_tpm_v1",
        )

    assert jobs == [{
        "job_id": "123",
        "job_title": "Technical Program Manager",
        "description": "Own software delivery and cross-functional execution.",
        "company": "Example Corp",
        "provider": "linkedin",
        "search_query": "Technical Program Manager",
        "archetype": "software_tpm",
        "filter_profile": "software_tpm_v1",
        "scrape_run_id": scraper.SCRAPE_RUN_ID,
    }]


def test_process_linkedin_query_raises_clear_error_for_unknown_archetype():
    with patch.object(scraper, "_fetch_linkedin_job_ids", return_value=["123"]), \
         patch.object(scraper.supabase_utils, "get_existing_jobs_from_supabase", return_value=(set(), set())):
        with pytest.raises(ValueError, match="Unknown archetype 'not_real'"):
            scraper.process_linkedin_query(
                search_query="Technical Program Manager",
                location="Canada",
                archetype="not_real",
            )


def test_process_linkedin_query_raises_clear_error_for_missing_filter_profile(monkeypatch):
    original_configs = scraper.config.ARCHETYPE_CONFIGS
    monkeypatch.setattr(
        scraper.config,
        "ARCHETYPE_CONFIGS",
        {"broken": {"provider": "linkedin", "search_queries": [], "location": "Canada"}},
    )

    with patch.object(scraper, "_fetch_linkedin_job_ids", return_value=["123"]), \
         patch.object(scraper.supabase_utils, "get_existing_jobs_from_supabase", return_value=(set(), set())):
        with pytest.raises(ValueError, match=r"missing required config key\(s\): filter_profile"):
            scraper.process_linkedin_query(
                search_query="Technical Program Manager",
                location="Canada",
                archetype="broken",
            )

    monkeypatch.setattr(scraper.config, "ARCHETYPE_CONFIGS", original_configs)
