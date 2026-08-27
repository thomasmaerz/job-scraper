import logging

import scraper


def test_main_consumes_canonical_saver_id_lists_for_all_sources(monkeypatch, caplog):
    linkedin_jobs = [{"job_id": "linkedin-source-id", "provider": "linkedin"}]
    careers_future_jobs = [
        {"job_id": "careers-future-source-id", "provider": "careers_future"}
    ]
    linkedin_save_calls = []
    generic_save_calls = []

    monkeypatch.setattr(scraper.config, "SCRAPING_SOURCES", {"linkedin", "careers_future"})
    monkeypatch.setattr(scraper.config, "LINKEDIN_LAST_SUCCESS_AT", None)
    monkeypatch.setattr(
        scraper.config,
        "ARCHETYPE_CONFIGS",
        {
            "software_tpm": {
                "provider": "linkedin",
                "location": "Canada",
                "filter_profile": "software_tpm_v1",
                "search_queries": ["Technical Program Manager"],
            }
        },
    )
    monkeypatch.setattr(
        scraper.config,
        "MAX_JOBS_PER_SEARCH",
        {"linkedin": 10, "careers_future": 10},
    )
    monkeypatch.setattr(
        scraper.config,
        "CAREERS_FUTURE_SEARCH_QUERIES",
        ["Program Manager"],
    )
    monkeypatch.setattr(
        scraper,
        "process_linkedin_query",
        lambda **kwargs: linkedin_jobs,
    )
    monkeypatch.setattr(
        scraper,
        "process_careers_future_query",
        lambda query, limit=None: careers_future_jobs,
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "save_linkedin_jobs_canonicalized",
        lambda jobs: linkedin_save_calls.append(jobs) or ["linkedin-canonical-id"],
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "save_jobs_canonicalized",
        lambda jobs: generic_save_calls.append(jobs) or ["careers-future-canonical-id"],
    )

    with caplog.at_level(logging.INFO):
        saved_job_ids = scraper.main()

    assert saved_job_ids == [
        "linkedin-canonical-id",
        "careers-future-canonical-id",
    ]
    assert linkedin_save_calls == [linkedin_jobs]
    assert generic_save_calls == [careers_future_jobs]
    assert "Total new jobs saved across all queries: 2" in caplog.text
