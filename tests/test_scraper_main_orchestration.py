import logging
from uuid import UUID

import scraper
import supabase_utils


def test_record_scrape_success_persists_top_level_execution_scope(monkeypatch):
    execution_run_id = "8f5fe31a-6f10-4b47-a754-aac9fe83c8bb"
    writes = []

    class FakeQuery:
        def upsert(self, payload, **kwargs):
            writes.append((payload, kwargs))
            return self

        def execute(self):
            return None

    class FakeSupabase:
        def table(self, table_name):
            assert table_name == "ingestion_runs"
            return FakeQuery()

    monkeypatch.setattr(supabase_utils, "supabase", FakeSupabase())

    supabase_utils.record_scrape_success(execution_run_id)

    payload, options = writes[0]
    assert payload["id"] == execution_run_id
    assert payload["provider"] == "scraper"
    assert payload["query_scope"] == "top-level-execution"
    assert payload["status"] == "complete"
    assert payload["coverage_complete"] is True
    assert payload["started_at"] == payload["finished_at"]
    assert payload.get("search_query") is None
    assert options == {"on_conflict": "id"}


def test_main_consumes_canonical_saver_id_lists_for_all_sources(monkeypatch, caplog):
    query_run_id = "c76e1632-1a6a-42b5-8f2c-da3949988dc5"
    linkedin_jobs = [{
        "job_id": "linkedin-source-id",
        "provider": "linkedin",
        "scrape_run_id": query_run_id,
    }]
    careers_future_jobs = [
        {"job_id": "careers-future-source-id", "provider": "careers_future"}
    ]
    linkedin_save_calls = []
    generic_save_calls = []
    success_run_ids = []

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
    monkeypatch.setattr(
        scraper.supabase_utils,
        "record_scrape_success",
        success_run_ids.append,
    )

    with caplog.at_level(logging.INFO):
        saved_job_ids = scraper.main()

    assert saved_job_ids == [
        "linkedin-canonical-id",
        "careers-future-canonical-id",
    ]
    assert linkedin_save_calls == [linkedin_jobs]
    assert generic_save_calls == [careers_future_jobs]
    assert len(success_run_ids) == 1
    execution_run_id = UUID(success_run_ids[0])
    assert execution_run_id.version == 4
    assert success_run_ids[0] != query_run_id
    assert success_run_ids[0] not in saved_job_ids
    assert "Total new jobs saved across all queries: 2" in caplog.text


def test_main_uses_careers_future_saver_ids_without_renormalizing(monkeypatch):
    careers_future_jobs = [
        {"job_id": "careers-future-source-id", "provider": "careers_future"}
    ]
    canonical_ids = ["careers-future-canonical-id", "existing-canonical-id"]

    monkeypatch.setattr(scraper.config, "SCRAPING_SOURCES", {"careers_future"})
    monkeypatch.setattr(
        scraper.config,
        "MAX_JOBS_PER_SEARCH",
        {"careers_future": 10},
    )
    monkeypatch.setattr(
        scraper.config,
        "CAREERS_FUTURE_SEARCH_QUERIES",
        ["Program Manager"],
    )
    monkeypatch.setattr(
        scraper,
        "process_careers_future_query",
        lambda query, limit=None: careers_future_jobs,
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "save_jobs_canonicalized",
        lambda jobs: canonical_ids,
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "record_scrape_success",
        lambda execution_run_id: None,
    )

    def fail_if_renormalized(value):
        raise AssertionError(f"main must not renormalize saver-returned IDs: {value!r}")

    monkeypatch.setattr(
        scraper.supabase_utils,
        "normalize_job_identifier",
        fail_if_renormalized,
    )

    assert scraper.main() == canonical_ids
