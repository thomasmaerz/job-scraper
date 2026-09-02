import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import scraper
import supabase_utils
from test_scrape_configuration import rpc_payload
from scrape_configuration import parse_scrape_configuration


def configuration():
    payload = rpc_payload(locations=["canada", "usa"])
    payload["settings"]["lookback_days"] = 2
    payload["settings"]["max_jobs_per_query"] = 7
    payload["settings"]["max_pages_per_query"] = 3
    payload["settings"]["request_delay_ms"] = 500
    # This assertion verifies deterministic execution ordering. Parallel worker
    # behavior and profile isolation are covered separately below.
    payload["settings"]["concurrent_queries"] = 1
    return parse_scrape_configuration(payload)


def test_run_configuration_passes_exact_settings_and_query_provenance(monkeypatch):
    calls = []
    monkeypatch.setattr(scraper.supabase_utils, "get_last_successful_scrape_at", lambda: None)
    monkeypatch.setattr(scraper, "process_linkedin_query", lambda **kwargs: calls.append(kwargs) or [])

    scraper._run_database_configured_linkedin(configuration(), [])

    assert [(call["query_kind"], call["location_scope"], call["geography_id"]) for call in calls] == [
        ("precision", "canada", "CA"), ("precision", "usa", "US"),
        ("recall", "canada", "CA"), ("recall", "usa", "US"),
    ]
    assert all(call["archetype"] == "technology_delivery" for call in calls)
    assert all(call["limit"] == 7 and call["max_start"] == 50 for call in calls)
    assert all(call["request_delay_ms"] == 500 for call in calls)
    assert all(call["posting_date_filter"] == "r172800" for call in calls)
    assert calls[0]["query_id"].startswith("en:precision:10:")
    assert all(call["query_language"] == "en" for call in calls)


def test_configured_queries_use_bounded_workers_and_isolated_runtime_profiles(monkeypatch):
    payload = rpc_payload(locations=["canada"])
    payload["settings"]["concurrent_queries"] = 2
    enabled_lanes = {"technology_delivery", "systems_platform_ops"}
    for lane in payload["lanes"]:
        lane["enabled"] = lane["archetype"] in enabled_lanes
    configured = parse_scrape_configuration(payload)
    original_configs = deepcopy(scraper.config.ARCHETYPE_CONFIGS)
    worker_counts = []
    fetch_calls = []
    persisted = []

    class RecordingExecutor:
        def __init__(self, max_workers):
            worker_counts.append(max_workers)

        def map(self, function, values):
            return [function(value) for value in values]

        def shutdown(self, wait=True):
            assert wait is True

    def process(**kwargs):
        fetch_calls.append(kwargs)
        return [{
            "job_id": f"source-{kwargs['archetype']}",
            "archetype": kwargs["archetype"],
            "filter_profile": kwargs["runtime_profile"]["filter_profile"],
            "query_scope": kwargs["query_id"],
        }]

    monkeypatch.setattr(scraper, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(scraper, "process_linkedin_query", process)
    monkeypatch.setattr(scraper.supabase_utils, "get_last_successful_scrape_at", lambda: None)
    monkeypatch.setattr(
        scraper.supabase_utils,
        "save_linkedin_jobs_canonicalized_with_mapping",
        lambda jobs, run_context=None: supabase_utils.CanonicalSaveResult(
            canonical_ids=[f"canonical-{jobs[0]['archetype']}"],
            canonical_by_source={jobs[0]["job_id"]: f"canonical-{jobs[0]['archetype']}"},
            canonical_ids_by_input=[f"canonical-{jobs[0]['archetype']}"],
        ),
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "persist_lane_filter_state",
        lambda canonical_id, lane, job, runtime_profile: persisted.append(
            (canonical_id, lane, job["filter_profile"], runtime_profile["filter_profile"])
        ),
    )

    scraper._run_database_configured_linkedin(configured, [])

    assert worker_counts == [2]
    assert scraper.config.ARCHETYPE_CONFIGS == original_configs
    assert [call["archetype"] for call in fetch_calls] == [
        "technology_delivery", "technology_delivery",
        "systems_platform_ops", "systems_platform_ops",
    ]
    assert [call["runtime_profile"]["filter_profile"] for call in fetch_calls] == [
        "technology_delivery_v1", "technology_delivery_v1",
        "systems_platform_ops_v1", "systems_platform_ops_v1",
    ]
    assert persisted == [
        ("canonical-technology_delivery", "technology_delivery", "technology_delivery_v1", "technology_delivery_v1"),
        ("canonical-technology_delivery", "technology_delivery", "technology_delivery_v1", "technology_delivery_v1"),
        ("canonical-systems_platform_ops", "systems_platform_ops", "systems_platform_ops_v1", "systems_platform_ops_v1"),
        ("canonical-systems_platform_ops", "systems_platform_ops", "systems_platform_ops_v1", "systems_platform_ops_v1"),
    ]


def test_concurrent_queries_one_keeps_serial_executor_free_path(monkeypatch):
    configured = configuration().model_copy(update={
        "settings": configuration().settings.model_copy(update={"concurrent_queries": 1}),
    })
    execution = scraper.build_search_executions(configured)[0]
    monkeypatch.setattr(scraper, "build_search_executions", lambda _: [execution])
    monkeypatch.setattr(scraper.supabase_utils, "get_last_successful_scrape_at", lambda: None)
    monkeypatch.setattr(scraper, "process_linkedin_query", lambda **kwargs: [])
    monkeypatch.setattr(
        scraper,
        "ThreadPoolExecutor",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("serial path created an executor")),
    )

    scraper._run_database_configured_linkedin(configured, [])


def test_configured_save_uses_per_input_canonical_mapping_for_filter_state(monkeypatch):
    config = configuration()
    execution = scraper.build_search_executions(config)[0]
    jobs = [
        {"job_id": "source-z", "job_title": "First"},
        {"job_id": "source-a", "job_title": "Second"},
        {"job_id": "source-z", "job_title": "Duplicate source"},
    ]
    calls = []
    monkeypatch.setattr(scraper.supabase_utils, "get_last_successful_scrape_at", lambda: None)
    monkeypatch.setattr(scraper, "build_search_executions", lambda _: [execution])
    monkeypatch.setattr(scraper, "process_linkedin_query", lambda **kwargs: jobs)
    monkeypatch.setattr(
        scraper.supabase_utils,
        "save_linkedin_jobs_canonicalized_with_mapping",
        lambda value, run_context=None: supabase_utils.CanonicalSaveResult(
            canonical_ids=["canonical-1"],
            canonical_by_source={"source-z": "canonical-1", "source-a": "canonical-1"},
            canonical_ids_by_input=["canonical-1", "canonical-1", "canonical-1"],
        ),
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "persist_lane_filter_state",
        lambda canonical_id, lane, job, **kwargs: calls.append(
            (canonical_id, job["job_id"], job["job_title"])
        ),
    )

    saved = []
    scraper._run_database_configured_linkedin(config, saved)

    assert saved == ["canonical-1"]
    assert calls == [
        ("canonical-1", "source-z", "First"),
        ("canonical-1", "source-a", "Second"),
        ("canonical-1", "source-z", "Duplicate source"),
    ]


def test_configured_run_restores_process_global_dedup_setting(monkeypatch):
    config = configuration()
    config = config.model_copy(update={
        "settings": config.settings.model_copy(update={"deduplicate_jobs": False}),
    })
    monkeypatch.setattr(scraper.config, "ENABLE_REPOST_DEDUP", True)
    monkeypatch.setattr(scraper.supabase_utils, "get_last_successful_scrape_at", lambda: None)
    monkeypatch.setattr(scraper, "build_search_executions", lambda _: ())

    scraper._run_database_configured_linkedin(config, [])


def test_production_query_function_serializes_shared_supabase_client(monkeypatch):
    payload = rpc_payload(locations=["canada"])
    payload["settings"]["concurrent_queries"] = 3
    config = parse_scrape_configuration(payload)
    monkeypatch.setattr(scraper.supabase_utils, "get_last_successful_scrape_at", lambda: None)
    calls = []
    fake_process = lambda **kwargs: calls.append(kwargs) or []
    fake_process.__module__ = scraper.__name__
    monkeypatch.setattr(scraper, "process_linkedin_query", fake_process)
    monkeypatch.setattr(
        scraper,
        "ThreadPoolExecutor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("shared Supabase client must serialize")),
    )

    scraper._run_database_configured_linkedin(config, [])
    assert calls

    assert scraper.config.ENABLE_REPOST_DEDUP is True


def test_configured_batches_share_one_write_through_canonical_snapshot(monkeypatch):
    configured = configuration()
    candidate_loads = []
    inserted = []
    updated = []
    contexts = []

    jobs = [
        {
            "job_id": "source-1",
            "provider": "linkedin",
            "company": "Acme",
            "job_title": "Technical Program Manager",
            "location": "Toronto, Ontario, Canada",
            "description": "Own software delivery and cross-functional execution. " * 8,
            "archetype": "technology_delivery",
        },
        {
            "job_id": "source-2",
            "provider": "linkedin",
            "company": "Acme",
            "job_title": "Technical Program Manager",
            "location": "Toronto, Ontario, Canada",
            "description": "Own software delivery and cross-functional execution. " * 8,
            "archetype": "technology_delivery",
        },
    ]

    def process(**kwargs):
        contexts.append(kwargs["run_context"])
        return [jobs[len(contexts) - 1]] if len(contexts) <= 2 else []

    class UpdateQuery:
        def update(self, payload):
            updated.append(payload)
            return self

        def eq(self, *_args):
            return self

        def is_(self, *_args):
            return self

        def execute(self):
            return SimpleNamespace(data=[{"job_id": "source-1"}])

    monkeypatch.setattr(scraper.config, "ENABLE_LINKEDIN_RELIST_TRACKING", False)
    monkeypatch.setattr(scraper.supabase_utils, "get_last_successful_scrape_at", lambda: None)
    monkeypatch.setattr(scraper, "process_linkedin_query", process)
    monkeypatch.setattr(
        scraper.supabase_utils,
        "get_canonical_candidates",
        lambda provider: candidate_loads.append(provider) or [],
    )
    monkeypatch.setattr(
        scraper.supabase_utils,
        "save_job_to_supabase",
        lambda payload: inserted.append(payload) or payload["job_id"],
    )
    monkeypatch.setattr(scraper.supabase_utils, "upsert_job_archetype_membership", lambda *_args: None)
    monkeypatch.setattr(scraper.supabase_utils, "persist_lane_filter_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scraper.supabase_utils,
        "supabase",
        SimpleNamespace(table=lambda _name: UpdateQuery()),
    )

    saved = []
    scraper._run_database_configured_linkedin(configured, saved)

    assert candidate_loads == ["linkedin"]
    assert len({id(context) for context in contexts}) == 1
    assert len(inserted) == 1
    assert len(updated) == 1
    assert updated[0]["latest_job_id"] == "source-2"
    assert saved[:2] == ["source-1", "source-1"]


def test_process_query_persists_membership_for_already_known_canonical_job(monkeypatch):
    monkeypatch.setattr(scraper.config, "ENABLE_LINKEDIN_RELIST_TRACKING", False)
    memberships = []
    with patch.object(scraper, "_fetch_linkedin_job_ids", return_value=["source-123"]), \
         patch.object(scraper.supabase_utils, "get_existing_jobs_from_supabase", return_value=({"source-123"}, set())), \
         patch.object(scraper.supabase_utils, "get_incomplete_linkedin_metadata_ids", return_value=set()), \
         patch.object(scraper.supabase_utils, "get_canonical_job_ids_for_sources", return_value={"source-123": "canonical-1"}), \
         patch.object(scraper.supabase_utils, "upsert_job_archetype_membership", side_effect=lambda job_id, job: memberships.append((job_id, job))):
        jobs = scraper.process_linkedin_query(
            "Technical Project Manager",
            "Canada",
            archetype="software_tpm",
            query_id="en:precision:10:abc",
            query_kind="precision",
            lane="technology_delivery",
            location_scope="canada",
            geography_id="CA",
        )

    assert jobs == []
    assert memberships[0][0] == "canonical-1"
    assert memberships[0][1]["lane"] == "technology_delivery"
    assert memberships[0][1]["search_query"] == "Technical Project Manager"


def test_process_query_query_scope_and_new_job_include_provenance(monkeypatch):
    monkeypatch.setattr(scraper.config, "ENABLE_LINKEDIN_RELIST_TRACKING", False)
    detail = {
        "job_id": "123", "job_title": "Technical Project Manager",
        "company": "Example", "description": "Own delivery.", "provider": "linkedin",
    }
    with patch.object(scraper, "_fetch_linkedin_job_ids", return_value=["123"]), \
         patch.object(scraper, "_fetch_linkedin_job_details", return_value=(detail, {})), \
         patch.object(scraper.supabase_utils, "get_existing_jobs_from_supabase", return_value=(set(), set())), \
         patch.object(scraper.supabase_utils, "get_incomplete_linkedin_metadata_ids", return_value=set()), \
         patch.object(scraper.supabase_utils, "get_canonical_job_ids_for_sources", return_value={}):
        jobs = scraper.process_linkedin_query(
            "Technical Project Manager", "Canada", archetype="software_tpm",
            query_id="en:precision:10:abc", query_kind="precision",
            query_language="en",
            lane="technology_delivery", location_scope="canada", geography_id="CA",
        )

    scope = json.loads(jobs[0]["query_scope"])
    assert scope["lane"] == "technology_delivery"
    assert scope["query_id"] == "en:precision:10:abc"
    assert scope["location_scope"] == "canada"
    assert jobs[0]["search_query_type"] == "precision"
    assert jobs[0]["search_query_language"] == "en"
    assert scope["language"] == "en"
