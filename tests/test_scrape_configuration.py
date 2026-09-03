import json
from types import SimpleNamespace

import pytest

from lane_catalog import CANONICAL_LANE_SLUGS, canonical_lane_slug
from scrape_configuration import (
    LocationRegion,
    ScrapeConfigurationError,
    build_search_executions,
    expand_location_scopes,
    load_scrape_configuration,
    parse_scrape_configuration,
)


def migration_payload(*, locations=None):
    lanes = []
    for sort_order, archetype in enumerate(CANONICAL_LANE_SLUGS):
        lanes.append({
            "archetype": archetype,
            "display_name": archetype.replace("_", " ").title(),
            "description": f"{archetype} description",
            "routing_guidance": f"Route {archetype} titles first.",
            "title_include": [f"{archetype} title"],
            "title_exclude": ["excluded title"],
            "description_include": ["positive signal"],
            "description_exclude": ["negative signal"],
            "enabled": archetype == "technology_delivery",
            "resume_profile_ready": archetype == "technology_delivery",
            "sort_order": sort_order,
            "locations": locations or ["canada"],
            "queries": [
                {
                    "query": f'"{archetype} manager"',
                    "query_type": "precision",
                    "language": "en",
                    "sort_order": 10,
                    "enabled": True,
                },
                {
                    "query": f"{archetype} AND operations",
                    "query_type": "recall",
                    "language": "en",
                    "sort_order": 20,
                    "enabled": True,
                },
            ],
        })
    # get_scraper_configuration strips query IDs/timestamps but retains FK archetype.
    return {
        "version": 1,
        "revision": 7,
        "aliases": {"software_tpm": "technology_delivery"},
        "settings": {
            "scraping_enabled": True,
            "lookback_days": 3,
            "max_jobs_per_query": 25,
            "max_pages_per_query": 4,
            "request_delay_ms": 750,
            "concurrent_queries": 3,
            "deduplicate_jobs": True,
            "fetch_descriptions": True,
            "score_jobs": False,
            "options": {"linkedin_job_type": "F"},
            "updated_at": "2026-09-01T00:00:00+00:00",
        },
        "lanes": lanes,
    }


def rpc_payload(*, locations=None):
    payload = migration_payload(locations=locations)
    # SQL removes q.id/timestamps but retains the query's FK archetype.
    for lane in payload["lanes"]:
        for query in lane["queries"]:
            query["archetype"] = lane["archetype"]
    return payload


def test_exact_migration_contract_parses_without_field_translation():
    configuration = parse_scrape_configuration(rpc_payload())

    lane = configuration.lanes[0]
    assert lane.archetype == "technology_delivery"
    assert lane.routing_guidance == "Route technology_delivery titles first."
    assert lane.title_include == ["technology_delivery title"]
    assert lane.resume_profile_ready is True
    assert lane.queries[0].query_type.value == "precision"
    assert configuration.settings.lookback_days == 3
    assert canonical_lane_slug("software_tpm") == "technology_delivery"


def test_contract_rejects_old_adapter_fields_and_unknown_fields():
    payload = rpc_payload()
    payload["settings"]["lookback_hours"] = 72

    with pytest.raises(ScrapeConfigurationError, match="lookback_hours"):
        parse_scrape_configuration(payload)


def test_resume_profile_readiness_belongs_to_lane_not_nested_query():
    payload = rpc_payload()
    payload["lanes"][0]["resume_profile_ready"] = False
    assert parse_scrape_configuration(payload).lanes[0].resume_profile_ready is False

    payload["lanes"][0]["queries"][0]["resume_profile_ready"] = True
    with pytest.raises(ScrapeConfigurationError, match="resume_profile_ready"):
        parse_scrape_configuration(payload)


def test_contract_keeps_scalar_types_strict():
    payload = rpc_payload()
    payload["settings"]["max_pages_per_query"] = "4"

    with pytest.raises(ScrapeConfigurationError, match="max_pages_per_query"):
        parse_scrape_configuration(payload)


def test_scrape_options_require_safe_global_pacing():
    payload = rpc_payload()
    payload["settings"]["options"] = {"global_request_interval_ms": 2499}
    with pytest.raises(ScrapeConfigurationError, match="global_request_interval_ms"):
        parse_scrape_configuration(payload)

    payload["settings"]["options"] = {"request_jitter_ms": -1}
    with pytest.raises(ScrapeConfigurationError, match="request_jitter_ms"):
        parse_scrape_configuration(payload)


def test_location_expansion_is_deterministic_and_independently_selectable():
    geographies = expand_location_scopes(
        [LocationRegion.canada, LocationRegion.usa, LocationRegion.eea]
    )

    assert [(item.location_scope.value, item.geography_id) for item in geographies[:4]] == [
        ("canada", "CA"), ("usa", "US"), ("eea", "AT"), ("eea", "BE"),
    ]
    assert len(geographies) == 32


def test_executions_use_per_lane_locations_and_migration_sort_order():
    configuration = parse_scrape_configuration(rpc_payload(locations=["usa", "canada"]))
    executions = build_search_executions(configuration)

    assert [(item.query.query_type.value, item.geography.geography_id) for item in executions] == [
        ("precision", "CA"), ("precision", "US"),
        ("recall", "CA"), ("recall", "US"),
    ]
    assert executions[0].query.query_id.startswith("en:precision:10:")


def test_execution_archetype_override_selects_one_enabled_lane_and_alias():
    configuration = parse_scrape_configuration(rpc_payload())
    canonical = build_search_executions(configuration, "technology_delivery")
    alias = build_search_executions(configuration, "software_tpm")

    assert canonical
    assert {item.lane.archetype for item in canonical} == {"technology_delivery"}
    assert alias == canonical


def test_execution_archetype_override_rejects_unknown_or_disabled_lane():
    configuration = parse_scrape_configuration(rpc_payload())

    with pytest.raises(ScrapeConfigurationError, match="Unknown SCRAPE_ARCHETYPE"):
        build_search_executions(configuration, "unknown_lane")
    with pytest.raises(ScrapeConfigurationError, match="is disabled"):
        build_search_executions(configuration, "network_infrastructure")


def test_enabled_lane_requires_enabled_precision_recall_and_location():
    payload = rpc_payload()
    payload["lanes"][0]["queries"][1]["enabled"] = False

    with pytest.raises(ScrapeConfigurationError, match="enabled precision and recall"):
        parse_scrape_configuration(payload)


def test_default_source_calls_actual_migration_rpc_only():
    calls = []

    class FakeRpc:
        def execute(self):
            return SimpleNamespace(data=rpc_payload())

    class FakeDb:
        def rpc(self, name):
            calls.append(name)
            return FakeRpc()

        def table(self, name):
            raise AssertionError(f"configuration must not read direct tables: {name}")

    configuration = load_scrape_configuration(db=FakeDb(), environ={})

    assert configuration.revision == 7
    assert calls == ["get_scraper_configuration"]


def test_missing_actual_rpc_fails_without_obsolete_rpc_or_table_fallback():
    class MissingRpc:
        def execute(self):
            raise RuntimeError("PGRST202 get_scraper_configuration not found")

    class FakeDb:
        def rpc(self, name):
            assert name == "get_scraper_configuration"
            return MissingRpc()

    with pytest.raises(ScrapeConfigurationError, match=r"get_scraper_configuration\(\) failed"):
        load_scrape_configuration(db=FakeDb(), environ={})


def test_explicit_env_and_file_overrides_require_complete_contract(tmp_path):
    encoded = json.dumps(rpc_payload(locations=["eea"]))
    env_configuration = load_scrape_configuration(
        db=object(),
        environ={"SCRAPE_CONFIG_SOURCE": "env", "SCRAPE_CONFIG_JSON": encoded},
    )
    path = tmp_path / "scrape.json"
    path.write_text(encoded, encoding="utf-8")
    file_configuration = load_scrape_configuration(
        db=object(),
        environ={"SCRAPE_CONFIG_SOURCE": "file", "SCRAPE_CONFIG_FILE": str(path)},
    )

    assert env_configuration.lanes[0].locations == [LocationRegion.eea]
    assert file_configuration == env_configuration


def test_override_mode_errors_are_actionable():
    with pytest.raises(ScrapeConfigurationError, match="requires non-empty SCRAPE_CONFIG_JSON"):
        load_scrape_configuration(db=object(), environ={"SCRAPE_CONFIG_SOURCE": "env"})
    with pytest.raises(ScrapeConfigurationError, match="Expected db, env, or file"):
        load_scrape_configuration(db=object(), environ={"SCRAPE_CONFIG_SOURCE": "merge"})
