from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

import linkedin_discovery
import supabase_utils
from linkedin_source_policy import (
    ConsumedGrant,
    LinkedInCircuitOpen,
    LinkedInRequestDeadlineExceeded,
)


def settings(**options):
    return SimpleNamespace(
        max_pages_per_query=3,
        max_jobs_per_query=25,
        options=options,
    )


def test_adaptive_options_reject_non_strict_integers():
    with pytest.raises(linkedin_discovery.DiscoveryError, match="must be an integer"):
        linkedin_discovery.adaptive_options(settings(min_pages_per_query="3"))

    with pytest.raises(linkedin_discovery.DiscoveryError, match="must be an integer"):
        linkedin_discovery.adaptive_options(settings(max_detail_tasks_per_run=True))


def test_adaptive_runtime_budget_fits_hourly_workflow():
    options = linkedin_discovery.adaptive_options(settings())

    assert options["search_runtime"] == 1_620
    assert options["detail_runtime"] == 300
    assert options["search_runtime"] + options["detail_runtime"] <= 1_920

    with pytest.raises(linkedin_discovery.DiscoveryError, match="runtime budgets"):
        linkedin_discovery.adaptive_options(settings(
            max_search_runtime_seconds=1_620,
            max_detail_runtime_seconds=301,
        ))


def test_adaptive_retry_after_accepts_http_dates():
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=30)
    response = SimpleNamespace(headers={"Retry-After": format_datetime(retry_at, usegmt=True)})

    assert 28 <= linkedin_discovery._retry_after_seconds(response) <= 30


def test_search_classifier_accepts_exact_linkedin_empty_results_body():
    response = SimpleNamespace(
        status_code=200,
        url="https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search",
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=b"<!DOCTYPE html>\n\n<!---->  ",
        text="<!DOCTYPE html>\n\n<!---->  ",
    )

    kind, _soup, elements = linkedin_discovery.classify_search_response(response)

    assert kind == "no_results"
    assert elements == []


def test_search_classifier_rejects_unrecognized_empty_html():
    response = SimpleNamespace(
        status_code=200,
        url="https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search",
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=b"<!DOCTYPE html><html></html>",
        text="<!DOCTYPE html><html></html>",
    )

    with pytest.raises(
        linkedin_discovery.DiscoveryError,
        match="LinkedIn returned unrecognized zero-card HTML",
    ):
        linkedin_discovery.classify_search_response(response)


def test_detail_drain_delegates_atomic_task_application(monkeypatch):
    transitions = []
    task = {
        "id": 7,
        "source_job_id": "source-1",
        "search_card": {},
        "provenance": {"lane": "technology_delivery"},
        "first_ingestion_run_id": "run-1",
        "lease_token": "lease-1",
    }
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "claim_linkedin_discovery_tasks",
        lambda *_args, **_kwargs: [task],
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "transition_linkedin_discovery_task",
        lambda *args, **kwargs: transitions.append((args, kwargs)),
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "heartbeat_linkedin_discovery_task",
        lambda *_args, **_kwargs: "2026-09-04T12:10:00Z",
    )

    result = linkedin_discovery._drain_tasks(
        4,
        1,
        user_agent="ua",
        detail_fetch=lambda *_args, **_kwargs: ({"job_id": "source-1"}, {}),
        save_details=lambda received_task, _worker_id, job: (
            "canonical-1"
            if received_task is task and job["job_id"] == "source-1"
            else None
        ),
    )

    assert result == ["canonical-1"]
    assert transitions == []


def test_detail_drain_propagates_source_circuit(monkeypatch):
    task = {
        "id": 7,
        "source_job_id": "source-1",
        "search_card": {},
        "provenance": {},
        "first_ingestion_run_id": "run-1",
        "lease_token": "lease-1",
    }
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "claim_linkedin_discovery_tasks",
        lambda *_args, **_kwargs: [task],
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "transition_linkedin_discovery_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("circuit failures must not be downgraded")
        ),
    )

    with pytest.raises(LinkedInCircuitOpen):
        linkedin_discovery._drain_tasks(
            4,
            1,
            user_agent="ua",
            detail_fetch=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                LinkedInCircuitOpen("blocked")
            ),
            save_details=lambda *_args: None,
        )


def test_detail_drain_does_not_transition_ambiguous_atomic_application(monkeypatch):
    task = {
        "id": 7,
        "source_job_id": "source-1",
        "search_card": {},
        "provenance": {"lane": "technology_delivery"},
        "first_ingestion_run_id": "run-1",
        "lease_token": "lease-1",
    }
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "claim_linkedin_discovery_tasks",
        lambda *_args, **_kwargs: [task],
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "heartbeat_linkedin_discovery_task",
        lambda *_args, **_kwargs: "2026-09-04T12:10:00Z",
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "transition_linkedin_discovery_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ambiguous publication must not be transitioned")
        ),
    )

    with pytest.raises(supabase_utils.CanonicalTaskApplyAmbiguous):
        linkedin_discovery._drain_tasks(
            4,
            1,
            user_agent="ua",
            detail_fetch=lambda *_args, **_kwargs: ({"job_id": "source-1"}, {}),
            save_details=lambda *_args: (_ for _ in ()).throw(
                supabase_utils.CanonicalTaskApplyAmbiguous("timeout")
            ),
        )


def test_detail_drain_does_not_retry_transition_after_cleanup_lease_loss(monkeypatch):
    task = {
        "id": 7,
        "source_job_id": "source-1",
        "search_card": {},
        "provenance": {"lane": "technology_delivery"},
        "first_ingestion_run_id": "run-1",
        "lease_token": "lease-1",
    }
    transitions = []
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "claim_linkedin_discovery_tasks",
        lambda *_args, **_kwargs: [task],
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "heartbeat_linkedin_discovery_task",
        lambda *_args, **_kwargs: "2026-09-04T12:10:00Z",
    )

    def lose_lease(*args, **_kwargs):
        transitions.append(args)
        raise supabase_utils.CanonicalTaskLeaseLost("lost")

    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "transition_linkedin_discovery_task",
        lose_lease,
    )

    result = linkedin_discovery._drain_tasks(
        4,
        1,
        user_agent="ua",
        detail_fetch=lambda *_args, **_kwargs: ({"job_id": "source-1"}, {}),
        save_details=lambda *_args: (_ for _ in ()).throw(ValueError("bad plan")),
    )

    assert result == []
    assert len(transitions) == 1


def test_detail_drain_preserves_processing_error_when_cleanup_fails(monkeypatch):
    task = {
        "id": 7,
        "source_job_id": "source-1",
        "search_card": {},
        "provenance": {"lane": "technology_delivery"},
        "first_ingestion_run_id": "run-1",
        "lease_token": "lease-1",
    }
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "claim_linkedin_discovery_tasks",
        lambda *_args, **_kwargs: [task],
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "heartbeat_linkedin_discovery_task",
        lambda *_args, **_kwargs: "2026-09-04T12:10:00Z",
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "transition_linkedin_discovery_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    with pytest.raises(ValueError, match="bad plan") as raised:
        linkedin_discovery._drain_tasks(
            4,
            1,
            user_agent="ua",
            detail_fetch=lambda *_args, **_kwargs: ({"job_id": "source-1"}, {}),
            save_details=lambda *_args: (_ for _ in ()).throw(ValueError("bad plan")),
        )

    assert isinstance(raised.value.__cause__, RuntimeError)


def test_page_window_is_computed_after_durable_grant(monkeypatch):
    started_at = datetime(2026, 9, 4, 12, 0, 1, 250000, tzinfo=timezone.utc)

    class Gate:
        def acquire(self, *_args, **_kwargs):
            return ConsumedGrant("grant-1", started_at)

        def finish(self, *_args):
            return None

    response = SimpleNamespace(
        status_code=200,
        url="https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search",
        headers={"Content-Type": "text/html"},
        text="<ul><li>card</li></ul>",
        content=b"<ul><li>card</li></ul>",
    )
    monkeypatch.setattr(linkedin_discovery.requests, "get", lambda *_args, **_kwargs: response)
    scope = {
        "scope_key": "scope-1",
        "query": "TPM",
        "location": "Canada",
        "job_type": "F",
        "work_types": "2",
        "geo_id": None,
        "source_window_earliest_at": "2026-09-04T11:00:00+00:00",
    }

    page = linkedin_discovery._request_page(
        scope,
        1,
        user_agent="ua",
        gate=Gate(),
        parse_cards=lambda _elements: [{"job_id": "123"}],
        physical_attempts=[0],
        physical_limit=1,
    )

    earliest = datetime.fromisoformat(page["source_window_earliest_at"])
    manifest_earliest = datetime.fromisoformat(scope["source_window_earliest_at"])
    assert earliest <= manifest_earliest
    assert page["lookback_seconds"] == 3602


def test_transient_search_failure_preserves_resumable_cycle(monkeypatch):
    class Gate:
        def acquire(self, *_args, **_kwargs):
            return ConsumedGrant(
                "grant-1", datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
            )

        def finish(self, *_args):
            return None

    response = SimpleNamespace(
        status_code=503,
        url="https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search",
        headers={"Content-Type": "text/html"},
        text="unavailable",
        content=b"unavailable",
    )
    monkeypatch.setattr(linkedin_discovery.config, "MAX_RETRIES", 0)
    monkeypatch.setattr(
        linkedin_discovery.requests, "get", lambda *_args, **_kwargs: response
    )

    with pytest.raises(linkedin_discovery.RetryableDiscoveryInterruption):
        linkedin_discovery._request_page(
            {
                "scope_key": "scope-1",
                "query": "TPM",
                "location": "Canada",
                "job_type": "F",
                "work_types": "1,2,3",
                "geo_id": 101174742,
                "source_window_earliest_at": "2026-09-04T11:00:00+00:00",
            },
            1,
            user_agent="ua",
            gate=Gate(),
            parse_cards=lambda _elements: [],
            physical_attempts=[0],
            physical_limit=1,
        )


def test_search_grant_deadline_preserves_resumable_cycle():
    class Gate:
        def acquire(self, *_args, **_kwargs):
            raise LinkedInRequestDeadlineExceeded("deadline elapsed")

    with pytest.raises(
        linkedin_discovery.RetryableDiscoveryInterruption,
        match="deadline elapsed",
    ):
        linkedin_discovery._request_page(
            {
                "scope_key": "scope-1",
                "query": "TPM",
                "location": "Canada",
                "job_type": "F",
                "work_types": "1,2,3",
                "geo_id": None,
                "source_window_earliest_at": "2026-09-04T11:00:00+00:00",
            },
            1,
            user_agent="ua",
            gate=Gate(),
            parse_cards=lambda _elements: [],
            physical_attempts=[0],
            physical_limit=1,
        )


def test_scope_window_uses_per_scope_success_and_persisted_depth(monkeypatch):
    execution = SimpleNamespace(
        lane=SimpleNamespace(archetype="technology_delivery"),
        query=SimpleNamespace(
            query="TPM",
            query_type=SimpleNamespace(value="precision"),
            language="en",
            query_id="q1",
        ),
        geography=SimpleNamespace(
            location="Canada",
            location_scope=SimpleNamespace(value="canada"),
            geography_id="CA",
            geo_id=101174742,
        ),
    )
    prior_success = datetime.now(timezone.utc) - timedelta(hours=10)
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "prepare_linkedin_discovery_scope_state",
        lambda keys, _floor: {
            "states": {keys[0]: {
                "last_operational_success_at": prior_success.isoformat(),
                "recommended_pages": 5,
            }},
            "debt": {},
        },
    )
    configuration = SimpleNamespace(
        settings=SimpleNamespace(lookback_days=2)
    )
    options = linkedin_discovery.adaptive_options(settings())

    manifest = linkedin_discovery._scope_manifest(
        configuration, [execution], options
    )[0]

    earliest = datetime.fromisoformat(manifest["source_window_earliest_at"])
    latest = datetime.fromisoformat(manifest["source_window_latest_at"])
    assert 15 <= (latest - earliest).total_seconds() / 3600 <= 17
    assert manifest["target_pages"] == 5


def test_pending_debt_uses_hard_depth_and_original_window(monkeypatch):
    execution = SimpleNamespace(
        lane=SimpleNamespace(archetype="technology_delivery"),
        query=SimpleNamespace(
            query="TPM",
            query_type=SimpleNamespace(value="precision"),
            language="en",
            query_id="q1",
        ),
        geography=SimpleNamespace(
            location="Canada",
            location_scope=SimpleNamespace(value="canada"),
            geography_id="CA",
            geo_id=None,
        ),
    )
    debt_earliest = datetime.now(timezone.utc) - timedelta(hours=72)
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "prepare_linkedin_discovery_scope_state",
        lambda keys, _floor: {
            "states": {},
            "debt": {
                keys[0]: {"source_window_earliest_at": debt_earliest.isoformat()}
            },
        },
    )
    options = linkedin_discovery.adaptive_options(settings(hard_max_pages_per_query=20))

    manifest = linkedin_discovery._scope_manifest(
        SimpleNamespace(settings=SimpleNamespace(lookback_days=2)),
        [execution],
        options,
    )[0]

    assert manifest["target_pages"] == 20
    assert datetime.fromisoformat(manifest["source_window_earliest_at"]) <= debt_earliest


def test_long_gap_uses_recovery_cap_and_records_only_the_expired_boundary(monkeypatch):
    execution = SimpleNamespace(
        lane=SimpleNamespace(archetype="technology_delivery"),
        query=SimpleNamespace(
            query="TPM", query_type=SimpleNamespace(value="precision"),
            language="en", query_id="q1",
        ),
        geography=SimpleNamespace(
            location="Canada", location_scope=SimpleNamespace(value="canada"),
            geography_id="CA", geo_id=None,
        ),
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "prepare_linkedin_discovery_scope_state",
        lambda keys, _floor: {
            "states": {keys[0]: {
                "last_operational_success_at": (
                    datetime.now(timezone.utc) - timedelta(hours=200)
                ).isoformat(),
                "recommended_pages": 6,
            }},
            "debt": {},
        },
    )

    manifest = linkedin_discovery._scope_manifest(
        SimpleNamespace(settings=SimpleNamespace(lookback_days=2)),
        [execution],
        linkedin_discovery.adaptive_options(settings()),
    )[0]

    assert manifest["truncated_window_earliest_at"] is None
    assert manifest["truncated_window_latest_at"] is None
    assert manifest["expired_window_earliest_at"] is not None
    assert manifest["expired_window_latest_at"] is not None


def test_recoverable_gap_is_not_artificially_truncated(monkeypatch):
    execution = SimpleNamespace(
        lane=SimpleNamespace(archetype="technology_delivery"),
        query=SimpleNamespace(
            query="TPM", query_type=SimpleNamespace(value="precision"),
            language="en", query_id="q1",
        ),
        geography=SimpleNamespace(
            location="Canada", location_scope=SimpleNamespace(value="canada"),
            geography_id="CA", geo_id=None,
        ),
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "prepare_linkedin_discovery_scope_state",
        lambda keys, _floor: {
            "states": {keys[0]: {
                "last_operational_success_at": (
                    datetime.now(timezone.utc) - timedelta(hours=30)
                ).isoformat(),
                "recommended_pages": 6,
            }},
            "debt": {},
        },
    )

    manifest = linkedin_discovery._scope_manifest(
        SimpleNamespace(settings=SimpleNamespace(lookback_days=2)),
        [execution],
        linkedin_discovery.adaptive_options(settings()),
    )[0]

    covered_hours = (
        datetime.fromisoformat(manifest["source_window_latest_at"])
        - datetime.fromisoformat(manifest["source_window_earliest_at"])
    ).total_seconds() / 3600
    assert covered_hours >= 36
    assert manifest["truncated_window_earliest_at"] is None
    assert manifest["expired_window_earliest_at"] is None


def test_cycle_replay_uses_persisted_window(monkeypatch):
    execution = SimpleNamespace(
        lane=SimpleNamespace(archetype="technology_delivery"),
        query=SimpleNamespace(
            query="TPM", query_type=SimpleNamespace(value="precision"),
            language="en", query_id="q1",
        ),
        geography=SimpleNamespace(
            location="Canada", location_scope=SimpleNamespace(value="canada"),
            geography_id="CA", geo_id=None,
        ),
    )
    configuration = SimpleNamespace(
        revision=1,
        version="v1",
        settings=SimpleNamespace(
            lookback_days=2,
            max_pages_per_query=3,
            max_jobs_per_query=25,
            options={},
        ),
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "get_resumable_linkedin_discovery_cycle",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "prepare_linkedin_discovery_scope_state",
        lambda _keys, _floor: {"states": {}, "debt": {}},
    )
    monkeypatch.setattr(
        linkedin_discovery, "configuration_hash", lambda _configuration: "a" * 64,
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils, "create_linkedin_discovery_cycle",
        lambda **kwargs: {
            "cycle_id": 3,
            "discovery_sequence": 4,
            "search_status": "sealed",
            "scopes": [{
                "scope_key": kwargs["scopes"][0]["scope_key"],
                "ingestion_run_id": "run-1",
                "next_page": 2,
                "status": "complete",
                "query_scope": "persisted-scope",
                "request_anchor_at": "2026-09-01T12:00:00+00:00",
                "source_window_earliest_at": "2026-09-01T06:00:00+00:00",
                "source_window_latest_at": "2026-09-01T12:00:00+00:00",
                "minimum_pages": 1,
                "target_pages": 3,
            }],
        },
    )
    monkeypatch.setattr(
        linkedin_discovery.supabase_utils,
        "resolve_eligible_failed_linkedin_discovery_cycles",
        lambda _cycle_id: 0,
    )
    monkeypatch.setattr(linkedin_discovery, "_drain_tasks", lambda *_args, **_kwargs: [])

    result = linkedin_discovery.run_discovery(
        configuration,
        [execution],
        parse_cards=lambda _elements: [],
        detail_fetch=lambda *_args, **_kwargs: None,
        save_details=lambda *_args: None,
        partial=False,
    )

    assert result.cycle_id == 3
