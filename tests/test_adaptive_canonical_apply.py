import hashlib
from types import SimpleNamespace

import httpx
import pytest

import supabase_utils


RUN_ID = "00000000-0000-0000-0000-000000000001"
LEASE_TOKEN = "00000000-0000-0000-0000-000000000002"
FILTER_PROFILE = {
    "company_blocklist": [],
    "title_entry_level_blocklist": [],
    "title_blocklist": [],
    "desc_blocklist": [],
    "title_include": [],
    "description_include": [],
}


def task():
    return {
        "id": 7,
        "provider": "linkedin",
        "source_job_id": "source-1",
        "first_ingestion_run_id": RUN_ID,
        "lease_token": LEASE_TOKEN,
        "membership_provenance_revision": 0,
    }


def job():
    return {
        "job_id": "source-1",
        "provider": "linkedin",
        "company": "Example",
        "job_title": "Technical Program Manager",
        "location": "Toronto, Ontario, Canada",
        "description": "Lead technical delivery.",
        "lane": "technology_delivery",
        "search_query": "Technical Program Manager",
        "search_query_id": "q-1",
        "search_query_type": "precision",
        "search_query_language": "en",
        "search_location_scope": "canada",
        "geography_id": "CA",
        "detail_metadata_checked_at": "2026-09-04T12:00:00+00:00",
    }


def empty_context():
    return supabase_utils.CanonicalRunContext(
        candidates_by_provider={"linkedin": []},
        existing_job_ids_by_provider={"linkedin": set()},
        company_title_keys_by_provider={"linkedin": set()},
        canonical_by_source_by_provider={"linkedin": {}},
    )


def context_with_existing():
    existing = {
        "job_id": "canonical-1",
        "provider": "linkedin",
        "company": "Example",
        "job_title": "Technical Program Manager",
        "location": "Toronto, Ontario, Canada",
        "description": "Lead technical delivery.",
        "listing_instances": [{
            "job_id": "source-1",
            "posted_at": "2026-08-01",
            "scraped_at": "2026-08-01T12:00:00+00:00",
            "last_seen_at": "2026-08-01T12:00:00+00:00",
        }],
        "seen_count": 1,
        "posting_wave_count": 1,
        "repost_count": 0,
        "same_id_relist_count": 0,
        "last_seen_at": "2026-08-01T12:00:00+00:00",
    }
    return supabase_utils.CanonicalRunContext(
        candidates_by_provider={"linkedin": [existing]},
        existing_job_ids_by_provider={"linkedin": {"canonical-1", "source-1"}},
        company_title_keys_by_provider={
            "linkedin": {("example", "technical program manager")}
        },
        canonical_by_source_by_provider={
            "linkedin": {"canonical-1": "canonical-1", "source-1": "canonical-1"}
        },
    )


def test_application_is_versioned_and_contains_all_atomic_side_effects(monkeypatch):
    monkeypatch.setattr(supabase_utils.config, "ENABLE_REPOST_DEDUP", True)
    monkeypatch.setattr(supabase_utils.config, "ENABLE_LINKEDIN_RELIST_TRACKING", True)

    application = supabase_utils.build_linkedin_discovery_task_application(
        task(), job(), run_context=empty_context(), runtime_profile=FILTER_PROFILE
    )

    assert application["version"] == "linkedin-canonical-task-apply-v3"
    assert application["provider_candidate_set_revision"] == hashlib.sha256(b"").hexdigest()
    assert application["membership_provenance_revision"] == 0
    assert application["source"] == {
        "provider": "linkedin",
        "source_job_id": "source-1",
        "ingestion_run_id": RUN_ID,
    }
    assert application["canonical"]["action"] == "insert"
    assert application["canonical"]["payload"]["job_id"] == "source-1"
    assert "search_query_id" not in application["canonical"]["payload"]
    assert application["content_version"]["ingestion_run_id"] == RUN_ID
    assert len(application["memberships"]) == 1
    assert application["memberships"][0]["archetype"] == "technology_delivery"
    assert application["memberships"][0]["query_scope"]["query_id"] == "q-1"
    assert application["memberships"][0]["filter_status"] == "included"


def test_application_preserves_two_lanes_and_two_queries_in_one_lane(monkeypatch):
    monkeypatch.setattr(supabase_utils.config, "ENABLE_REPOST_DEDUP", True)
    monkeypatch.setattr(supabase_utils.config, "ENABLE_LINKEDIN_RELIST_TRACKING", True)
    provenances = [
        {
            "lane": "technology_delivery", "archetype": "technology_delivery",
            "query_id": "q-2", "query": "Program Manager", "query_type": "recall",
            "language": "en", "location_scope": "canada", "geography_id": "CA",
            "observed_at": "2026-09-04T12:02:00+00:00",
        },
        {
            "lane": "systems_platform_ops", "archetype": "systems_platform_ops",
            "query_id": "q-3", "query": "Infrastructure Manager", "query_type": "precision",
            "language": "en", "location_scope": "canada", "geography_id": "CA",
            "observed_at": "2026-09-04T12:03:00+00:00",
        },
        {
            "lane": "technology_delivery", "archetype": "technology_delivery",
            "query_id": "q-1", "query": "Technical Program Manager", "query_type": "precision",
            "language": "en", "location_scope": "canada", "geography_id": "CA",
            "observed_at": "2026-09-04T12:01:00+00:00",
        },
    ]
    multi_lane_task = {**task(), "membership_provenances": provenances + [provenances[0]]}
    runtime_profiles = {
        "technology_delivery": FILTER_PROFILE,
        "systems_platform_ops": {**FILTER_PROFILE, "company_blocklist": [r"^Example$"]},
    }

    application = supabase_utils.build_linkedin_discovery_task_application(
        multi_lane_task,
        job(),
        run_context=empty_context(),
        runtime_profiles=runtime_profiles,
    )
    reversed_application = supabase_utils.build_linkedin_discovery_task_application(
        {**multi_lane_task, "membership_provenances": list(reversed(multi_lane_task["membership_provenances"]))},
        job(),
        run_context=empty_context(),
        runtime_profiles=runtime_profiles,
    )

    assert application["memberships"] == reversed_application["memberships"]
    assert len(application["memberships"]) == 3
    assert [
        (membership["archetype"], membership["query_id"])
        for membership in application["memberships"]
    ] == [
        ("systems_platform_ops", "q-3"),
        ("technology_delivery", "q-1"),
        ("technology_delivery", "q-2"),
    ]
    assert [membership["filter_status"] for membership in application["memberships"]] == [
        "filtered", "included", "included",
    ]


def test_relist_application_carries_cas_and_evidence(monkeypatch):
    monkeypatch.setattr(supabase_utils.config, "ENABLE_REPOST_DEDUP", True)
    monkeypatch.setattr(supabase_utils.config, "ENABLE_LINKEDIN_RELIST_TRACKING", True)
    relisted = {
        **job(),
        "same_id_relist_candidate": True,
        "same_id_relist_date": "2026-09-04",
        "same_id_relist_evidence": {"reason": "new trusted date"},
        "posted_at": "2026-09-04",
        "salary_text": "$150,000",
        "recruiter_name": "Casey Recruiter",
        "detail_metadata_checked_at": "2026-09-04T12:00:00+00:00",
    }

    application = supabase_utils.build_linkedin_discovery_task_application(
        task(), relisted, run_context=context_with_existing(), runtime_profile=FILTER_PROFILE
    )

    assert application["canonical"]["action"] == "accepted_relist"
    assert application["canonical"]["canonical_job_id"] == "canonical-1"
    assert application["canonical"]["expected"] == {
        "last_seen_at": "2026-08-01T12:00:00+00:00",
        "listing_instances": context_with_existing().candidates_by_provider["linkedin"][0][
            "listing_instances"
        ],
    }
    assert application["relist"] == {
        "relisted_on": "2026-09-04",
        "observed_at": "2026-09-04T12:00:00+00:00",
        "evidence": {"reason": "new trusted date"},
    }
    assert application["canonical"]["payload"]["salary_text"] == "$150,000"
    assert application["canonical"]["payload"]["recruiter_name"] == "Casey Recruiter"


def test_candidate_set_revision_is_sorted_distinct_and_utf8_length_framed():
    candidates = [
        {"job_id": "z"},
        {"job_id": "a"},
        {"job_id": "z"},
        {"job_id": "é"},
    ]

    assert supabase_utils._provider_candidate_set_revision(candidates) == hashlib.sha256(
        "1:a1:01:z1:02:é1:0".encode("utf-8")
    ).hexdigest()


def test_candidate_set_revision_changes_when_a_candidate_changes():
    original = [{"job_id": "canonical-1", "canonical_revision": 4}]
    changed = [{"job_id": "canonical-1", "canonical_revision": 5}]

    assert supabase_utils._provider_candidate_set_revision(original) != (
        supabase_utils._provider_candidate_set_revision(changed)
    )


class RpcSequence:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def rpc(self, name, args):
        self.calls.append((name, args))
        outcome = self.outcomes.pop(0)

        class Execute:
            def execute(self):
                if isinstance(outcome, Exception):
                    raise outcome
                return SimpleNamespace(data=outcome)

        return Execute()


def test_ambiguous_transport_retry_reuses_the_exact_application(monkeypatch):
    monkeypatch.setattr(supabase_utils.config, "ENABLE_REPOST_DEDUP", True)
    monkeypatch.setattr(supabase_utils.config, "ENABLE_LINKEDIN_RELIST_TRACKING", True)
    db = RpcSequence([
        httpx.ReadTimeout("timed out"),
        {"outcome": "applied", "canonical_job_id": "source-1", "canonical_revision": 0},
    ])

    result = supabase_utils.apply_linkedin_discovery_task_canonical(
        task(),
        "worker-1",
        job(),
        run_context=empty_context(),
        runtime_profile=FILTER_PROFILE,
        db=db,
    )

    assert result == "source-1"
    assert len(db.calls) == 2
    assert db.calls[0][1]["p_application"] == db.calls[1][1]["p_application"]


def test_ambiguous_transport_failure_is_typed_and_does_not_mutate_context(monkeypatch):
    monkeypatch.setattr(supabase_utils.config, "ENABLE_REPOST_DEDUP", True)
    monkeypatch.setattr(supabase_utils.config, "ENABLE_LINKEDIN_RELIST_TRACKING", True)
    context = empty_context()
    db = RpcSequence([httpx.ReadTimeout("first"), httpx.ReadTimeout("second")])

    with pytest.raises(supabase_utils.CanonicalTaskApplyAmbiguous):
        supabase_utils.apply_linkedin_discovery_task_canonical(
            task(),
            "worker-1",
            job(),
            run_context=context,
            runtime_profile=FILTER_PROFILE,
            db=db,
        )

    assert context.candidates_by_provider["linkedin"] == []


def test_stale_plan_invalidates_snapshot_heartbeats_and_replans(monkeypatch):
    monkeypatch.setattr(supabase_utils.config, "ENABLE_REPOST_DEDUP", True)
    monkeypatch.setattr(supabase_utils.config, "ENABLE_LINKEDIN_RELIST_TRACKING", False)
    monkeypatch.setattr(supabase_utils, "get_canonical_candidates", lambda provider: [])
    db = RpcSequence([
        {"outcome": "stale_plan", "canonical_job_id": "canonical-1"},
        "2026-09-04T12:10:00+00:00",
        {"outcome": "applied", "canonical_job_id": "source-1", "canonical_revision": 0},
    ])

    result = supabase_utils.apply_linkedin_discovery_task_canonical(
        task(),
        "worker-1",
        job(),
        run_context=context_with_existing(),
        runtime_profile=FILTER_PROFILE,
        db=db,
    )

    assert result == "source-1"
    assert [name for name, _args in db.calls] == [
        "apply_linkedin_discovery_task_canonical",
        "heartbeat_linkedin_discovery_task",
        "apply_linkedin_discovery_task_canonical",
    ]
    assert db.calls[0][1]["p_application"]["canonical"]["action"] == "update"
    assert db.calls[2][1]["p_application"]["canonical"]["action"] == "insert"


def test_stale_provenance_replan_uses_the_locked_task_snapshot(monkeypatch):
    monkeypatch.setattr(supabase_utils.config, "ENABLE_REPOST_DEDUP", True)
    monkeypatch.setattr(supabase_utils.config, "ENABLE_LINKEDIN_RELIST_TRACKING", False)
    monkeypatch.setattr(supabase_utils, "get_canonical_candidates", lambda provider: [])
    first_provenance = {
        "lane": "technology_delivery",
        "archetype": "technology_delivery",
        "query_id": "q-1",
        "query": "Technical Program Manager",
        "query_type": "precision",
        "language": "en",
        "location_scope": "canada",
        "geography_id": "CA",
        "observed_at": "2026-09-04T12:01:00+00:00",
    }
    second_provenance = {
        "lane": "systems_platform_ops",
        "archetype": "systems_platform_ops",
        "query_id": "q-2",
        "query": "Infrastructure Manager",
        "query_type": "precision",
        "language": "en",
        "location_scope": "canada",
        "geography_id": "CA",
        "observed_at": "2026-09-04T12:03:00+00:00",
    }
    db = RpcSequence([
        {
            "outcome": "stale_plan",
            "canonical_job_id": None,
            "task_membership_provenances": [first_provenance, second_provenance],
            "task_membership_provenance_revision": 1,
        },
        "2026-09-04T12:10:00+00:00",
        {"outcome": "applied", "canonical_job_id": "source-1", "canonical_revision": 0},
    ])

    result = supabase_utils.apply_linkedin_discovery_task_canonical(
        task(),
        "worker-1",
        job(),
        run_context=empty_context(),
        runtime_profiles={
            "technology_delivery": FILTER_PROFILE,
            "systems_platform_ops": FILTER_PROFILE,
        },
        db=db,
    )

    assert result == "source-1"
    replanned = db.calls[2][1]["p_application"]
    assert replanned["membership_provenance_revision"] == 1
    assert [membership["archetype"] for membership in replanned["memberships"]] == [
        "systems_platform_ops", "technology_delivery"
    ]


@pytest.mark.parametrize(
    "operation,args",
    [
        (supabase_utils.heartbeat_linkedin_discovery_task, (7, "worker-1", LEASE_TOKEN)),
        (
            supabase_utils.transition_linkedin_discovery_task,
            (7, "worker-1", LEASE_TOKEN, "failed_retryable"),
        ),
    ],
)
def test_task_lease_sqlstate_is_translated(operation, args):
    db = RpcSequence([
        RuntimeError({"code": "40001", "message": "task lease lost"}),
    ])

    with pytest.raises(supabase_utils.CanonicalTaskLeaseLost):
        operation(*args, db=db)


def test_stale_replan_heartbeat_lease_loss_is_typed(monkeypatch):
    monkeypatch.setattr(supabase_utils.config, "ENABLE_REPOST_DEDUP", True)
    monkeypatch.setattr(supabase_utils.config, "ENABLE_LINKEDIN_RELIST_TRACKING", False)
    db = RpcSequence([
        {"outcome": "stale_plan", "canonical_job_id": "source-1"},
        RuntimeError({"code": "40001", "message": "task lease lost"}),
    ])

    with pytest.raises(supabase_utils.CanonicalTaskLeaseLost):
        supabase_utils.apply_linkedin_discovery_task_canonical(
            task(),
            "worker-1",
            job(),
            run_context=empty_context(),
            runtime_profile=FILTER_PROFILE,
            db=db,
        )
