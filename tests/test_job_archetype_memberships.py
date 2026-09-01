from types import SimpleNamespace

import supabase_utils


class AtomicMembershipRpc:
    """Small RPC-contract fake; union behavior belongs to the database function."""

    def __init__(self):
        self.calls = []
        self.state = {}

    def rpc(self, name, args):
        assert name == "record_job_archetype_membership"
        self.calls.append((name, args))
        owner = self

        class Execute:
            def execute(self):
                key = (args["p_job_id"], args["p_archetype"])
                owner.state.setdefault(key, set()).add(
                    tuple(sorted(args["p_query_scope"].items()))
                )
                return SimpleNamespace(data={
                    "job_id": key[0],
                    "archetype": key[1],
                    "matched_queries": [dict(item) for item in sorted(owner.state[key])],
                })

        return Execute()

    def table(self, name):
        raise AssertionError(f"membership persistence must not directly access table {name}")


def match(lane, query, query_id, scope="canada", observed_at=None):
    value = {
        "lane": lane,
        "archetype": lane,
        "search_query": query,
        "search_query_id": query_id,
        "search_query_type": "precision",
        "search_query_language": "en",
        "search_location_scope": scope,
        "geography_id": "CA" if scope == "canada" else "US",
    }
    if observed_at is not None:
        value["observed_at"] = observed_at
    return value


def test_membership_calls_atomic_rpc_with_exact_args(monkeypatch):
    db = AtomicMembershipRpc()
    monkeypatch.setattr(supabase_utils, "supabase", db)

    supabase_utils.upsert_job_archetype_membership(
        "canonical-1",
        match("technology_delivery", "Technical Project Manager", "q-1"),
    )

    assert db.calls == [("record_job_archetype_membership", {
        "p_job_id": "canonical-1",
        "p_archetype": "technology_delivery",
        "p_query_scope": {
            "lane": "technology_delivery",
            "query_id": "q-1",
            "query_type": "precision",
            "query": "Technical Project Manager",
            "language": "en",
            "location_scope": "canada",
            "geography_id": "CA",
        },
        "p_query_id": "q-1",
        "p_query": "Technical Project Manager",
        "p_query_type": "precision",
        "p_language": "en",
        "p_location_scope": "canada",
        "p_geography_id": "CA",
    })]


def test_valid_observed_timestamp_is_forwarded_and_invalid_timestamp_uses_defaults(monkeypatch):
    db = AtomicMembershipRpc()
    monkeypatch.setattr(supabase_utils, "supabase", db)

    supabase_utils.upsert_job_archetype_membership(
        "canonical-1",
        match("technology_delivery", "TPM", "q-1", observed_at="2026-09-01T12:00:00Z"),
    )
    supabase_utils.upsert_job_archetype_membership(
        "canonical-1",
        match("technology_delivery", "TPM", "q-2", observed_at="not-a-timestamp"),
    )

    first_args = db.calls[0][1]
    assert first_args["p_first_matched_at"] == "2026-09-01T12:00:00+00:00"
    assert first_args["p_last_matched_at"] == "2026-09-01T12:00:00+00:00"
    assert "p_first_matched_at" not in db.calls[1][1]
    assert "p_last_matched_at" not in db.calls[1][1]


def test_two_queries_in_one_lane_union_at_atomic_rpc_boundary(monkeypatch):
    db = AtomicMembershipRpc()
    monkeypatch.setattr(supabase_utils, "supabase", db)

    supabase_utils.upsert_job_archetype_membership(
        "canonical-1", match("technology_delivery", "Technical Project Manager", "q-1")
    )
    supabase_utils.upsert_job_archetype_membership(
        "canonical-1", match("technology_delivery", "Technical Program Manager", "q-2", "usa")
    )

    union = [dict(item) for item in db.state[("canonical-1", "technology_delivery")]]
    assert {item["query_id"] for item in union} == {"q-1", "q-2"}
    assert {item["location_scope"] for item in union} == {"canada", "usa"}


def test_one_job_matching_multiple_lanes_uses_separate_atomic_memberships(monkeypatch):
    db = AtomicMembershipRpc()
    monkeypatch.setattr(supabase_utils, "supabase", db)

    supabase_utils.upsert_job_archetype_membership(
        "canonical-1", match("technology_delivery", "Program Manager", "q-pm")
    )
    supabase_utils.upsert_job_archetype_membership(
        "canonical-1", match("systems_platform_ops", "Infrastructure Engineer", "q-infra")
    )

    assert set(db.state) == {
        ("canonical-1", "technology_delivery"),
        ("canonical-1", "systems_platform_ops"),
    }


def test_legacy_software_tpm_and_unstamped_linkedin_jobs_use_canonical_lane(monkeypatch):
    db = AtomicMembershipRpc()
    monkeypatch.setattr(supabase_utils, "supabase", db)

    supabase_utils.upsert_job_archetype_membership(
        "legacy-1", match("software_tpm", "Technical Project Manager", "legacy-q")
    )
    supabase_utils.upsert_job_archetype_membership(
        "legacy-2", {"provider": "linkedin", "search_query": "Technical Project Manager"}
    )

    assert ("legacy-1", "technology_delivery") in db.state
    assert ("legacy-2", "technology_delivery") in db.state


class CanonicalMutation:
    def __init__(self, response_job_id):
        self.response_job_id = response_job_id

    def update(self, payload):
        return self

    def eq(self, key, value):
        return self

    def execute(self):
        return SimpleNamespace(data=[{"job_id": self.response_job_id}])


class CanonicalAndMembershipDb(AtomicMembershipRpc):
    def table(self, name):
        assert name == supabase_utils.config.SUPABASE_TABLE_NAME
        return CanonicalMutation("canonical-1")


def existing_canonical():
    return {
        "job_id": "canonical-1",
        "provider": "linkedin",
        "company": "Example",
        "job_title": "Infrastructure Engineer",
        "location": "Toronto",
        "description": "Operate VMware infrastructure.",
        "listing_instances": [{"job_id": "source-old"}],
        "seen_count": 1,
        "posting_wave_count": 1,
        "repost_count": 0,
        "last_seen_at": "2026-08-01T00:00:00+00:00",
    }


def canonical_match(query, query_id):
    return {
        "job_id": "source-repost",
        "provider": "linkedin",
        "company": "Example",
        "job_title": "Infrastructure Engineer",
        "location": "Toronto",
        "description": "Operate VMware infrastructure.",
        **match("systems_platform_ops", query, query_id),
    }


def test_repost_and_duplicate_canonical_saves_never_lose_membership_provenance(monkeypatch):
    db = CanonicalAndMembershipDb()
    existing = existing_canonical()
    monkeypatch.setattr(supabase_utils, "supabase", db)
    monkeypatch.setattr(supabase_utils, "get_canonical_candidates", lambda provider: [existing])
    monkeypatch.setattr(supabase_utils, "find_canonical_match", lambda job, rows: existing)
    monkeypatch.setattr(supabase_utils.config, "ENABLE_LINKEDIN_RELIST_TRACKING", False)

    first = supabase_utils.save_jobs_canonicalized([
        canonical_match("Infrastructure Engineer", "q-infra")
    ])
    second = supabase_utils.save_jobs_canonicalized([
        canonical_match("VMware Administrator", "q-vmware")
    ])

    assert first == second == ["canonical-1"]
    assert len(db.calls) == 2
    union = [dict(item) for item in db.state[("canonical-1", "systems_platform_ops")]]
    assert {item["query_id"] for item in union} == {"q-infra", "q-vmware"}
