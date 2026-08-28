from types import SimpleNamespace

import merge_historical_reposts


def test_build_merge_plan_keeps_cross_location_variants_separate():
    common = " ".join(f"token{index}" for index in range(100))
    rows = [
        {
            "job_id": "1",
            "company": "Acme",
            "job_title": "Senior Project Manager - Toronto",
            "location": "Toronto",
            "description": common + " first",
            "description_fingerprint": "one",
            "scraped_at": "2026-01-01",
        },
        {
            "job_id": "2",
            "company": "Acme",
            "job_title": "Senior Project Manager - Calgary",
            "location": "Calgary",
            "description": common + " second",
            "description_fingerprint": "two",
            "scraped_at": "2026-02-01",
            "applicant_count": 25,
            "salary_text": "$100,000-$120,000",
        },
    ]

    plan = merge_historical_reposts.build_merge_plan(rows)

    assert plan == []


def test_historical_and_live_matching_are_equivalent_for_same_location():
    description = " ".join(f"delivery token{index}" for index in range(100))
    rows = [
        {"job_id": "1", "company": "Acme", "job_title": "Senior Project Manager", "location": "Toronto", "description": description, "description_fingerprint": "same", "scraped_at": "2026-01-01"},
        {"job_id": "2", "company": "Acme", "job_title": "Senior Project Manager", "location": "Toronto", "description": description, "description_fingerprint": "same", "scraped_at": "2026-02-01", "applicant_count": 5},
    ]

    plan = merge_historical_reposts.build_merge_plan(rows)
    live_match = merge_historical_reposts.supabase_utils.find_canonical_match(rows[1], [rows[0]])

    assert len(plan) == 1
    assert live_match == rows[0]


def test_apply_flow_stages_and_merges_through_serialized_rpcs(monkeypatch):
    rpc_calls = []

    class Query:
        def execute(self):
            return SimpleNamespace(data=[])

    class FakeDb:
        def table(self, name):
            raise AssertionError(f"plan staging must use an RPC, not table {name}")

        def rpc(self, name, params=None):
            rpc_calls.append((name, params))
            return Query()

    plan = [{"source_job_id": "old", "survivor_job_id": "new", "match_method": "exact_fingerprint", "match_similarity": None}]
    monkeypatch.setattr(merge_historical_reposts, "fetch_jobs", lambda: [])
    monkeypatch.setattr(merge_historical_reposts, "build_merge_plan", lambda rows: plan)
    monkeypatch.setattr(merge_historical_reposts.supabase_utils, "supabase", FakeDb())

    merge_historical_reposts.run(apply=True)

    assert rpc_calls == [
        ("replace_historical_repost_plan", {"p_plan": plan}),
        ("merge_historical_repost_plan", None),
    ]


def test_dry_run_does_not_stage_or_merge(monkeypatch):
    class FakeDb:
        def rpc(self, name, params=None):
            raise AssertionError(f"dry run must not call {name}")

    plan = [
        {
            "source_job_id": "old",
            "survivor_job_id": "new",
            "match_method": "exact_fingerprint",
            "match_similarity": None,
        }
    ]
    monkeypatch.setattr(merge_historical_reposts, "fetch_jobs", lambda: [])
    monkeypatch.setattr(merge_historical_reposts, "build_merge_plan", lambda rows: plan)
    monkeypatch.setattr(merge_historical_reposts.supabase_utils, "supabase", FakeDb())

    assert merge_historical_reposts.run(apply=False) == {
        "groups": 1,
        "redundant_rows": 1,
        "exact": 1,
        "fuzzy": 0,
    }
