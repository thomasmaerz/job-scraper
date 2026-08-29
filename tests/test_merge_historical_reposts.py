from types import SimpleNamespace

import merge_historical_reposts


def test_build_merge_plan_keeps_similar_cross_location_variants_separate():
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


def test_build_merge_plan_folds_exact_cross_location_variants():
    description = "**Job Title: Senior Project Manager**\n\n" + " ".join(f"delivery token{index}" for index in range(100))
    rows = [
        {"job_id": "1", "company": "Acme", "job_title": "Senior Project Manager - Toronto", "location": "Toronto", "description": description, "description_fingerprint": "same", "scraped_at": "2026-01-01"},
        {"job_id": "2", "company": "Acme", "job_title": "Senior Project Manager - Calgary", "location": "Calgary", "description": description, "description_fingerprint": "same", "scraped_at": "2026-02-01"},
    ]

    plan = merge_historical_reposts.build_merge_plan(rows)

    assert len(plan) == 1
    assert plan[0]["match_method"] == "body_hash_fuzzy_title"
    assert merge_historical_reposts.audit_merge_plan(rows, plan)["groups"][0]["source_locations"] == ["calgary", "toronto"]


def test_build_merge_plan_folds_exact_relq_title_variants():
    description = "**Job Title: IT Agile Project Manager with AWS**\n\n" + " ".join(f"agile aws token{index}" for index in range(100))
    rows = [
        {"job_id": "1", "company": "RELQ TECHNOLOGIES", "job_title": "IT Project manager", "location": "Canada", "description": description, "description_fingerprint": "same", "scraped_at": "2026-01-01"},
        {"job_id": "2", "company": "RELQ TECHNOLOGIES", "job_title": "IT Agile PM- Canada- Remote", "location": "Canada", "description": description, "description_fingerprint": "same", "scraped_at": "2026-02-01"},
    ]

    plan = merge_historical_reposts.build_merge_plan(rows)

    assert len(plan) == 1
    assert plan[0]["match_method"] == "body_hash_fuzzy_title"


def test_build_merge_plan_accepts_same_body_hash_when_source_formatting_differs():
    common = " ".join(f"token{index}" for index in range(100))
    rows = [
        {"job_id": "1", "company": "Acme", "job_title": "Program Manager", "location": "Toronto", "description": common + " corporate", "description_fingerprint": "same", "scraped_at": "2026-01-01"},
        {"job_id": "2", "company": "Acme", "job_title": "Program Manager - Engineering", "location": "Calgary", "description": common + " engineering", "description_fingerprint": "same", "scraped_at": "2026-02-01"},
    ]

    plan = merge_historical_reposts.build_merge_plan(rows)
    assert len(plan) == 1
    assert plan[0]["match_method"] == "body_hash_fuzzy_title"


def test_build_merge_plan_rejects_exact_text_for_titles_below_fuzzy_threshold():
    description = " ".join(f"corporate engineering token{index}" for index in range(100))
    rows = [
        {"job_id": "1", "company": "CN", "job_title": "Senior Expert - Corporate Services", "location": "Montreal", "description": description, "description_fingerprint": "same", "scraped_at": "2026-01-01"},
        {"job_id": "2", "company": "CN", "job_title": "Junior Data Analyst", "location": "Montreal", "description": description, "description_fingerprint": "same", "scraped_at": "2026-02-01"},
    ]

    assert merge_historical_reposts.build_merge_plan(rows) == []


def test_survivor_prefers_applied_workflow_over_multiple_weaker_flags():
    description = " ".join(f"delivery token{index}" for index in range(100))
    rows = [
        {"job_id": "applied", "company": "Acme", "job_title": "Project Manager", "location": "Toronto", "description": description, "description_fingerprint": "same", "scraped_at": "2026-01-01", "status": "applied"},
        {"job_id": "flags", "company": "Acme", "job_title": "Project Manager", "location": "Toronto", "description": description, "description_fingerprint": "same", "scraped_at": "2026-02-01", "notes": "note", "is_interested": True},
    ]

    plan = merge_historical_reposts.build_merge_plan(rows)

    assert plan[0]["survivor_job_id"] == "applied"


def test_run_stages_the_complete_plan_atomically(monkeypatch):
    rows = [
        {"job_id": "1", "company": "Acme", "job_title": "Project Manager", "location": "Toronto", "description": "same", "description_fingerprint": "same", "scraped_at": "2026-01-01"},
        {"job_id": "2", "company": "Acme", "job_title": "Project Manager", "location": "Toronto", "description": "same", "description_fingerprint": "same", "scraped_at": "2026-02-01"},
    ]
    plan = [{"source_job_id": "1", "survivor_job_id": "2", "match_method": "exact_fingerprint", "match_similarity": 1.0}]
    calls = []

    class Rpc:
        def execute(self):
            return type("Response", (), {"data": 1})()

    class Db:
        def rpc(self, name, payload=None):
            calls.append((name, payload))
            return Rpc()

    monkeypatch.setattr(merge_historical_reposts, "fetch_jobs", lambda: rows)
    monkeypatch.setattr(merge_historical_reposts, "build_merge_plan", lambda _: plan)
    monkeypatch.setattr(merge_historical_reposts.supabase_utils, "supabase", Db())
    import sys
    monkeypatch.setitem(sys.modules, "analyze_jobs", type("Analyze", (), {"rebuild_keyword_insights": staticmethod(lambda db: None)})())

    merge_historical_reposts.run(apply=True)

    assert calls[0] == ("replace_historical_repost_plan", {"p_plan": plan})
    assert calls[1] == ("merge_historical_repost_plan", None)


def test_run_refuses_to_apply_fuzzy_plan(monkeypatch):
    rows = [{"job_id": "1"}, {"job_id": "2"}]
    plan = [{"source_job_id": "1", "survivor_job_id": "2", "match_method": "fuzzy_description", "match_similarity": 0.99}]
    monkeypatch.setattr(merge_historical_reposts, "fetch_jobs", lambda: rows)
    monkeypatch.setattr(merge_historical_reposts, "build_merge_plan", lambda _: plan)

    import pytest
    with pytest.raises(ValueError, match="not exact and complete"):
        merge_historical_reposts.run(apply=True, body_hash_only=False)


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
