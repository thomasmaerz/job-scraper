import importlib
import sys
from types import SimpleNamespace

import supabase_utils


def test_get_resume_score_uses_job_scoring_client_without_reasoning(monkeypatch):
    fake_supabase_utils = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "supabase_utils", fake_supabase_utils)
    score_jobs = importlib.import_module("score_jobs")

    calls = []

    class FakeClient:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return "87"

    monkeypatch.setattr(score_jobs, "job_scoring_client", FakeClient(), raising=False)

    result = score_jobs.get_resume_score_from_ai(
        "Resume text",
        {
            "job_id": "job-1",
            "company": "Acme",
            "job_title": "Technical Project Manager",
            "description": "Lead technical delivery.",
            "level": "Senior",
        },
    )

    assert result == 87
    assert "reasoning_effort" not in calls[0]
    assert "temperature" not in calls[0]


def test_scheduled_scoring_skips_successfully_when_db_setting_is_false(monkeypatch):
    import scheduled_scoring
    configuration = SimpleNamespace(settings=SimpleNamespace(score_jobs=False))
    monkeypatch.setattr(scheduled_scoring, "load_scrape_configuration", lambda db: configuration)

    worker = lambda _lane: (_ for _ in ()).throw(AssertionError("must not score"))
    assert scheduled_scoring.run_configured_scoring(worker, db=object()) == {
        "status": "skipped_score_jobs_disabled"
    }


def test_scheduled_scoring_loads_db_setting_before_lane_orchestration(monkeypatch):
    import scheduled_scoring
    calls = []
    configuration = SimpleNamespace(settings=SimpleNamespace(score_jobs=True))
    monkeypatch.setattr(
        scheduled_scoring,
        "load_scrape_configuration",
        lambda db: calls.append(("load", db)) or configuration,
    )
    monkeypatch.setattr(
        scheduled_scoring,
        "run_enabled_lanes",
        lambda worker, **kwargs: calls.append(("run", kwargs)) or {"ok": True},
    )
    db = object()

    assert scheduled_scoring.run_configured_scoring(
        lambda _lane: None,
        db=db, archetype_override="network_infrastructure"
    ) == {"ok": True}
    assert calls == [
        ("load", db),
        ("run", {"db": db, "override": "network_infrastructure"}),
    ]


def test_get_top_scored_jobs_to_apply_excludes_filtered_jobs(monkeypatch):
    class FakeQuery:
        def __init__(self):
            self.calls = []

        def select(self, value):
            self.calls.append(("select", value))
            return self

        def eq(self, key, value):
            self.calls.append(("eq", key, value))
            return self

        @property
        def not_(self):
            self.calls.append(("not_",))
            return self

        def is_(self, key, value):
            self.calls.append(("is_", key, value))
            return self

        def order(self, key, desc=False):
            self.calls.append(("order", key, desc))
            return self

        def limit(self, value):
            self.calls.append(("limit", value))
            return self

        def execute(self):
            self.calls.append(("execute",))
            return SimpleNamespace(data=[{"job_id": "job-1", "resume_score": 92}])

    class FakeDb:
        def __init__(self):
            self.query = FakeQuery()

        def table(self, name):
            assert name == supabase_utils.config.SUPABASE_TABLE_NAME
            return self.query

    db = FakeDb()
    monkeypatch.setattr(supabase_utils, "supabase", db)

    result = supabase_utils.get_top_scored_jobs_to_apply(10)

    assert result == [{"job_id": "job-1", "resume_score": 92}]
    assert ("eq", "is_active", True) in db.query.calls
    assert ("eq", "status", "new") in db.query.calls
    assert ("eq", "is_filtered", False) in db.query.calls
    assert ("not_",) in db.query.calls
    assert ("is_", "resume_score", None) in db.query.calls
    assert ("order", "resume_score", True) in db.query.calls
    assert ("limit", 10) in db.query.calls


def test_update_job_score_isolated_to_membership_lane(monkeypatch):
    calls = []
    class Query:
        def update(self, payload): calls.append(("update", payload)); return self
        def eq(self, key, value): calls.append(("eq", key, value)); return self
        def execute(self): return SimpleNamespace(data=[{}])
    class Db:
        def table(self, name): calls.append(("table", name)); return Query()
    monkeypatch.setattr(supabase_utils, "supabase", Db())

    assert supabase_utils.update_job_score("job-1", 88, archetype="data_pm") is True
    assert ("table", "job_archetype_memberships") in calls
    assert ("eq", "job_id", "job-1") in calls
    assert ("eq", "archetype", "data_pm") in calls
    assert not any(call == ("table", supabase_utils.config.SUPABASE_TABLE_NAME) for call in calls)


def test_scores_for_same_job_do_not_overwrite_another_lane(monkeypatch):
    scores = {}
    class Query:
        def __init__(self): self.payload = None; self.job = None; self.lane = None
        def update(self, payload): self.payload = payload; return self
        def eq(self, key, value):
            if key == "job_id": self.job = value
            if key == "archetype": self.lane = value
            return self
        def execute(self): scores[(self.job, self.lane)] = self.payload["match_score"]; return SimpleNamespace(data=[{}])
    class Db:
        def table(self, name): assert name == "job_archetype_memberships"; return Query()
    monkeypatch.setattr(supabase_utils, "supabase", Db())
    supabase_utils.update_job_score("job-1", 91, archetype="data_pm")
    supabase_utils.update_job_score("job-1", 64, archetype="network_infrastructure")
    assert scores == {("job-1", "data_pm"): 91, ("job-1", "network_infrastructure"): 64}
