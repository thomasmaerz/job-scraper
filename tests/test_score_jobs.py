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
