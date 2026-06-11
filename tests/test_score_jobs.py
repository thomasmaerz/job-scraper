import importlib
import sys
from types import SimpleNamespace


def test_get_resume_score_uses_job_scoring_client_with_medium_reasoning(monkeypatch):
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
    assert calls[0]["reasoning_effort"] == "medium"
    assert "temperature" not in calls[0]
