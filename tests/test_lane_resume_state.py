from types import SimpleNamespace

import supabase_utils


def test_resume_link_update_isolated_to_membership_lane(monkeypatch):
    calls = []

    class Query:
        def update(self, payload): calls.append(("update", payload)); return self
        def eq(self, key, value): calls.append(("eq", key, value)); return self
        def execute(self): return SimpleNamespace(data=[{}])

    class Db:
        def table(self, name): calls.append(("table", name)); return Query()

    monkeypatch.setattr(supabase_utils, "supabase", Db())
    assert supabase_utils.update_job_with_resume_link(
        "job-1", "resume-1", archetype="network_infrastructure"
    ) is True
    assert ("table", "job_archetype_memberships") in calls
    assert ("eq", "archetype", "network_infrastructure") in calls


def test_customized_resume_persists_lane_base_resume_and_job(monkeypatch):
    inserted = []

    class Query:
        def insert(self, payload): inserted.append(payload); return self
        def execute(self): return SimpleNamespace(data=[{"id": "resume-1"}])

    class Db:
        def table(self, name):
            assert name == supabase_utils.config.SUPABASE_CUSTOMIZED_RESUMES_TABLE_NAME
            return Query()

    class Resume:
        email = "test@example.com"
        def model_dump(self, **kwargs): return {"name": "Test", "email": self.email}

    monkeypatch.setattr(supabase_utils, "supabase", Db())
    assert supabase_utils.save_customized_resume(
        Resume(), "job-1/resume.pdf", archetype="software_tpm",
        base_resume_id="base-1", job_id="job-1",
    ) == "resume-1"
    assert inserted[0]["archetype"] == "technology_delivery"
    assert inserted[0]["base_resume_id"] == "base-1"


def test_same_job_two_lanes_use_distinct_canonical_resume_versions():
    from lane_resume_storage import customized_resume_storage_path

    delivery = customized_resume_storage_path(
        "software_tpm", "job-1", "resume-delivery"
    )
    network = customized_resume_storage_path(
        "network_infrastructure", "job-1", "resume-network"
    )

    assert delivery == "technology_delivery/job-1/resume-delivery.pdf"
    assert network == "network_infrastructure/job-1/resume-network.pdf"
    assert delivery != network


def test_customized_resume_id_matches_row_and_storage_version(monkeypatch):
    inserted = []

    class Query:
        def insert(self, payload): inserted.append(payload); return self
        def execute(self): return SimpleNamespace(data=[{"id": inserted[0]["id"]}])

    class Db:
        def table(self, _name): return Query()

    class Resume:
        email = "test@example.com"
        def model_dump(self, **_kwargs): return {"name": "Test", "email": self.email}

    monkeypatch.setattr(supabase_utils, "supabase", Db())
    path = "network_infrastructure/job-1/version-1.pdf"
    result = supabase_utils.save_customized_resume(
        Resume(), path, archetype="network_infrastructure", job_id="job-1",
        customized_resume_id="version-1",
    )
    assert result == "version-1"
    assert inserted[0]["id"] == "version-1"
    assert inserted[0]["resume_link"] == path
    assert inserted[0]["archetype"] == "network_infrastructure"
    assert inserted[0]["job_id"] == "job-1"


def test_resume_updates_for_same_job_do_not_cross_lane(monkeypatch):
    updates = {}

    class Query:
        def __init__(self): self.payload = None; self.job_id = None; self.lane = None
        def update(self, payload): self.payload = payload; return self
        def eq(self, key, value):
            if key == "job_id": self.job_id = value
            if key == "archetype": self.lane = value
            return self
        def execute(self): updates[(self.job_id, self.lane)] = self.payload; return SimpleNamespace(data=[{}])

    class Db:
        def table(self, name): assert name == "job_archetype_memberships"; return Query()

    monkeypatch.setattr(supabase_utils, "supabase", Db())
    supabase_utils.update_job_with_resume_link("job-1", "resume-a", archetype="data_pm")
    supabase_utils.update_job_with_resume_link("job-1", "resume-b", archetype="network_infrastructure")
    assert updates[("job-1", "data_pm")]["customized_resume_id"] == "resume-a"
    assert updates[("job-1", "network_infrastructure")]["customized_resume_id"] == "resume-b"
