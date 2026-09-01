import importlib

import config


def test_software_tpm_archetype_contains_all_current_linkedin_queries():
    software_tpm = config.ARCHETYPE_CONFIGS["software_tpm"]

    assert software_tpm["provider"] == "linkedin"
    assert software_tpm["filter_profile"] == "software_tpm_v1"
    assert software_tpm["search_queries"] == [
        "IT Project Manager",
        "Technical Project Manager",
        "Information Technology Project Manager",
        "Technical Program Manager",
    ]


def test_software_tpm_is_explicit_technology_delivery_compatibility_alias():
    assert config.ARCHETYPE_CONFIGS["software_tpm"] is config.ARCHETYPE_CONFIGS["technology_delivery"]


def test_software_tpm_desc_blocklist_does_not_include_aerospace_defense_rule():
    patterns = config.ARCHETYPE_CONFIGS["software_tpm"]["desc_blocklist"]
    assert r"aerospace.*defense|defense.*aerospace" not in patterns


def test_software_tpm_keeps_construction_filters():
    archetype = config.ARCHETYPE_CONFIGS["software_tpm"]

    assert r"\bconstruction\b" in archetype["title_blocklist"]
    assert r"construction firm" in archetype["desc_blocklist"]
    assert r"\bProcore\b" in archetype["desc_blocklist"]


def test_job_insights_max_jobs_is_environment_configurable(monkeypatch):
    monkeypatch.setenv("JOB_INSIGHTS_MAX_JOBS", "37")
    reloaded = importlib.reload(config)

    assert reloaded.JOB_INSIGHTS_MAX_JOBS == 37

    monkeypatch.delenv("JOB_INSIGHTS_MAX_JOBS")
    importlib.reload(config)
