import pytest


@pytest.fixture(autouse=True)
def legacy_linkedin_discovery_mode(monkeypatch):
    monkeypatch.setenv("LINKEDIN_DISCOVERY_MODE", "legacy")
