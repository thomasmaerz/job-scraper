from downstream_orchestration import enabled_lane_slugs, run_enabled_lanes


class Lane:
    def __init__(self, archetype, enabled, sort_order):
        self.archetype = archetype
        self.enabled = enabled
        self.sort_order = sort_order


def test_enabled_lanes_are_db_ordered_and_override_is_canonical(monkeypatch):
    monkeypatch.setattr(
        "downstream_orchestration.load_scrape_configuration",
        lambda db: type("Configuration", (), {"lanes": [
            Lane("network_infrastructure", True, 4),
            Lane("technology_delivery", True, 1),
            Lane("systems_platform_ops", False, 0),
        ]})(),
    )
    assert enabled_lane_slugs(object()) == (
        "technology_delivery", "network_infrastructure",
    )
    assert enabled_lane_slugs(object(), "software_tpm") == ("technology_delivery",)


def test_worker_runs_each_enabled_lane_once_in_order(monkeypatch):
    monkeypatch.setattr(
        "downstream_orchestration.enabled_lane_slugs",
        lambda db, override=None: ("technology_delivery", "network_infrastructure"),
    )
    calls = []
    result = run_enabled_lanes(lambda lane: calls.append(lane) or lane.upper(), db=object())
    assert calls == ["technology_delivery", "network_infrastructure"]
    assert list(result) == calls
