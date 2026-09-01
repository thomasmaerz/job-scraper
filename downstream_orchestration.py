"""Deterministic DB-configured orchestration for lane-aware workers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from lane_catalog import canonical_lane_slug
from scrape_configuration import load_scrape_configuration


logger = logging.getLogger(__name__)


def enabled_lane_slugs(db: Any, override: str | None = None) -> tuple[str, ...]:
    """Return one requested lane, or every enabled canonical lane in DB order."""
    if override and override.strip():
        return (canonical_lane_slug(override.strip()),)
    configuration = load_scrape_configuration(db=db)
    return tuple(
        lane.archetype
        for lane in sorted(
            (lane for lane in configuration.lanes if lane.enabled),
            key=lambda lane: (lane.sort_order, lane.archetype),
        )
    )


def run_enabled_lanes(
    worker: Callable[[str], Any], *, db: Any, override: str | None = None
) -> dict[str, Any]:
    """Run a worker once per lane, retaining deterministic result ordering."""
    results: dict[str, Any] = {}
    lanes = enabled_lane_slugs(db, override)
    logger.info("Running downstream worker for lanes: %s", ", ".join(lanes))
    for archetype in lanes:
        results[archetype] = worker(archetype)
    return results
