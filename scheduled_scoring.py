"""Dependency-light gate for DB-configured scheduled scoring."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from downstream_orchestration import run_enabled_lanes
from scrape_configuration import load_scrape_configuration


logger = logging.getLogger(__name__)


def run_configured_scoring(
    worker: Callable[[str], Any], *, db: Any, archetype_override: str | None = None
) -> dict[str, Any] | dict[str, str]:
    """Gate the scheduled multi-lane worker on scrape_settings.score_jobs."""
    configuration = load_scrape_configuration(db=db)
    if not configuration.settings.score_jobs:
        logger.info("Scheduled scoring skipped: scrape_settings.score_jobs is false.")
        return {"status": "skipped_score_jobs_disabled"}
    return run_enabled_lanes(worker, db=db, override=archetype_override)
