from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_scraper_workflow_serializes_runs_and_has_timeout():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "scrape_jobs.yml").read_text()
    )

    assert workflow["concurrency"] == {
        "group": "daily-job-scraper",
        "cancel-in-progress": False,
    }
    assert workflow["jobs"]["scrape"]["timeout-minutes"] == 30
