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
    recovery_step = next(
        step
        for step in workflow["jobs"]["scrape"]["steps"]
        if step["name"] == "Determine LinkedIn recovery window"
    )
    assert recovery_step["env"]["REQUESTED_LOOKBACK_HOURS"] == "${{ inputs.lookback_hours || '48' }}"
    assert "--status success" in recovery_step["run"]
    assert '--branch "$GITHUB_REF_NAME"' in recovery_step["run"]
    assert "LINKEDIN_LAST_SUCCESS_AT" in recovery_step["run"]
