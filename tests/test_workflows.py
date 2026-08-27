from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_scraper_workflow_serializes_runs_and_has_timeout():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "scrape_jobs.yml").read_text()
    )

    assert workflow["concurrency"] == {
        "group": "linkedin-freehire-pipeline",
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


def test_same_id_relist_backfill_is_manual_and_dry_run_by_default():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "backfill_same_id_relists.yml").read_text()
    )

    dispatch = workflow[True]["workflow_dispatch"]
    assert dispatch["inputs"]["limit"]["default"] == "500"
    assert dispatch["inputs"]["apply"]["default"] == "false"
    run = workflow["jobs"]["backfill"]["steps"][-1]["run"]
    assert "backfill_same_id_relists.py" in run
    assert "--apply" in run
    assert 'case "$limit" in' in run


def test_freehire_frontfill_runs_after_scrape_and_on_recovery_schedule():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "freehire_compat.yml").read_text()
    )
    assert workflow[True]["schedule"][0]["cron"] == "15 */4 * * *"
    run = workflow["jobs"]["classify"]["steps"][-1]["run"]
    assert 'case "$FREEHIRE_CLASSIFY_LIMIT" in' in run
    assert workflow[True]["workflow_dispatch"]["inputs"]["limit"]["default"] == "200"
    run = workflow["jobs"]["classify"]["steps"][-1]["run"]
    assert "backfill_freehire_compat.py" in run
    assert "--apply" in run

    scrape = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "scrape_jobs.yml").read_text()
    )
    assert scrape["jobs"]["scrape"]["steps"][-1]["run"] == "python frontfill_freehire_compat.py"
