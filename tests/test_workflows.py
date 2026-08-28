from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def load_workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def triggers(workflow):
    # PyYAML 1.1 reads the unquoted GitHub Actions `on` key as boolean true.
    return workflow[True]


def test_hourly_pipeline_serializes_runs_and_preserves_manual_lookback():
    workflow = load_workflow("scrape_jobs.yml")

    assert workflow["name"] == "Hourly Job Publication Pipeline"
    assert triggers(workflow)["schedule"] == [{"cron": "5 * * * *"}]
    assert triggers(workflow)["workflow_dispatch"]["inputs"]["lookback_hours"]["default"] == "48"
    assert workflow["concurrency"] == {
        "group": "linkedin-freehire-pipeline",
        "cancel-in-progress": False,
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["scrape"]["permissions"] == {
        "actions": "read",
        "contents": "read",
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


def test_hourly_pipeline_has_separate_strictly_dependent_jobs_and_caps():
    workflow = load_workflow("scrape_jobs.yml")
    jobs = workflow["jobs"]

    assert list(jobs) == [
        "scrape",
        "freehire_compat",
        "analyze_jobs",
        "publication_gate",
        "downstream_dispatch",
    ]
    assert jobs["freehire_compat"]["needs"] == "scrape"
    assert jobs["analyze_jobs"]["needs"] == "freehire_compat"
    assert jobs["publication_gate"]["needs"] == ["scrape", "freehire_compat", "analyze_jobs"]
    assert jobs["downstream_dispatch"]["needs"] == "publication_gate"
    assert jobs["freehire_compat"]["timeout-minutes"] == 45
    assert jobs["analyze_jobs"]["timeout-minutes"] == 45
    assert jobs["analyze_jobs"]["concurrency"] == {
        "group": "linkedin-job-insights-analysis",
        "cancel-in-progress": False,
    }
    assert jobs["publication_gate"]["timeout-minutes"] == 10
    assert jobs["downstream_dispatch"]["timeout-minutes"] == 5

    compat_step = jobs["freehire_compat"]["steps"][-1]
    assert compat_step["env"]["FREEHIRE_CLASSIFY_LIMIT"] == "300"
    assert compat_step["env"]["FREEHIRE_DRAIN_BACKLOG"] == "false"
    assert compat_step["run"] == "python frontfill_freehire_compat.py"

    analyze_step = jobs["analyze_jobs"]["steps"][-1]
    assert analyze_step["env"]["JOB_INSIGHTS_MAX_JOBS"] == "100"
    assert analyze_step["env"]["JOB_INSIGHTS_BACKFILL_ALL"] == "false"
    assert analyze_step["env"]["JOB_INSIGHTS_REPLACEMENT_BACKFILL"] == "false"
    assert analyze_step["run"] == "python analyze_jobs.py"


def test_publication_gate_checks_results_and_production_contract():
    workflow = load_workflow("scrape_jobs.yml")
    gate = workflow["jobs"]["publication_gate"]
    run = gate["steps"][-1]["run"]

    assert gate["if"] == "${{ always() }}"
    assert "python publication_gate.py" in run
    assert '--scrape-result "${{ needs.scrape.result }}"' in run
    assert '--freehire-compat-result "${{ needs.freehire_compat.result }}"' in run
    assert '--analyze-jobs-result "${{ needs.analyze_jobs.result }}"' in run
    assert gate["outputs"]["scrape_watermark"] == "${{ steps.gate.outputs.scrape_watermark }}"


def test_downstream_dispatch_is_configurable_and_does_not_guess_or_print_token():
    workflow = load_workflow("scrape_jobs.yml")
    step = workflow["jobs"]["downstream_dispatch"]["steps"][0]
    run = step["run"]

    assert step["env"]["DOWNSTREAM_SYNC_REPOSITORY"] == "${{ vars.DOWNSTREAM_SYNC_REPOSITORY }}"
    assert step["env"]["DOWNSTREAM_SYNC_TOKEN"] == "${{ secrets.DOWNSTREAM_SYNC_TOKEN }}"
    assert "Downstream dispatch is not configured" in run
    assert "job-scraper-publication-ready" in run
    assert "https://api.github.com/repos/${DOWNSTREAM_SYNC_REPOSITORY}/dispatches" in run
    assert "source_repository" in run
    assert "SOURCE_SHA" in run
    assert "SOURCE_RUN_ID" in run
    assert "SCRAPE_WATERMARK" in run
    assert "set -x" not in run
    assert "echo $DOWNSTREAM_SYNC_TOKEN" not in run


def test_freehire_recovery_is_manual_bounded_and_dry_run_by_default():
    workflow = load_workflow("freehire_compat.yml")
    event = triggers(workflow)
    inputs = event["workflow_dispatch"]["inputs"]

    assert "schedule" not in event
    assert workflow["concurrency"]["group"] == "linkedin-freehire-pipeline"
    assert workflow["permissions"] == {"contents": "read"}
    assert inputs["limit"]["default"] == "1000"
    assert "1000" in inputs["limit"]["options"]
    assert inputs["apply"]["default"] is False
    run = workflow["jobs"]["classify"]["steps"][-1]["run"]
    assert 'case "$FREEHIRE_CLASSIFY_LIMIT" in' in run
    assert 'case "$FREEHIRE_APPLY" in' in run
    assert "100|300|500|1000" in run
    assert "replacement_before is required" in run
    assert '--replacement-before "$FREEHIRE_REPLACEMENT_BEFORE"' in run
    assert 'if [[ "$FREEHIRE_APPLY" == "true" ]]; then args+=(--apply); fi' in run
    assert "backfill_freehire_compat.py" in run


def test_analyze_recovery_is_manual_only_and_serialized():
    workflow = load_workflow("analyze_jobs.yml")

    assert "schedule" not in triggers(workflow)
    assert "workflow_dispatch" in triggers(workflow)
    assert "concurrency" not in workflow
    assert workflow["jobs"]["analyze"]["concurrency"] == {
        "group": "linkedin-job-insights-analysis",
        "cancel-in-progress": False,
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["analyze"]["timeout-minutes"] == 120


def test_same_id_relist_backfill_is_manual_and_dry_run_by_default():
    workflow = load_workflow("backfill_same_id_relists.yml")
    dispatch = triggers(workflow)["workflow_dispatch"]

    assert dispatch["inputs"]["limit"]["default"] == "500"
    assert dispatch["inputs"]["limit"]["options"] == ["100", "500", "1000", "5000"]
    assert dispatch["inputs"]["apply"]["default"] == "false"
    run = workflow["jobs"]["backfill"]["steps"][-1]["run"]
    assert "backfill_same_id_relists.py" in run
    assert "--apply" in run
    assert 'case "$limit" in' in run
    assert "100|500|1000|5000" in run
