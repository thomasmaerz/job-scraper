from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_downstream_workflows_default_to_enabled_lanes_with_optional_override():
    analyze = load_workflow("analyze_jobs.yml")
    score = load_workflow("score_jobs.yml")
    resume = load_workflow("hourly_resume_customization.yml")
    hourly = load_workflow("scrape_jobs.yml")

    assert triggers(analyze)["workflow_dispatch"]["inputs"]["archetype"]["default"] == ""
    assert triggers(score)["workflow_dispatch"]["inputs"]["archetype"]["default"] == ""
    assert triggers(resume)["workflow_dispatch"]["inputs"]["archetype"]["default"] == ""

    analyze_step = next(step for step in analyze["jobs"]["analyze"]["steps"] if step["name"] == "Run job insights analysis")
    assert analyze_step["env"]["JOB_INSIGHTS_ARCHETYPE"] == "${{ github.event.inputs.archetype || '' }}"
    score_step = next(step for step in score["jobs"]["score"]["steps"] if step["name"] == "Run job scoring script")
    assert score_step["env"]["JOB_SCORE_ARCHETYPE"] == "${{ inputs.archetype || '' }}"
    resume_step = next(step for step in resume["jobs"]["customize_resumes"]["steps"] if step["name"] == "Run resume customization script")
    assert resume_step["env"]["JOB_RESUME_ARCHETYPE"] == "${{ inputs.archetype || '' }}"
    scrape_step = next(step for step in hourly["jobs"]["scrape"]["steps"] if step.get("name") == "Run scraper script")
    assert scrape_step["env"]["SCRAPE_ARCHETYPE"] == "${{ inputs.archetype || '' }}"


def test_scrape_workflow_exports_only_manual_recovery_lookback():
    hourly = load_workflow("scrape_jobs.yml")
    recovery_step = next(
        step
        for step in hourly["jobs"]["scrape"]["steps"]
        if step.get("name") == "Configure manual LinkedIn recovery window"
    )

    assert recovery_step["if"] == "${{ github.event_name == 'workflow_dispatch' }}"
    assert recovery_step["env"]["REQUESTED_LOOKBACK_HOURS"] == "${{ inputs.lookback_hours }}"
    assert "LINKEDIN_RECOVERY_LOOKBACK_HOURS=${REQUESTED_LOOKBACK_HOURS}" in recovery_step["run"]


def test_scrape_workflow_allows_multi_lane_serial_runtime():
    hourly = load_workflow("scrape_jobs.yml")

    assert hourly["jobs"]["scrape"]["timeout-minutes"] == 180


def test_scoring_and_resume_workflows_have_defense_in_depth_concurrency_groups():
    score = load_workflow("score_jobs.yml")
    resume = load_workflow("hourly_resume_customization.yml")
    assert score["concurrency"] == {"group": "lane-scoring", "cancel-in-progress": False}
    assert resume["concurrency"] == {"group": "lane-resume-generation", "cancel-in-progress": False}


def test_all_linkedin_producer_workflows_share_source_concurrency_group():
    expected = {"group": "linkedin-freehire-pipeline", "cancel-in-progress": False}

    assert load_workflow("scrape_jobs.yml")["concurrency"] == expected
    assert load_workflow("job_manager.yml")["concurrency"] == expected
    assert load_workflow("backfill_linkedin_metadata.yml")["concurrency"] == expected


def load_workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def triggers(workflow):
    # PyYAML 1.1 reads the unquoted GitHub Actions `on` key as boolean true.
    return workflow[True]


def test_hourly_pipeline_serializes_runs_and_preserves_manual_lookback():
    workflow = load_workflow("scrape_jobs.yml")

    assert workflow["name"] == "Hourly Job Publication Pipeline"
    assert triggers(workflow)["schedule"] == [{
        "cron": "5 0,9-21 * * *",
        "timezone": "America/New_York",
    }]
    assert triggers(workflow)["workflow_dispatch"]["inputs"]["lookback_hours"]["default"] == "48"
    assert triggers(workflow)["workflow_dispatch"]["inputs"]["archetype"]["default"] == ""
    assert workflow["concurrency"] == {
        "group": "linkedin-freehire-pipeline",
        "cancel-in-progress": False,
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["scrape"]["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    assert workflow["jobs"]["scrape"]["timeout-minutes"] == 180

    recovery_step = next(
        step
        for step in workflow["jobs"]["scrape"]["steps"]
        if step["name"] == "Configure manual LinkedIn recovery window"
    )
    assert recovery_step["env"]["REQUESTED_LOOKBACK_HOURS"] == "${{ inputs.lookback_hours }}"
    assert "LINKEDIN_RECOVERY_LOOKBACK_HOURS" in recovery_step["run"]
    scrape_step = next(step for step in workflow["jobs"]["scrape"]["steps"] if step.get("id") == "scraper")
    assert scrape_step["env"]["LINKEDIN_DISCOVERY_MODE"] == "adaptive_queue"


def test_hourly_pipeline_has_separate_strictly_dependent_jobs_and_caps():
    workflow = load_workflow("scrape_jobs.yml")
    jobs = workflow["jobs"]

    assert list(jobs) == [
        "scrape",
        "freehire_compat",
        "publication_gate",
    ]
    assert jobs["freehire_compat"]["needs"] == "scrape"
    assert jobs["freehire_compat"]["if"] == (
        "${{ inputs.archetype == '' || inputs.archetype == null }}"
    )
    assert jobs["publication_gate"]["needs"] == ["scrape", "freehire_compat"]
    assert jobs["freehire_compat"]["timeout-minutes"] == 45
    assert jobs["publication_gate"]["timeout-minutes"] == 10

    compat_step = jobs["freehire_compat"]["steps"][-1]
    assert compat_step["env"]["FREEHIRE_CLASSIFY_LIMIT"] == "300"
    assert compat_step["env"]["FREEHIRE_DRAIN_BACKLOG"] == "false"
    assert compat_step["run"] == "python incremental_freehire_compat.py"

    analyze = load_workflow("analyze_jobs.yml")
    assert triggers(analyze)["workflow_run"] == {
        "workflows": ["Hourly Job Publication Pipeline"],
        "types": ["completed"],
    }


def test_publication_gate_checks_results_and_production_contract():
    workflow = load_workflow("scrape_jobs.yml")
    gate = workflow["jobs"]["publication_gate"]
    run = gate["steps"][-1]["run"]

    assert gate["if"] == "${{ always() && (inputs.archetype == '' || inputs.archetype == null) }}"
    assert "python publication_gate.py" in run
    assert '--scrape-result "${{ needs.scrape.result }}"' in run
    assert '--freehire-compat-result "${{ needs.freehire_compat.result }}"' in run
    assert '--discovery-cycle-id "${{ needs.scrape.outputs.discovery_cycle_id }}"' in run
    assert "--analyze-jobs-result" not in run
    assert gate["outputs"]["scrape_watermark"] == "${{ steps.gate.outputs.scrape_watermark }}"
    assert gate["outputs"]["generation"] == "${{ steps.gate.outputs.generation }}"
    assert gate["outputs"]["schema_version"] == "${{ steps.gate.outputs.schema_version }}"
    assert gate["outputs"]["outcome"] == "${{ steps.gate.outputs.outcome }}"


def test_source_pipeline_is_pull_only_and_ends_after_finalization():
    workflow = load_workflow("scrape_jobs.yml")

    assert list(workflow["jobs"])[-1] == "publication_gate"


def test_timezone_schedule_preserves_wall_clock_hours_across_dst():
    schedule = triggers(load_workflow("scrape_jobs.yml"))["schedule"][0]

    assert schedule["timezone"] == "America/New_York"
    assert schedule["cron"] == "5 0,9-21 * * *"
    eastern = ZoneInfo(schedule["timezone"])
    winter = datetime(2025, 1, 15, 21, 5, tzinfo=eastern)
    summer = datetime(2025, 7, 15, 21, 5, tzinfo=eastern)
    assert winter.utcoffset() != summer.utcoffset()
    assert (winter.hour, winter.minute) == (21, 5)
    assert (summer.hour, summer.minute) == (21, 5)


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


def test_analyze_worker_runs_after_successful_source_pipeline_and_is_serialized():
    workflow = load_workflow("analyze_jobs.yml")

    assert triggers(workflow)["workflow_run"] == {
        "workflows": ["Hourly Job Publication Pipeline"],
        "types": ["completed"],
    }
    assert "workflow_dispatch" in triggers(workflow)
    assert "concurrency" not in workflow
    assert workflow["jobs"]["analyze"]["concurrency"] == {
        "group": "linkedin-job-insights-analysis",
        "cancel-in-progress": False,
    }
    assert workflow["jobs"]["analyze"]["if"] == (
        "${{ github.event_name == 'workflow_dispatch' || "
        "(github.event.workflow_run.conclusion == 'success' && "
        "github.event.workflow_run.event == 'schedule') }}"
    )
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
