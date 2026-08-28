"""Verify pipeline results and report the production publication contract."""

import argparse
import os
from pathlib import Path

import config


PIPELINE_JOBS = ("scrape", "freehire_compat", "analyze_jobs")


def _get_db():
    from supabase import create_client

    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("Supabase URL and Key must be set in environment variables or config.")
    return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)


def verify_pipeline_results(results: dict[str, str]) -> None:
    failures = {
        name: results.get(name, "missing")
        for name in PIPELINE_JOBS
        if results.get(name) != "success"
    }
    if failures:
        detail = ", ".join(f"{name}={result}" for name, result in failures.items())
        raise RuntimeError(f"Publication gate blocked: pipeline jobs did not succeed ({detail})")


def _count(db, table: str, filters: tuple[tuple[str, object], ...] = ()) -> int:
    query = db.table(table).select("job_id", count="exact")
    for field, value in filters:
        query = query.eq(field, value)
    response = query.limit(1).execute()
    if response.count is None:
        raise RuntimeError("Production count query did not return an exact count")
    return response.count


def _count_not_null(db, table: str, field: str, filters=()) -> int:
    query = db.table(table).select("job_id", count="exact")
    for filter_field, value in filters:
        query = query.eq(filter_field, value)
    response = query.not_.is_(field, None).limit(1).execute()
    if response.count is None:
        raise RuntimeError("Production count query did not return an exact count")
    return response.count


def query_publication_state(db=None) -> dict[str, int | str | None]:
    db = db or _get_db()
    jobs = config.SUPABASE_TABLE_NAME
    counts: dict[str, int | str | None] = {
        "total": _count(db, jobs, (("provider", "linkedin"),)),
        "current": _count(
            db, jobs, (("provider", "linkedin"), ("freehire_compat_status", "current"))
        ),
        "pending": _count(
            db, jobs, (("provider", "linkedin"), ("freehire_compat_status", "pending"))
        ),
        "failed": _count(
            db, jobs, (("provider", "linkedin"), ("freehire_compat_status", "failed"))
        ),
        "processing": _count(
            db, jobs, (("provider", "linkedin"), ("freehire_compat_status", "processing"))
        ),
        "prior_publication": _count_not_null(
            db, jobs, "freehire_compat_classified_at", (("provider", "linkedin"),)
        ),
        # public.freehire_jobs is the per-row publication gate. A historical
        # pending/failed backlog is reported above but does not block current rows.
        "published": _count(db, "freehire_jobs"),
        "scrape_watermark": None,
    }
    # The service-role-only view is the row contract itself. A current row
    # absent from it is stale/incomplete; every counted published row passed
    # all view predicates without duplicating that contract in Python.
    counts["stale"] = int(counts["current"]) - int(counts["published"])
    try:
        response = (
            db.table("scrape_run_state")
            .select("last_successful_scrape_at")
            .eq("id", 1)
            .limit(1)
            .execute()
        )
        if response.data:
            counts["scrape_watermark"] = response.data[0].get("last_successful_scrape_at")
    except Exception as exc:  # The watermark contract predates some deployments.
        print(f"Scrape watermark unavailable: {exc}")
    return counts


def validate_publication_state(state: dict[str, int | str | None]) -> None:
    current = int(state["current"] or 0)
    published = int(state["published"] or 0)
    if published > current:
        raise RuntimeError(
            f"Publication gate blocked: published={published} exceeds current={current}"
        )
    if not state.get("scrape_watermark"):
        raise RuntimeError("Publication gate blocked: scrape watermark is absent")
    if int(state.get("prior_publication") or 0) > 0 and published == 0:
        raise RuntimeError(
            "Publication gate blocked: zero published rows after prior classification publication exists"
        )


def write_github_outputs(state: dict[str, int | str | None], output_path: str | None) -> None:
    if not output_path:
        return
    lines = []
    for key, value in state.items():
        text = "" if value is None else str(value)
        if "\n" in text or "\r" in text:
            raise ValueError(f"Unsafe multiline GitHub output for {key}")
        lines.append(f"{key}={text}\n")
    with Path(output_path).open("a", encoding="utf-8") as output:
        output.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    for job in PIPELINE_JOBS:
        parser.add_argument(f"--{job.replace('_', '-')}-result", required=True)
    args = parser.parse_args()
    results = {
        "scrape": args.scrape_result,
        "freehire_compat": args.freehire_compat_result,
        "analyze_jobs": args.analyze_jobs_result,
    }
    verify_pipeline_results(results)
    state = query_publication_state()
    validate_publication_state(state)
    print(
        "Publication state: "
        f"total={state['total']} current={state['current']} pending={state['pending']} "
        f"failed={state['failed']} processing={state['processing']} stale={state['stale']} "
        f"published={state['published']} prior_publication={state['prior_publication']} "
        f"scrape_watermark={state['scrape_watermark'] or 'unavailable'}"
    )
    write_github_outputs(state, os.getenv("GITHUB_OUTPUT"))


if __name__ == "__main__":
    main()
