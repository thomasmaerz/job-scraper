"""Verify pipeline results and finalize an immutable publication generation."""

import argparse
import os
from pathlib import Path

import config


PIPELINE_JOBS = ("scrape", "freehire_compat")
PUBLICATION_SCHEMA_VERSION = "freehire-publication-v1"


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
    if published == 0:
        raise RuntimeError("Publication gate blocked: zero published rows")


def finalize_publication(db, state: dict[str, int | str | None]) -> dict[str, int | str]:
    watermark = state.get("scrape_watermark")
    if not isinstance(watermark, str) or not watermark:
        raise RuntimeError("Publication finalization requires a non-null scrape watermark")

    response = db.rpc(
        "finalize_freehire_publication",
        {"p_source_scrape_watermark": watermark},
    ).execute()
    rows = response.data or []
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise RuntimeError("Publication finalization did not return exactly one state row")

    publication = rows[0]
    generation = publication.get("generation")
    row_count = publication.get("row_count")
    returned_watermark = publication.get("source_scrape_watermark")
    published_at = publication.get("published_at")
    schema_version = publication.get("schema_version")
    if not isinstance(generation, int) or generation <= 0:
        raise RuntimeError("Publication finalization returned an invalid generation")
    if not isinstance(row_count, int) or row_count < 0:
        raise RuntimeError("Publication finalization returned an invalid row count")
    if returned_watermark != watermark:
        raise RuntimeError(
            "Publication finalization returned a different scrape watermark "
            f"({returned_watermark!r} != {watermark!r})"
        )
    if not isinstance(published_at, str) or not published_at:
        raise RuntimeError("Publication finalization returned an invalid publication time")
    if schema_version != PUBLICATION_SCHEMA_VERSION:
        raise RuntimeError(
            "Publication finalization returned an unsupported schema version "
            f"({schema_version!r})"
        )
    return {
        "generation": generation,
        "published_at": published_at,
        "source_scrape_watermark": returned_watermark,
        "row_count": row_count,
        "schema_version": schema_version,
    }


def prune_publication_generations(db) -> int:
    """Prune at most one expired generation outside the publish transaction."""
    response = db.rpc(
        "prune_freehire_publication_generations",
        {"p_keep_generations": 3, "p_max_generations": 3},
    ).execute()
    value = response.data
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("prune_freehire_publication_generations")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError("Publication pruning returned an invalid deletion count")
    return value


def try_prune_publication_generations(db) -> int:
    """Best-effort retention maintenance after an atomic publication commit."""
    try:
        return prune_publication_generations(db)
    except Exception as exc:
        print(f"Publication pruning deferred: {exc}")
        return 0


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
    }
    verify_pipeline_results(results)
    db = _get_db()
    state = query_publication_state(db)
    validate_publication_state(state)
    state.update(finalize_publication(db, state))
    pruned_generations = try_prune_publication_generations(db)
    print(
        "Publication state: "
        f"total={state['total']} current={state['current']} pending={state['pending']} "
        f"failed={state['failed']} processing={state['processing']} stale={state['stale']} "
        f"published={state['published']} prior_publication={state['prior_publication']} "
        f"scrape_watermark={state['scrape_watermark'] or 'unavailable'} "
        f"generation={state['generation']} published_at={state['published_at']} "
        f"schema_version={state['schema_version']} row_count={state['row_count']} "
        f"pruned_generations={pruned_generations}"
    )
    write_github_outputs(state, os.getenv("GITHUB_OUTPUT"))


if __name__ == "__main__":
    main()
