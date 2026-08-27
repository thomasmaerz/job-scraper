"""Backfill and frontfill the Freehire compatibility contract. Dry-run by default."""

import argparse
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone

import config
import freehire_compat


SELECT_FIELDS = (
    "job_id,latest_job_id,company,job_title,level,location,description,provider,"
    "posted_at,scraped_at,first_seen_at,last_seen_at,last_seen_posted_at,"
    "freehire_category,freehire_seniority,is_remote,freehire_remote_evidence,"
    "freehire_compat_status,freehire_compat_input_hash,freehire_compat_import_hash,"
    "freehire_compat_model,freehire_compat_prompt_version,freehire_compat_schema_version,"
    "freehire_compat_attempts,freehire_compat_claimed_at,freehire_compat_next_retry_at,"
    "last_checked,detail_metadata_checked_at,salary_text,salary_min,salary_max,salary_currency,"
    "applicant_count,applicant_count_text,applicant_count_type,recruiter_name,"
    "recruiter_profile_url,recruiter_identifier,original_job_id,seen_count,posting_wave_count,"
    "repost_count,same_id_relist_count,listing_instances,archetype,search_query,filter_profile,"
    "is_filtered,is_entry_level_filtered,filter_reason,description_fingerprint"
)


def _get_db():
    from supabase import create_client

    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("Supabase URL and Key must be set in environment variables or config.")
    return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)


def get_upper_bound(db) -> str | None:
    rows = (
        db.table(config.SUPABASE_TABLE_NAME)
        .select("job_id")
        .eq("provider", "linkedin")
        .order("job_id", desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )
    return str(rows[0]["job_id"]) if rows else None


def fetch_candidates(
    db,
    last_job_id: str | None = None,
    upper_bound: str | None = None,
    page_size: int | None = None,
) -> list[dict]:
    page_size = page_size or config.FREEHIRE_CLASSIFY_PAGE_SIZE
    query = (
        db.table(config.SUPABASE_TABLE_NAME)
        .select(SELECT_FIELDS)
        .eq("provider", "linkedin")
        .order("job_id", desc=False)
    )
    if last_job_id is not None:
        query = query.gt("job_id", last_job_id)
    if upper_bound is not None:
        query = query.lte("job_id", upper_bound)
    return query.range(0, page_size - 1).execute().data or []


def _valid_current(row: dict, input_hash: str) -> bool:
    return (
        row.get("freehire_compat_status") == "current"
        and row.get("freehire_compat_input_hash") == input_hash
        and row.get("freehire_compat_schema_version") == config.FREEHIRE_COMPAT_SCHEMA_VERSION
        and row.get("freehire_compat_prompt_version") == config.FREEHIRE_COMPAT_PROMPT_VERSION
        and freehire_compat.normalize_category(row.get("freehire_category")) is not None
        and freehire_compat.normalize_seniority(row.get("freehire_seniority")) is not None
    )


def _expired_claim(row: dict) -> bool:
    if row.get("freehire_compat_status") != "processing":
        return False
    value = row.get("freehire_compat_claimed_at")
    if not value:
        return True
    try:
        claimed_at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return True
    return claimed_at < datetime.now(timezone.utc) - timedelta(minutes=30)


def _claim(db, row: dict, worker_id: str) -> bool:
    response = db.rpc("claim_freehire_compat_job", {
        "p_job_id": str(row["job_id"]),
        "p_expected_input_hash": freehire_compat.compute_classification_hash(row),
        "p_expected_source_snapshot": freehire_compat.source_snapshot(row),
        "p_worker_id": worker_id,
    }).execute()
    return response.data is True or response.data == [True]


def _persist(db, job_id: str, input_hash: str, payload: dict, worker_id: str) -> bool:
    response = db.rpc("persist_freehire_compat_result", {
        "p_job_id": job_id,
        "p_expected_input_hash": input_hash,
        "p_expected_source_snapshot": payload.pop("_expected_source_snapshot"),
        "p_worker_id": worker_id,
        "p_payload": payload,
    }).execute()
    return response.data is True or response.data == [True]


def _retry_ready(row: dict) -> bool:
    attempts = int(row.get("freehire_compat_attempts") or 0)
    if attempts >= config.FREEHIRE_CLASSIFY_MAX_DURABLE_ATTEMPTS:
        return False
    value = row.get("freehire_compat_next_retry_at")
    if not value:
        return True
    try:
        retry_at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    return retry_at <= datetime.now(timezone.utc)


def run(
    apply: bool = False,
    limit: int | None = None,
    drain_backlog: bool = False,
    replacement_backfill: bool = False,
    db=None,
    client=None,
) -> dict:
    db = db or _get_db()
    if not apply:
        client = None
    if client is None and apply:
        from llm_client import freehire_classify_client

        client = freehire_classify_client
    limit = limit or config.FREEHIRE_CLASSIFY_LIMIT
    counts = {
        "scanned": 0,
        "unchanged": 0,
        "metadata_updated": 0,
        "would_classify": 0,
        "classified": 0,
        "failed": 0,
        "claimed_elsewhere": 0,
        "cooldown_or_exhausted": 0,
        "remote_true": 0,
        "remote_false": 0,
        "llm_requests": 0,
        "retries": 0,
        "splits": 0,
    }
    # --drain-backlog controls eligibility/sweep intent only. `limit` remains a
    # hard per-process ceiling so a recovery run cannot become unbounded.
    upper_bound = get_upper_bound(db)
    if upper_bound is None:
        return counts

    last_job_id = None
    pending: list[dict] = []
    while True:
        page = fetch_candidates(db, last_job_id=last_job_id, upper_bound=upper_bound)
        if not page:
            break
        for row in page:
            counts["scanned"] += 1
            is_remote, evidence = freehire_compat.classify_remote(row)
            counts["remote_true" if is_remote else "remote_false"] += 1
            input_hash = freehire_compat.compute_classification_hash(row)
            import_hash = freehire_compat.compute_import_hash(
                {**row, "freehire_remote_evidence": evidence}, is_remote=is_remote
            )
            current = _valid_current(row, input_hash) and not replacement_backfill
            deterministic_current = (
                row.get("is_remote") == is_remote
                and row.get("freehire_remote_evidence") == evidence
                and row.get("freehire_compat_import_hash") == import_hash
            )
            if current and deterministic_current:
                counts["unchanged"] += 1
                continue
            if current:
                counts["metadata_updated"] += 1
                if apply:
                    db.rpc("apply_freehire_compat_metadata", {
                        "p_job_id": str(row["job_id"]),
                        "p_expected_source_snapshot": freehire_compat.source_snapshot(row),
                        "p_payload": {
                            "is_remote": is_remote,
                            "freehire_remote_evidence": evidence,
                            "freehire_compat_import_hash": import_hash,
                        },
                    }).execute()
                continue
            if row.get("freehire_compat_status") == "processing" and not _expired_claim(row):
                counts["claimed_elsewhere"] += 1
                continue
            if row.get("freehire_compat_status") == "failed" and not _retry_ready(row):
                counts["cooldown_or_exhausted"] += 1
                continue
            if len(pending) < limit:
                pending.append(row)
                counts["would_classify"] += 1
        last_job_id = str(page[-1]["job_id"])
        if len(page) < config.FREEHIRE_CLASSIFY_PAGE_SIZE:
            break

    if not apply or not pending:
        return counts

    worker_id = str(uuid.uuid4())
    request_budget = config.FREEHIRE_CLASSIFY_REQUEST_BUDGET
    for pending_batch in freehire_compat.pack_batches(pending, model=freehire_compat.model_name(client)):
        if request_budget <= 0:
            break
        claimed = [row for row in pending_batch if _claim(db, row, worker_id)]
        batch = []
        for claimed_row in claimed:
            fresh = (
                db.table(config.SUPABASE_TABLE_NAME)
                .select(SELECT_FIELDS)
                .eq("job_id", str(claimed_row["job_id"]))
                .eq("freehire_compat_status", "processing")
                .limit(1)
                .execute().data or []
            )
            if fresh:
                batch.append(fresh[0])
        counts["claimed_elsewhere"] += len(pending_batch) - len(batch)
        if not batch:
            continue
        batch_id = str(uuid.uuid4())
        outcome = freehire_compat.classify_batch(
            batch,
            client=client,
            max_requests=request_budget,
        )
        request_budget -= outcome.requests
        counts["llm_requests"] += outcome.requests
        counts["retries"] += outcome.retries
        counts["splits"] += outcome.splits
        by_id = {str(row["job_id"]): row for row in batch}
        for job_id, classification in outcome.results.items():
            row = by_id[job_id]
            attempts = int(row.get("freehire_compat_attempts") or 0) + 1
            payload = freehire_compat.build_current_payload(
                row,
                classification,
                client=client,
                batch_id=batch_id,
                attempts=attempts,
                result_model=outcome.result_models.get(job_id),
            )
            payload["_expected_source_snapshot"] = freehire_compat.source_snapshot(row)
            if _persist(db, job_id, freehire_compat.compute_classification_hash(row), payload, worker_id):
                counts["classified"] += 1
            else:
                counts["claimed_elsewhere"] += 1
        for job_id, error in outcome.failures.items():
            row = by_id[job_id]
            attempts = int(row.get("freehire_compat_attempts") or 0) + 1
            payload = freehire_compat.build_failure_payload(row, error, client=client, attempts=attempts)
            payload["_expected_source_snapshot"] = freehire_compat.source_snapshot(row)
            if _persist(db, job_id, freehire_compat.compute_classification_hash(row), payload, worker_id):
                counts["failed"] += 1

    return counts


def result_status(result: dict) -> str:
    if result.get("failed", 0) and not result.get("classified", 0):
        return "all_failed"
    if result.get("failed", 0):
        return "partial_success"
    return "success"


def result_exit_code(result: dict) -> int:
    """Return nonzero only when all attempted classifications failed."""
    return int(result_status(result) == "all_failed")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=positive_int, default=int(os.getenv("FREEHIRE_CLASSIFY_LIMIT", config.FREEHIRE_CLASSIFY_LIMIT)))
    parser.add_argument("--drain-backlog", action="store_true", default=os.getenv("FREEHIRE_DRAIN_BACKLOG", "false").lower() == "true")
    parser.add_argument("--replacement-backfill", action="store_true", default=os.getenv("FREEHIRE_REPLACEMENT_BACKFILL", "false").lower() == "true")
    args = parser.parse_args(argv)
    result = run(
        apply=args.apply,
        limit=args.limit,
        drain_backlog=args.drain_backlog,
        replacement_backfill=args.replacement_backfill,
    )
    logging.info("Freehire compatibility status=%s stats=%s", result_status(result), result)
    print(result)
    return result_exit_code(result) if args.apply else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
