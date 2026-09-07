"""Backfill and incrementally maintain the Freehire compatibility contract. Dry-run by default."""

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
    "freehire_compat_confidence,freehire_compat_classified_at,freehire_compat_error,"
    "freehire_compat_attempts,freehire_compat_claimed_at,freehire_compat_claimed_by,"
    "freehire_compat_next_retry_at,"
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


def fetch_incremental_candidates(db, limit: int, now: datetime | None = None) -> list[dict]:
    """Fetch a bounded, actionable oldest-first work set without starvation."""
    now = now or datetime.now(timezone.utc)
    retry_cutoff = now.isoformat()
    lease_cutoff = (now - timedelta(minutes=30)).isoformat()
    stale_current = ",".join((
        "freehire_compat_input_hash.is.null",
        "freehire_compat_import_hash.is.null",
        "freehire_compat_model.is.null",
        "freehire_category.is.null",
        "freehire_compat_schema_version.is.null",
        f"freehire_compat_schema_version.neq.{config.FREEHIRE_COMPAT_SCHEMA_VERSION}",
        "freehire_compat_prompt_version.is.null",
        f"freehire_compat_prompt_version.neq.{config.FREEHIRE_COMPAT_PROMPT_VERSION}",
    ))
    eligibility = ",".join((
        "freehire_compat_status.eq.pending",
        "and(freehire_compat_status.eq.failed,"
        f"freehire_compat_attempts.lt.{config.FREEHIRE_CLASSIFY_MAX_DURABLE_ATTEMPTS},"
        f"or(freehire_compat_next_retry_at.is.null,freehire_compat_next_retry_at.lte.{retry_cutoff}))",
        "and(freehire_compat_status.eq.processing,"
        f"or(freehire_compat_claimed_at.is.null,freehire_compat_claimed_at.lt.{lease_cutoff}))",
        f"and(freehire_compat_status.eq.current,or({stale_current}))",
    ))
    return (
        db.table(config.SUPABASE_TABLE_NAME)
        .select(SELECT_FIELDS)
        .eq("provider", "linkedin")
        .not_.is_("description", None)
        .or_(eligibility)
        .order("last_seen_at", desc=False, nullsfirst=True)
        .order("job_id", desc=False)
        .limit(limit)
        .execute()
        .data
        or []
    )


def _valid_current(row: dict, input_hash: str) -> bool:
    return (
        row.get("freehire_compat_status") == "current"
        and row.get("freehire_compat_input_hash") == input_hash
        and row.get("freehire_compat_schema_version") == config.FREEHIRE_COMPAT_SCHEMA_VERSION
        and row.get("freehire_compat_prompt_version") == config.FREEHIRE_COMPAT_PROMPT_VERSION
        and bool(row.get("freehire_compat_model"))
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


def _rpc_job_ids(data) -> set[str]:
    if not data:
        return set()
    if isinstance(data, dict):
        data = [data]
    return {
        str(row.get("job_id") if isinstance(row, dict) else row)
        for row in data
        if row is not None
    }


def _claim_many(
    db,
    rows: list[dict],
    worker_id: str,
    replacement_before: datetime | None = None,
) -> set[str]:
    claims = [{
        "job_id": str(row["job_id"]),
        "expected_input_hash": freehire_compat.compute_classification_hash(row),
        "expected_source_snapshot": freehire_compat.source_snapshot(row),
    } for row in rows]
    params = {"p_claims": claims, "p_worker_id": worker_id}
    if replacement_before is not None:
        params["p_replacement_before"] = replacement_before.isoformat()
    return _rpc_job_ids(db.rpc("claim_freehire_compat_jobs", params).execute().data)


def _persist_many(db, results: list[dict], worker_id: str) -> set[str]:
    return _rpc_job_ids(db.rpc("persist_freehire_compat_results", {
        "p_results": results,
        "p_worker_id": worker_id,
    }).execute().data)


def _apply_metadata_many(db, updates: list[dict]) -> set[str]:
    return _rpc_job_ids(db.rpc("apply_freehire_compat_metadata_batch", {
        "p_updates": updates,
    }).execute().data)


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


def _replacement_cutoff(value: str | datetime | None) -> datetime | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("replacement_before must include a timezone")
    return parsed.astimezone(timezone.utc)


def run(
    apply: bool = False,
    limit: int | None = None,
    drain_backlog: bool = False,
    replacement_backfill: bool = False,
    replacement_before: str | datetime | None = None,
    db=None,
    client=None,
) -> dict:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    replacement_cutoff = _replacement_cutoff(replacement_before)
    if replacement_backfill and replacement_cutoff is None:
        raise ValueError(
            "replacement_backfill requires replacement_before so capped reruns resume safely"
        )
    db = db or _get_db()
    if not apply:
        client = None
    if client is None and apply:
        from llm_client import freehire_classify_client

        client = freehire_classify_client
    if drain_backlog and not replacement_backfill:
        page_limit = limit or config.FREEHIRE_CLASSIFY_PAGE_SIZE
        total = None
        while True:
            page_result = run(
                apply=apply,
                limit=page_limit,
                drain_backlog=False,
                db=db,
                client=client,
            )
            if total is None:
                total = {key: 0 for key in page_result}
            for key, value in page_result.items():
                total[key] += value
            if not apply or page_result["scanned"] == 0:
                break
            progress = (
                page_result["classified"]
                + page_result["failed"]
                + page_result["metadata_updated"]
            )
            if progress == 0:
                logging.warning(
                    "Freehire compatibility drain stopped because no candidate made progress"
                )
                break
        return total or {
            "scanned": 0, "unchanged": 0, "metadata_updated": 0,
            "would_classify": 0, "classified": 0, "failed": 0,
            "claimed_elsewhere": 0, "cooldown_or_exhausted": 0,
            "remote_true": 0, "remote_false": 0, "llm_requests": 0,
            "retries": 0, "splits": 0,
        }
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
    full_scan = drain_backlog or replacement_backfill
    upper_bound = get_upper_bound(db) if full_scan else None
    last_job_id = None
    pending: list[dict] = []
    metadata_updates: list[dict] = []
    while True:
        if full_scan:
            if upper_bound is None:
                break
            page = fetch_candidates(db, last_job_id=last_job_id, upper_bound=upper_bound)
        else:
            page = fetch_incremental_candidates(db, limit)
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
            valid_current = _valid_current(row, input_hash)
            classified_at = row.get("freehire_compat_classified_at")
            replacement_eligible = False
            if replacement_cutoff is not None and valid_current:
                if not classified_at:
                    replacement_eligible = True
                else:
                    try:
                        replacement_eligible = (
                            datetime.fromisoformat(str(classified_at).replace("Z", "+00:00"))
                            < replacement_cutoff
                        )
                    except ValueError:
                        replacement_eligible = True
            current = valid_current and not replacement_eligible
            deterministic_current = (
                row.get("is_remote") == is_remote
                and row.get("freehire_remote_evidence") == evidence
                and row.get("freehire_compat_import_hash") == import_hash
            )
            if current and deterministic_current:
                counts["unchanged"] += 1
                continue
            if current:
                if apply:
                    metadata_updates.append({
                        "job_id": str(row["job_id"]),
                        "expected_source_snapshot": freehire_compat.source_snapshot(row),
                        "payload": {
                            "is_remote": is_remote,
                            "freehire_remote_evidence": evidence,
                            "freehire_compat_import_hash": import_hash,
                        },
                    })
                else:
                    counts["metadata_updated"] += 1
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
        if not full_scan:
            break
        last_job_id = str(page[-1]["job_id"])
        if len(page) < config.FREEHIRE_CLASSIFY_PAGE_SIZE:
            break

    if apply and metadata_updates:
        applied_metadata = _apply_metadata_many(db, metadata_updates)
        counts["metadata_updated"] += len(applied_metadata)
        counts["claimed_elsewhere"] += len(metadata_updates) - len(applied_metadata)

    if not apply or not pending:
        return counts

    worker_id = str(uuid.uuid4())
    for pending_batch in freehire_compat.pack_batches(pending, model=freehire_compat.model_name(client)):
        claimed_ids = _claim_many(db, pending_batch, worker_id, replacement_cutoff)
        batch = [row for row in pending_batch if str(row["job_id"]) in claimed_ids]
        counts["claimed_elsewhere"] += len(pending_batch) - len(batch)
        if not batch:
            continue
        batch_id = str(uuid.uuid4())
        outcome = freehire_compat.classify_batch(
            batch,
            client=client,
        )
        counts["llm_requests"] += outcome.requests
        counts["retries"] += outcome.retries
        counts["splits"] += outcome.splits
        by_id = {str(row["job_id"]): row for row in batch}
        persistence = []
        result_ids = set(outcome.results)
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
            persistence.append({
                "job_id": job_id,
                "expected_input_hash": freehire_compat.compute_classification_hash(row),
                "expected_source_snapshot": freehire_compat.source_snapshot(row),
                "payload": payload,
            })
        for job_id, error in outcome.failures.items():
            row = by_id[job_id]
            attempts = int(row.get("freehire_compat_attempts") or 0) + 1
            payload = freehire_compat.build_failure_payload(row, error, client=client, attempts=attempts)
            persistence.append({
                "job_id": job_id,
                "expected_input_hash": freehire_compat.compute_classification_hash(row),
                "expected_source_snapshot": freehire_compat.source_snapshot(row),
                "payload": payload,
            })
        persisted = _persist_many(db, persistence, worker_id)
        counts["classified"] += len(persisted & result_ids)
        counts["failed"] += len(persisted - result_ids)
        counts["claimed_elsewhere"] += len(persistence) - len(persisted)

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
    parser.add_argument(
        "--replacement-before",
        default=os.getenv("FREEHIRE_REPLACEMENT_BEFORE"),
        help="Stable ISO-8601 cutoff required for resumable replacement backfills",
    )
    args = parser.parse_args(argv)
    result = run(
        apply=args.apply,
        limit=args.limit,
        drain_backlog=args.drain_backlog,
        replacement_backfill=args.replacement_backfill,
        replacement_before=args.replacement_before,
    )
    logging.info("Freehire compatibility status=%s stats=%s", result_status(result), result)
    print(result)
    return result_exit_code(result) if args.apply else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
