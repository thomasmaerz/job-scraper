import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

from pydantic import BaseModel

import config
from llm_client import job_insights_client


logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"skill", "technology", "certification", "attribute"}

SYSTEM_PROMPT = """You extract high-signal job market keywords from batches of job postings.

Return JSON only using the provided schema. Include concise keywords that appear explicitly in the postings.
Use only these categories: skill, technology, certification, attribute.
Avoid duplicates within the same batch unless they are distinct keywords.
"""


class KeywordItem(BaseModel):
    keyword: str
    category: str


class KeywordList(BaseModel):
    keywords: list[KeywordItem]


class JobKeywordResult(BaseModel):
    job_id: str
    keywords: list[KeywordItem]


class JobKeywordResultList(BaseModel):
    jobs: list[JobKeywordResult]


def _get_db():
    from supabase import create_client

    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("Supabase URL and Key must be set in environment variables or config.")

    return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)


def _normalize_keyword(keyword: str) -> str:
    cleaned = keyword.strip()
    if not cleaned:
        return ""
    if cleaned.isupper():
        return cleaned
    return " ".join(part[:1].upper() + part[1:] for part in cleaned.split())


def _normalize_category(category: str) -> str:
    return category.strip().lower()


def parse_keyword_response(raw_response: str) -> dict[str, list[KeywordItem]]:
    parsed = JobKeywordResultList.model_validate_json(raw_response)
    return {job.job_id: job.keywords for job in parsed.jobs}


def fetch_unanalyzed_jobs(
    db=None,
    limit=None,
    archetype: str = config.DEFAULT_ARCHETYPE,
    backfill_all: bool = False,
    replacement_backfill: bool = False,
) -> list:
    db = db or _get_db()
    query = (
        db.table(config.SUPABASE_TABLE_NAME)
        .select("job_id, job_title, description, archetype, provider")
        .eq("is_filtered", False)
        .eq("archetype", archetype)
    )

    if not backfill_all and not replacement_backfill:
        query = query.eq("is_active", True).eq("job_state", "new")

    if replacement_backfill:
        query = query.not_.is_("insights_analyzed_at", None).is_("insights_reanalyzed_at", None)
    else:
        query = query.is_("insights_analyzed_at", None)

    query = query.not_.is_("description", None)

    if limit is None:
        limit = config.JOB_INSIGHTS_MAX_JOBS

    return query.limit(limit).execute().data or []


def extract_keywords_from_batch(batch, client=None, max_retries=None) -> dict[str, list[KeywordItem]]:
    client = client or job_insights_client
    max_retries = config.JOB_INSIGHTS_MAX_RETRIES if max_retries is None else max_retries

    prompt_lines = [
        "Extract keywords for each job posting individually.",
        "Return only structured JSON.",
    ]
    for job in batch:
        prompt_lines.append(f"Job ID: {job.get('job_id', '')}")
        prompt_lines.append(f"Title: {job.get('job_title', '')}")
        prompt_lines.append(f"Description: {job.get('description', '')}")
        prompt_lines.append("")

    prompt = "\n".join(prompt_lines)

    last_error = None
    for attempt in range(max_retries):
        try:
            raw_response = client.generate_content(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                reasoning_effort="low",
                response_format=JobKeywordResultList,
            )
            parsed = parse_keyword_response(raw_response)
            expected_job_ids = {str(job.get("job_id")) for job in batch if job.get("job_id") is not None}
            missing_job_ids = sorted(job_id for job_id in expected_job_ids if job_id not in parsed)
            if missing_job_ids:
                raise ValueError(
                    f"Missing keyword results for job_ids: {', '.join(missing_job_ids)}"
                )
            return parsed
        except Exception as exc:
            last_error = exc
            logger.warning("Keyword extraction failed on attempt %s: %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(config.JOB_INSIGHTS_SLEEP_SECONDS)

    if last_error is not None:
        raise last_error
    return []


def aggregate_keywords(all_keywords: list[KeywordItem]) -> dict:
    counts = defaultdict(int)
    for item in all_keywords:
        category = _normalize_category(item.category)
        if category not in VALID_CATEGORIES:
            continue

        keyword = _normalize_keyword(item.keyword)
        if not keyword:
            continue

        counts[(keyword, category)] += 1

    return dict(counts)


def build_job_keyword_facts(batch: list, extracted_keywords: dict[str, list[KeywordItem]]) -> list[dict]:
    facts = []
    for job in batch:
        job_id = job.get("job_id")
        if job_id is None:
            continue
        archetype = job.get("archetype")
        provider = job.get("provider")
        for item in extracted_keywords.get(str(job_id), []):
            category = _normalize_category(item.category)
            keyword = _normalize_keyword(item.keyword)
            if not keyword or category not in VALID_CATEGORIES:
                continue
            facts.append(
                {
                    "job_id": str(job_id),
                    "keyword": keyword,
                    "category": category,
                    "archetype": archetype,
                    "provider": provider,
                }
            )
    return facts


def upsert_job_keyword_facts(facts: list[dict], db=None) -> list[dict]:
    if not facts:
        return []

    db = db or _get_db()
    deduped = []
    seen = set()
    for fact in facts:
        key = (fact["job_id"], fact.get("archetype"), fact["keyword"], fact["category"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)

    batch_size = config.JOB_INSIGHTS_UPSERT_BATCH_SIZE
    inserted = []
    for start in range(0, len(deduped), batch_size):
        batch = deduped[start : start + batch_size]
        response = (
            db.table("job_keyword_insights")
            .upsert(batch, on_conflict="job_id,archetype,keyword,category", ignore_duplicates=True)
            .execute()
        )
        inserted.extend(response.data or [])

    return inserted


def replace_job_keyword_facts(job_ids: list[str], facts: list[dict], archetype: str | None = None, db=None) -> list[dict]:
    db = db or _get_db()

    delete_keys = []
    seen_delete_keys = set()
    for fact in facts:
        key = (fact["job_id"], fact.get("archetype"))
        if key in seen_delete_keys:
            continue
        seen_delete_keys.add(key)
        delete_keys.append(key)

    fact_job_ids = {fact["job_id"] for fact in facts}
    for job_id in job_ids:
        if job_id in fact_job_ids:
            continue
        delete_query = db.table("job_keyword_insights").delete().eq("job_id", job_id)
        if archetype is not None:
            delete_query = delete_query.eq("archetype", archetype)
        delete_query.execute()

    for job_id, archetype in delete_keys:
        db.table("job_keyword_insights").delete().eq("job_id", job_id).eq("archetype", archetype).execute()

    if not facts:
        return []

    deduped = []
    seen = set()
    for fact in facts:
        key = (fact["job_id"], fact.get("archetype"), fact["keyword"], fact["category"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)

    batch_size = config.JOB_INSIGHTS_UPSERT_BATCH_SIZE
    inserted = []
    for start in range(0, len(deduped), batch_size):
        batch = deduped[start : start + batch_size]
        response = (
            db.table("job_keyword_insights")
            .upsert(batch, on_conflict="job_id,archetype,keyword,category")
            .execute()
        )
        inserted.extend(response.data or [])

    return inserted


def update_keyword_insights_from_facts(source_facts: list[dict], db=None, affected_keys=None):
    if not source_facts and not affected_keys:
        return

    db = db or _get_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    affected_keys = affected_keys or {
        (fact.get("archetype"), fact["keyword"], fact["category"]) for fact in source_facts
    }
    counts = defaultdict(int)
    offset = 0
    page_size = config.JOB_INSIGHTS_DB_PAGE_SIZE
    while True:
        rows = (
            db.table("job_keyword_insights")
            .select("job_id, keyword, category, archetype")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
            or []
        )
        if not rows:
            break
        for row in rows:
            key = (row.get("archetype"), row["keyword"], row["category"])
            if key in affected_keys:
                counts[key] += 1
        offset += page_size

    rows = [
        {
            "archetype": archetype,
            "keyword": keyword,
            "category": category,
            "count": counts[(archetype, keyword, category)],
            "last_updated": timestamp,
        }
        for (archetype, keyword, category) in affected_keys
    ]

    batch_size = config.JOB_INSIGHTS_UPSERT_BATCH_SIZE
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        db.table("keyword_insights").upsert(batch, on_conflict="archetype,keyword,category").execute()


def rebuild_keyword_insights(db=None):
    db = db or _get_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    counts = defaultdict(int)
    offset = 0
    page_size = config.JOB_INSIGHTS_DB_PAGE_SIZE

    while True:
        rows = (
            db.table("job_keyword_insights")
            .select("keyword, category, archetype")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
            or []
        )
        if not rows:
            break

        for row in rows:
            counts[(row.get("archetype"), row["keyword"], row["category"])] += 1
        offset += page_size

    rebuilt_rows = [
        {
            "archetype": archetype,
            "keyword": keyword,
            "category": category,
            "count": count,
            "last_updated": timestamp,
        }
        for (archetype, keyword, category), count in counts.items()
    ]

    batch_size = config.JOB_INSIGHTS_UPSERT_BATCH_SIZE
    for start in range(0, len(rebuilt_rows), batch_size):
        batch = rebuilt_rows[start : start + batch_size]
        db.table("keyword_insights").upsert(batch, on_conflict="archetype,keyword,category").execute()

    rebuilt_keys = set(counts.keys())
    existing_keys = set()
    offset = 0
    while True:
        rows = (
            db.table("keyword_insights")
            .select("keyword, category, archetype")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
            or []
        )
        if not rows:
            break

        for row in rows:
            existing_keys.add((row.get("archetype"), row["keyword"], row["category"]))
        offset += page_size

    stale_keys = existing_keys - rebuilt_keys
    for archetype, keyword, category in stale_keys:
        db.table("keyword_insights").delete().eq("keyword", keyword).eq("category", category).eq("archetype", archetype).execute()


def mark_jobs_analyzed(job_ids: list, db=None, replacement_backfill: bool = False):
    if not job_ids:
        return

    db = db or _get_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {"insights_analyzed_at": timestamp}
    if replacement_backfill:
        payload["insights_reanalyzed_at"] = timestamp

    db.table(config.SUPABASE_TABLE_NAME).update(payload).in_("job_id", job_ids).execute()


def run(archetype: str = config.DEFAULT_ARCHETYPE, backfill_all: bool = False, replacement_backfill: bool = False):
    db = _get_db()
    processed_jobs = 0

    while True:
        jobs = fetch_unanalyzed_jobs(
            db=db,
            limit=config.JOB_INSIGHTS_MAX_JOBS,
            archetype=archetype,
            backfill_all=backfill_all,
            replacement_backfill=replacement_backfill,
        )
        if not jobs:
            if processed_jobs == 0:
                logger.info("No unanalyzed jobs found.")
            return processed_jobs

        batch_size = config.JOB_INSIGHTS_BATCH_SIZE
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start : start + batch_size]
            extracted = extract_keywords_from_batch(batch)
            facts = build_job_keyword_facts(batch, extracted)
            analyzed_job_ids = [str(job["job_id"]) for job in batch if job.get("job_id") is not None]
            replace_job_keyword_facts(analyzed_job_ids, facts, archetype=archetype, db=db)
            rebuild_keyword_insights(db=db)
            mark_jobs_analyzed(analyzed_job_ids, db=db, replacement_backfill=replacement_backfill)
            processed_jobs += len(analyzed_job_ids)

        if not backfill_all and not replacement_backfill:
            return processed_jobs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(
        archetype=os.getenv("JOB_INSIGHTS_ARCHETYPE", config.DEFAULT_ARCHETYPE),
        backfill_all=os.getenv("JOB_INSIGHTS_BACKFILL_ALL", "false").lower() == "true",
        replacement_backfill=os.getenv("JOB_INSIGHTS_REPLACEMENT_BACKFILL", "false").lower() == "true",
    )
