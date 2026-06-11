import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

from pydantic import BaseModel

import config
from llm_client import primary_client


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


def fetch_unanalyzed_jobs(db=None, limit=None) -> list:
    db = db or _get_db()
    query = (
        db.table(config.SUPABASE_TABLE_NAME)
        .select("job_id, job_title, description")
        .eq("is_active", True)
        .eq("job_state", "new")
        .is_("insights_analyzed_at", None)
        .not_.is_("description", None)
    )

    if limit is None:
        limit = config.JOB_INSIGHTS_MAX_JOBS

    return query.limit(limit).execute().data or []


def extract_keywords_from_batch(batch, client=None, max_retries=None) -> dict[str, list[KeywordItem]]:
    client = client or primary_client
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
                temperature=0.0,
                response_format=JobKeywordResultList,
            )
            return parse_keyword_response(raw_response)
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
        key = (fact["job_id"], fact["keyword"], fact["category"])
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
            .upsert(batch, on_conflict="job_id,keyword,category", ignore_duplicates=True)
            .execute()
        )
        inserted.extend(response.data or [])

    return inserted


def update_keyword_insights_from_facts(inserted_facts: list[dict], db=None):
    if not inserted_facts:
        return

    db = db or _get_db()
    timestamp = datetime.now(timezone.utc).isoformat()
    counts = defaultdict(int)
    for fact in inserted_facts:
        counts[(fact["keyword"], fact["category"])] += 1

    existing = {}
    offset = 0
    page_size = config.JOB_INSIGHTS_DB_PAGE_SIZE
    while True:
        rows = (
            db.table("keyword_insights")
            .select("keyword, category, count")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
            or []
        )
        if not rows:
            break
        for row in rows:
            existing[(row["keyword"], row["category"])] = row["count"]
        offset += page_size

    rows = [
        {
            "keyword": keyword,
            "category": category,
            "count": existing.get((keyword, category), 0) + count,
            "last_updated": timestamp,
        }
        for (keyword, category), count in counts.items()
    ]

    batch_size = config.JOB_INSIGHTS_UPSERT_BATCH_SIZE
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        db.table("keyword_insights").upsert(batch, on_conflict="keyword,category").execute()


def mark_jobs_analyzed(job_ids: list, db=None):
    if not job_ids:
        return

    db = db or _get_db()
    db.table(config.SUPABASE_TABLE_NAME).update(
        {"insights_analyzed_at": datetime.now(timezone.utc).isoformat()}
    ).in_("job_id", job_ids).execute()


def run(backfill_all: bool = False):
    db = _get_db()
    processed_jobs = 0

    while True:
        jobs = fetch_unanalyzed_jobs(db=db, limit=config.JOB_INSIGHTS_MAX_JOBS)
        if not jobs:
            if processed_jobs == 0:
                logger.info("No unanalyzed jobs found.")
            return processed_jobs

        batch_size = config.JOB_INSIGHTS_BATCH_SIZE
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start : start + batch_size]
            extracted = extract_keywords_from_batch(batch)
            facts = build_job_keyword_facts(batch, extracted)
            inserted_facts = upsert_job_keyword_facts(facts, db=db)
            update_keyword_insights_from_facts(inserted_facts, db=db)
            analyzed_job_ids = [str(job["job_id"]) for job in batch if job.get("job_id") is not None]
            mark_jobs_analyzed(analyzed_job_ids, db=db)
            processed_jobs += len(analyzed_job_ids)

        if not backfill_all:
            return processed_jobs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(backfill_all=os.getenv("JOB_INSIGHTS_BACKFILL_ALL", "false").lower() == "true")
