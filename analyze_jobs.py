import json
import logging
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


def _get_db():
    from supabase import create_client

    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("Supabase URL and Key must be set in environment variables or config.")

    return create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)


def _normalize_keyword(keyword: str) -> str:
    return keyword.strip().title()


def _normalize_category(category: str) -> str:
    return category.strip().lower()


def parse_keyword_response(raw_response: str) -> list[KeywordItem]:
    parsed = KeywordList.model_validate_json(raw_response)
    return parsed.keywords


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


def extract_keywords_from_batch(batch, client=None, max_retries=None) -> list[KeywordItem]:
    client = client or primary_client
    max_retries = config.JOB_INSIGHTS_MAX_RETRIES if max_retries is None else max_retries

    prompt_lines = [
        "Extract recurring job-market keywords from these job postings.",
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
                response_format=KeywordList,
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


def _fetch_existing_insights(db) -> dict:
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

    return existing


def upsert_insights(counts: dict, db=None):
    if not counts:
        return

    db = db or _get_db()
    existing = _fetch_existing_insights(db)
    timestamp = datetime.now(timezone.utc).isoformat()

    rows = []
    for (keyword, category), count in counts.items():
        rows.append(
            {
                "keyword": keyword,
                "category": category,
                "count": existing.get((keyword, category), 0) + count,
                "last_updated": timestamp,
            }
        )

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


def run():
    db = _get_db()
    jobs = fetch_unanalyzed_jobs(db=db, limit=config.JOB_INSIGHTS_MAX_JOBS)
    if not jobs:
        logger.info("No unanalyzed jobs found.")
        return

    all_keywords = []
    analyzed_job_ids = []
    batch_size = config.JOB_INSIGHTS_BATCH_SIZE

    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        extracted = extract_keywords_from_batch(batch)
        all_keywords.extend(extracted)
        analyzed_job_ids.extend(str(job["job_id"]) for job in batch if job.get("job_id") is not None)

    counts = aggregate_keywords(all_keywords)
    upsert_insights(counts, db=db)
    mark_jobs_analyzed(analyzed_job_ids, db=db)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
