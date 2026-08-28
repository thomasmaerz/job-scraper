import json
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
    payload = json.loads(raw_response)
    if isinstance(payload, list):
        payload = {"jobs": payload}
    parsed = JobKeywordResultList.model_validate(payload)
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
    expected_jobs = {
        str(job["job_id"]): job
        for job in batch
        if job.get("job_id") is not None
    }
    pending_jobs = expected_jobs.copy()
    extracted = {}

    last_error = None
    for attempt in range(max_retries):
        try:
            prompt_lines = [
                "Extract keywords for each job posting individually.",
                "Return only structured JSON.",
            ]
            for job in pending_jobs.values():
                prompt_lines.append(f"Job ID: {job.get('job_id', '')}")
                prompt_lines.append(f"Title: {job.get('job_title', '')}")
                prompt_lines.append(f"Description: {job.get('description', '')}")
                prompt_lines.append("")

            raw_response = client.generate_content(
                prompt="\n".join(prompt_lines),
                system_prompt=SYSTEM_PROMPT,
                reasoning_effort="low",
                response_format=JobKeywordResultList,
            )
            parsed = parse_keyword_response(raw_response)
            extracted.update(
                (job_id, keywords)
                for job_id, keywords in parsed.items()
                if job_id in expected_jobs
            )
            pending_jobs = {
                job_id: job
                for job_id, job in expected_jobs.items()
                if job_id not in extracted
            }
            if pending_jobs:
                raise ValueError(
                    f"Missing keyword results for job_ids: {', '.join(sorted(pending_jobs))}"
                )
            return extracted
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
    """Reject unsafe fact-only writes that would bypass aggregate maintenance."""
    if not facts:
        return []
    raise RuntimeError(
        "direct job keyword fact upserts are disabled; use replace_job_keyword_facts"
    )


def replace_job_keyword_facts(job_ids: list[str], facts: list[dict], archetype: str | None = None, db=None) -> list[dict]:
    """Atomically replace facts and apply their aggregate contribution delta."""
    db = db or _get_db()

    normalized_job_ids = list(dict.fromkeys(str(job_id) for job_id in job_ids))
    if not normalized_job_ids:
        return []

    fact_archetypes = {fact.get("archetype") for fact in facts}
    if archetype is None:
        if len(fact_archetypes) != 1:
            raise ValueError("archetype is required when replacing an empty or mixed-archetype fact set")
        archetype = next(iter(fact_archetypes))
    if not archetype:
        raise ValueError("archetype must be non-empty")
    if any(fact.get("archetype") != archetype for fact in facts):
        raise ValueError("all replacement facts must match archetype")

    allowed_job_ids = set(normalized_job_ids)
    deduped = []
    seen = set()
    for fact in facts:
        normalized = dict(fact)
        normalized["job_id"] = str(fact["job_id"])
        if normalized["job_id"] not in allowed_job_ids:
            raise ValueError("replacement facts must belong to p_job_ids")
        normalized["archetype"] = archetype
        key = (normalized["job_id"], archetype, normalized["keyword"], normalized["category"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)

    db.rpc(
        "replace_job_keyword_facts_and_refresh_aggregates",
        {
            "p_job_ids": normalized_job_ids,
            "p_archetype": archetype,
            "p_facts": deduped,
        },
    ).execute()
    return deduped


def _normalize_provider(provider) -> str:
    return provider or "unknown"


def update_keyword_insights_from_facts(source_facts: list[dict], db=None, affected_keys=None):
    """Compatibility entry point; aggregate repair uses the locked rebuild RPC."""
    if not source_facts and not affected_keys:
        return None
    return rebuild_keyword_insights(db=db)


def rebuild_keyword_insights(db=None):
    """Atomically rebuild aggregate counts from all persisted keyword facts."""
    db = db or _get_db()
    return db.rpc("rebuild_keyword_insights_atomic").execute().data


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
