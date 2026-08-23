"""Plan and apply high-confidence historical repost consolidation."""

import argparse
from collections import defaultdict

import config
import analyze_jobs
import supabase_utils


SELECT_FIELDS = (
    "job_id,company,job_title,description,description_fingerprint,scraped_at,last_seen_at,"
    "status,application_date,notes,customized_resume_id,is_interested,insights_analyzed_at,"
    "applicant_count,salary_text,recruiter_name,level,location"
)


def fetch_jobs(page_size: int = 1000) -> list[dict]:
    rows = []
    offset = 0
    while True:
        page = (
            supabase_utils.supabase.table(config.SUPABASE_TABLE_NAME)
            .select(SELECT_FIELDS)
            .eq("provider", "linkedin")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
            or []
        )
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += page_size


def _completeness(row: dict) -> int:
    fields = (
        "description",
        "level",
        "location",
        "applicant_count",
        "salary_text",
        "recruiter_name",
        "insights_analyzed_at",
    )
    return sum(row.get(field) is not None for field in fields)


def _survivor_key(row: dict) -> tuple:
    protected = sum(
        (
            row.get("status") in {"applied", "interviewing", "offer"},
            row.get("application_date") is not None,
            bool(row.get("notes")),
            row.get("customized_resume_id") is not None,
            row.get("is_interested") is True,
        )
    )
    observed_at = row.get("last_seen_at") or row.get("scraped_at") or ""
    return protected, _completeness(row), observed_at, str(row["job_id"])


def build_merge_plan(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        key = (
            supabase_utils.normalize_company(row.get("company")),
            supabase_utils.normalize_role_title(row.get("job_title")),
            supabase_utils.normalize_location(row.get("location")),
        )
        if all(key):
            buckets[key].append(row)

    plan = []
    threshold = getattr(config, "REPOST_DESCRIPTION_SIMILARITY_THRESHOLD", 0.90)
    for bucket in buckets.values():
        if len(bucket) < 2 or len(bucket) > 200:
            continue
        unclaimed = sorted(bucket, key=lambda row: str(row["job_id"]))
        while unclaimed:
            canonical = unclaimed.pop(0)
            cluster = [canonical]
            remaining = []
            methods = {}
            for candidate in unclaimed:
                exact = (
                    canonical.get("description_fingerprint")
                    and canonical.get("description_fingerprint") == candidate.get("description_fingerprint")
                )
                similarity = supabase_utils.description_similarity(
                    canonical.get("description"), candidate.get("description")
                )
                if exact or similarity >= threshold:
                    cluster.append(candidate)
                    methods[candidate["job_id"]] = ("exact_fingerprint" if exact else "fuzzy_description", similarity)
                else:
                    remaining.append(candidate)
            unclaimed = remaining
            if len(cluster) < 2:
                continue
            survivor = max(cluster, key=_survivor_key)
            for source in cluster:
                if source["job_id"] == survivor["job_id"]:
                    continue
                method, similarity = methods.get(source["job_id"], ("cluster_member", None))
                plan.append(
                    {
                        "source_job_id": source["job_id"],
                        "survivor_job_id": survivor["job_id"],
                        "match_method": method,
                        "match_similarity": similarity,
                    }
                )
    return plan


def run(apply: bool) -> dict:
    plan = build_merge_plan(fetch_jobs())
    summary = {
        "groups": len({row["survivor_job_id"] for row in plan}),
        "redundant_rows": len(plan),
        "exact": sum(row["match_method"] == "exact_fingerprint" for row in plan),
        "fuzzy": sum(row["match_method"] == "fuzzy_description" for row in plan),
    }
    if not apply or not plan:
        return summary
    supabase_utils.supabase.table("job_repost_merge_plan").delete().neq("source_job_id", "").execute()
    for start in range(0, len(plan), 200):
        supabase_utils.supabase.table("job_repost_merge_plan").insert(plan[start:start + 200]).execute()
    summary["merge_result"] = supabase_utils.supabase.rpc("merge_historical_repost_plan").execute().data
    analyze_jobs.rebuild_keyword_insights(db=supabase_utils.supabase)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(run(apply=args.apply))
