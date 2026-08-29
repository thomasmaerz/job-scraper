"""Plan and apply high-confidence historical repost consolidation."""

import argparse
from collections import defaultdict

import config
import supabase_utils


SELECT_FIELDS = (
    "job_id,company,job_title,description,description_fingerprint,scraped_at,last_seen_at,"
    "status,application_date,notes,customized_resume_id,is_interested,insights_analyzed_at,"
    "applicant_count,salary_text,recruiter_name,level,location,listing_instances"
)


def distinct_locations(row: dict) -> set[str]:
    locations = {supabase_utils.normalize_location(row.get("location"))}
    locations.update(
        supabase_utils.normalize_location(instance.get("location"))
        for instance in (row.get("listing_instances") or [])
        if isinstance(instance, dict)
    )
    locations.discard("")
    return locations


def audit_merge_plan(rows: list[dict], plan: list[dict]) -> dict:
    by_id = {str(row["job_id"]): row for row in rows}
    grouped = defaultdict(set)
    method_counts = defaultdict(int)
    for item in plan:
        grouped[str(item["survivor_job_id"])].add(str(item["source_job_id"]))
        method_counts[str(item["match_method"])] += 1

    groups = []
    missing_ids = set()
    for survivor_id, source_ids in sorted(grouped.items()):
        member_ids = {survivor_id, *source_ids}
        missing_ids.update(member_ids - by_id.keys())
        source_locations = set()
        for member_id in member_ids & by_id.keys():
            source_locations.update(distinct_locations(by_id[member_id]))
        groups.append({
            "survivor_job_id": survivor_id,
            "source_job_ids": sorted(source_ids),
            "source_locations": sorted(source_locations),
        })

    return {
        "groups": groups,
        "group_count": len(groups),
        "source_count": len(plan),
        "method_counts": dict(sorted(method_counts.items())),
        "missing_job_ids": sorted(missing_ids),
    }


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
    workflow_rank = {
        "offer": 4,
        "interviewing": 3,
        "applied": 2,
    }.get(row.get("status"), 1 if row.get("application_date") is not None else 0)
    protected = (
        workflow_rank,
        row.get("customized_resume_id") is not None,
        bool(row.get("notes")),
        row.get("is_interested") is True,
    )
    observed_at = row.get("last_seen_at") or row.get("scraped_at") or ""
    return protected, _completeness(row), observed_at, str(row["job_id"])


def build_merge_plan(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        company = supabase_utils.normalize_company(row.get("company"))
        if company:
            buckets[company].append(row)

    plan = []
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
                matched, method, similarity = supabase_utils.is_high_confidence_repost_match(canonical, candidate)
                if matched:
                    cluster.append(candidate)
                    methods[candidate["job_id"]] = (method, similarity)
                else:
                    remaining.append(candidate)
            unclaimed = remaining
            if len(cluster) < 2:
                continue
            survivor = max(cluster, key=_survivor_key)
            for source in cluster:
                if source["job_id"] == survivor["job_id"]:
                    continue
                method, similarity = methods.get(source["job_id"], (None, None))
                if method is None:
                    matched, method, similarity = supabase_utils.is_high_confidence_repost_match(source, survivor)
                    if not matched:
                        continue
                plan.append(
                    {
                        "source_job_id": source["job_id"],
                        "survivor_job_id": survivor["job_id"],
                        "match_method": method,
                        "match_similarity": similarity,
                    }
                )
    return plan


def run(apply: bool, body_hash_only: bool = True) -> dict:
    jobs = fetch_jobs()
    plan = build_merge_plan(jobs)
    if body_hash_only:
        plan = [
            row for row in plan
            if row["match_method"] in {"exact_fingerprint", "body_hash_fuzzy_title"}
        ]
    summary = {
        "groups": len({row["survivor_job_id"] for row in plan}),
        "redundant_rows": len(plan),
        "exact": sum(row["match_method"] == "exact_fingerprint" for row in plan),
        "body_hash_fuzzy_title": sum(row["match_method"] == "body_hash_fuzzy_title" for row in plan),
        "fuzzy": sum(row["match_method"] == "fuzzy_description" for row in plan),
        "audit": audit_merge_plan(jobs, plan),
    }
    if not apply or not plan:
        return summary
    all_exact = all(row["match_method"] in {"exact_fingerprint", "body_hash_fuzzy_title"} for row in plan)
    if not all_exact or summary["audit"]["missing_job_ids"]:
        raise ValueError("Refusing to apply a merge plan that is not exact and complete")
    summary["staged_rows"] = supabase_utils.supabase.rpc(
        "replace_historical_repost_plan", {"p_plan": plan}
    ).execute().data
    summary["merge_result"] = supabase_utils.supabase.rpc("merge_historical_repost_plan").execute().data
    import analyze_jobs

    analyze_jobs.rebuild_keyword_insights(db=supabase_utils.supabase)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-fuzzy", action="store_true")
    args = parser.parse_args()
    print(run(apply=args.apply, body_hash_only=not args.include_fuzzy))
