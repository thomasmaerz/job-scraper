# Stage 2: Compliant Parallel Fetch Workers

## Objective

Scale lane and geography coverage without running concurrent canonical database writers or treating multiple cloud IP addresses as a way to bypass source controls. Parallelism is for durable execution and latency hiding. Aggregate LinkedIn request volume remains governed by one source-wide policy.

The current scraper uses LinkedIn guest jobs endpoints, not the authenticated Voyager API:

- `https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search`
- `https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}`

Before implementation, review LinkedIn's current terms and robots guidance and prefer an authorized API or licensed data source where available. The design must stop rather than evade access controls, challenges, or explicit denials.

## Preconditions

Stage 2 starts only when production measurements show that one optimized worker cannot keep p95 hourly source work below 40–45 minutes. Required measurements:

- Requests and responses by endpoint, lane, query, and geography.
- Search-card and unique-detail yield.
- 429, 403, challenge, timeout, and 5xx rates.
- Retry and cooldown time.
- Queue age and p50/p95 task duration.
- Canonical writer throughput and conflict rate.

Do not add workers merely because GitHub Actions capacity is available.

## Architecture

### 1. Durable scrape tasks

Create one idempotent task for a bounded fetch shard:

```text
(scheduled_bucket, config_revision, archetype, query_id, geography_id)
```

Task fields:

- `id`, `idempotency_key`, `scheduled_for`, `config_revision`
- `archetype`, `query_id`, `geography_id`, `priority`
- `status`, `available_at`, `attempt_count`, `last_error_code`
- `leased_by`, `lease_expires_at`, `heartbeat_at`
- request/card/detail metrics and timestamps

Claim with a transaction using `FOR UPDATE SKIP LOCKED`. Expired leases become eligible again. Cap attempts and move terminal failures to a visible dead-letter state.

### 2. Source-wide distributed limiter

Every worker must acquire permission before every LinkedIn request. The limiter is shared across runners and endpoints and enforces:

- One aggregate minimum request interval.
- Bounded jitter.
- A maximum request budget per rolling window.
- `Retry-After` handling.
- Exponential backoff with full jitter for 429 and transient 5xx responses.
- A global cooldown/circuit breaker after repeated 429, 403, challenge, or denial responses.

Workers must not rotate identities, proxies, accounts, cookies, or IPs to evade limits. A different GitHub runner IP does not grant a separate request budget.

Start with one fetch worker. Increase to two only after an observation period shows low error rates and only while both share the same aggregate limiter. If the limiter grants the same total request rate, workers improve resilience and latency hiding, not total source pressure.

### 3. Immutable raw staging

Fetch workers write immutable observations, not canonical jobs:

```text
raw_search_observations
raw_job_details
```

Use stable uniqueness constraints such as:

```text
(task_id, source_job_id)
(source_job_id, content_hash)
```

Store request provenance, observed timestamps, response classification, and content hashes. Never mark a task complete before all staged rows commit.

Search observations from overlapping lanes remain separate provenance records. Detail payloads are deduplicated by source job ID/content hash so one successful fetch can satisfy all matching observations.

### 4. Single canonical writer

One writer drains unapplied staged observations in deterministic order. It alone performs:

- Fuzzy canonical matching.
- Canonical job insert/update.
- Listing-state and relist changes.
- Lane membership/filter-state writes.
- Applied markers and lane watermarks.

Enforce this independently of GitHub Actions with a PostgreSQL advisory lock. GitHub concurrency remains defense in depth. A crash leaves staging intact; the next writer resumes idempotently.

Do not permit multiple canonical writers until matching is moved into transactional database operations with suitable unique constraints, advisory locks for fuzzy buckets, and compare-and-swap listing-state updates.

### 5. Downstream workers

- Freehire continues to use its existing claim/lease/CAS protocol.
- Insights analysis receives equivalent claims before parallel execution.
- Publication reads only committed canonical/downstream state.
- Publication generation creation and pointer switching remain atomic.
- Old-generation pruning runs separately in bounded, resumable maintenance.

## Scheduling and fairness

The scheduler inserts all enabled work idempotently. Priority order:

1. Overdue lane/geography work.
2. Failed retry-ready work.
3. Current hourly precision searches.
4. Current hourly recall searches.
5. Lower-frequency expansion work.

Use per-lane/geography quotas so a high-yield lane cannot starve smaller lanes. Queue age, not workflow age, is the freshness authority.

## Failure behavior

- Network timeout: retry with bounded backoff.
- 429: honor `Retry-After`, open global cooldown when threshold is crossed.
- 403/challenge/denial: stop affected source work and alert; do not work around it.
- Worker termination: lease expires and another worker resumes.
- Staging write failure: task remains incomplete.
- Canonical write failure: staged observation remains unapplied.
- Publication/pruning failure: retain the prior complete generation; retry maintenance independently.

## Rollout

1. Add queue/staging schema and read-only metrics while retaining one worker.
2. Route one low-risk lane through staging and compare outputs with the current path.
3. Move all lanes to staging with one fetch worker and one canonical writer.
4. Validate idempotency by terminating workers mid-task and confirming lease recovery.
5. Enable a second fetch worker behind the shared limiter for a canary window.
6. Automatically return to one worker when 429/403/challenge or queue error thresholds rise.

## Acceptance criteria

- No concurrent canonical writers in production.
- Every source request obtains a shared limiter grant.
- Duplicate task delivery does not duplicate observations, canonical jobs, or memberships.
- Killing any worker requires no manual database repair.
- Every enabled lane/geography exposes last-success and queue-age metrics.
- Repeated source denial opens a global cooldown and stops requests.
- p95 freshness meets the configured SLA without increasing aggregate request rate beyond the approved budget.
- A rollback restores the single-worker path without data loss.

## References

- GitHub-hosted runners and dynamic IP ranges: https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- GitHub Actions concurrency: https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency
- Scheduled workflow limitations: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule
- Supabase/Postgres timeout configuration: https://supabase.com/docs/guides/database/postgres/timeouts
- PostgreSQL `SKIP LOCKED`: https://www.postgresql.org/docs/current/sql-select.html
- PostgreSQL advisory locks: https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS
