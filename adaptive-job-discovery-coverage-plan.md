# Adaptive Job Discovery and Coverage Plan

## Status

Validated planning document; not yet implemented. Repository validation completed September 3, 2026 against `main` at `ad92604`. The validation corrected the cycle barrier, durable queue, lease fencing, scope identity, adaptive fairness, lookback-gap, and migration ownership contracts. Implementation remains blocked until the Phase 0 evidence artifact and cross-repository migration ownership matrix described below are checked in.

This plan does not claim exhaustive LinkedIn coverage and does not authorize bypassing source controls. It preserves one canonical writer and requires one source-wide LinkedIn request policy across every search, detail, activity-check, backfill, and recovery producer. The current code has only a limiter shared inside one scraper process; workflow concurrency is not yet a source-wide limiter.

Related design: `stage2-parallelworkersplan.md` covers durable fetch workers if a single optimized worker later proves insufficient.

Before implementation, review LinkedIn's current terms and robots guidance and prefer an authorized API or licensed inventory feed if available. Record the review date and approved request policy in the rollout evidence. No optimization in this plan is a substitute for source authorization.

## Executive decision

The current fixed six-page search does not reliably catch up after a high-volume period. The scrape watermark records when an operationally successful run finished; it is not a cursor through LinkedIn's ranked inventory. A job can remain below the first 60 results for its entire eligible time window and never be observed.

The efficient path toward high recall is:

1. Use short, elapsed-time-based recent windows for normal runs so new jobs compete in a smaller result set.
2. Adapt page depth when a scope is saturated, within a conservative hard limit and the existing global request cadence.
3. Persist discovered source IDs before fetching details and drain them from a durable queue, so a per-query detail budget cannot discard discoveries.
4. Track per-scope saturation and coverage debt separately from the operational scrape watermark.
5. Run bounded off-peak recovery sweeps for unresolved coverage debt.
6. Partition persistently saturated search scopes by legitimate geography or query intent rather than relying only on deeper ranked pagination.
7. Reduce Supabase traffic so database overhead does not consume the source-work budget.

Even after these changes, an unauthenticated ranked guest endpoint cannot prove that all matching LinkedIn jobs were exposed. The target is measurable, bounded, high-recall coverage of the configured search surface.

## Current evidence

### Fixed-depth saturation

The figures in this section are production observations, not behavior enforced by the tracked configuration contract. Before Phase 1, check in a sanitized evidence artifact containing the configuration revision or content hash, extraction timestamp, source queries/log filters, deduplication definitions, and page-level output. Until that artifact exists, use these figures to motivate measurement, not as a reproducible acceptance baseline.

The 96-hour recovery run `33828299410` completed 40 query scopes and six pages per scope:

- 240 search requests.
- 2,171 per-query unique observations.
- 1,073 workflow-distinct source IDs.
- 69 detail requests.
- 20 minutes for the scrape job.

Page yield was:

| Page | Offset | Cards | New IDs versus prior pages in the same query |
|---:|---:|---:|---:|
| 1 | 0 | 400 | 400 |
| 2 | 10 | 399 | 357 |
| 3 | 20 | 400 | 341 |
| 4 | 30 | 400 | 362 |
| 5 | 40 | 400 | 355 |
| 6 | 50 | 400 | 356 |

Page 6 was full for every scope and continued to add IDs. Six pages are therefore a restored baseline, not an observed exhaustion point.

### Delayed discoveries from the recovery

The recovery produced 67 distinct source IDs with no earlier observation. Fifty were inferred to have appeared on pages 4-6. Fourteen were posted by September 2, including old listings found only through the recovery. Examples include:

- `4348329919`, Intermediate/Senior Field Tech, posted August 31, inferred page 4/rank 32.
- `4456526394`, Manager, Internal Controls, posted August 31, inferred page 4/rank 33.
- `4461097846`, Auditeur Interne, posted August 31, inferred page 5/rank 37.
- `4461213022`, Senior Internal Controls Auditor, posted August 31, inferred page 5/rank 42.
- `4461253632`, Building Systems & Quality Assurance Specialist, posted August 31, inferred page 6/rank 56.

These records prove delayed discovery and specific prior-query misses. They do not prove how LinkedIn ranked the jobs in every intervening run, and the 96-hour recovery changed both page depth and lookback.

### Existing durable evidence and its gap

The current scraper already writes `listing_observations` and `listing_states` for discovered cards before fetching details. That is valuable durable evidence, but it is not an executable detail queue:

- A source ID skipped by the 50-detail budget has no explicit pending-detail state.
- The next run can reconsider it only if the ranked search exposes it again.
- `save_listing_observations` currently skips a card when `posted_at` is absent even though the database column is nullable.
- Observations do not record page number or position, so historical page attribution is inferred.

The queue work below extends this existing persistence rather than replacing it.

## Coverage model

### What the current watermark guarantees

`scrape_run_state.last_successful_scrape_at` establishes that the configured source work completed operationally at a point in time. It supports extending a later relative lookback after a failed or delayed workflow.

It protects against:

- A workflow not running.
- A required request failing and aborting the run.
- A temporary outage that delays the next successful run.
- Accidentally publishing after incomplete configured work.

### What the watermark does not guarantee

It does not establish:

- That LinkedIn exposed every matching job.
- That all results in the relative time window were paginated.
- That the last requested page exhausted the ranked result set.
- That offset pages formed a stable snapshot.
- That a job below the page cap will move above it later.
- That every discovered ID received a detail fetch.
- That every configured query has equal recall.

The guest search is ranked and unstable, while the watermark is time-based. A time watermark cannot serve as an inventory cursor without a stable source ordering and continuation token.

### Required separation of concerns

Keep two independent concepts:

1. **Operational discovery watermark**: every required search request completed and every resulting discovery was durably persisted/queued.
2. **Coverage state/debt**: one or more scope/window attempts ended while still saturated or had deferred adaptive work.

A discovery cycle may be operationally successful but right-censored at its search cap. It may also have retryable detail work after its discovery watermark advances. Publication policy can allow right-censoring explicitly while retaining durable coverage debt for a recovery sweep. A later ordinary success, especially one using a narrower relative window, must not erase that debt automatically.

## Why repeated runs may or may not catch up

A later run can discover an omitted job if:

- Ranking changes in its favor.
- Competing jobs age out of the relative window.
- Another overlapping query ranks it higher.
- A geography/query partition exposes it in a smaller result set.
- A deeper or broader recovery sweep reaches it.

It may never catch up if:

- New arrivals keep it below the fixed page cap.
- It is removed before reaching a visible rank.
- It ages out before ranking improves.
- The guest endpoint never exposes it.
- It matches no configured query or partition.

High volume can make catch-up less likely: new jobs can continually push older jobs deeper. Hourly execution alone is not a completeness mechanism.

## Pagination and overlap semantics

Offsets `0,10,20,...` do not deliberately overlap. Duplicate IDs occur because separate HTTP requests do not share a stable result snapshot. Ranking can move between requests.

The recovery returned 2,399 card positions but only 2,171 unique IDs within their queries, demonstrating this upstream overlap. The scraper must deduplicate source IDs but continue past a duplicate-only page because a later page can still contain new IDs.

Search-query and time-window overlap are intentional:

- Query overlap improves recall and records provenance.
- Time overlap protects against delayed indexing, failed schedules, and rank changes.

The optimization target is repeated processing and payload, not eliminating protective overlap.

## Target invariants

1. Every repository LinkedIn producer uses one durable source-wide limiter and circuit for search, detail, activity-check, retry, backfill, and recovery traffic. Until that exists, every LinkedIn workflow shares one non-cancelling GitHub concurrency group and no uncoordinated producer is scheduled.
2. Explicit denial or challenge opens the durable source circuit and stops all source work. The circuit stores reason, triggering response class, `opened_at`, `open_until`, and reset evidence; every producer checks it before acquiring a request grant.
3. Required search/parser/enqueue failure prevents discovery success; detail failure retains a retryable task and blocks the canonical cutoff.
4. Search discovery is persisted before optional detail processing.
5. Every discovered source ID has a durable terminal or pending state.
6. Canonical writes remain serialized.
7. Retry operations are idempotent.
8. A right-censored scope records durable coverage debt.
9. Recovery does not rotate identities, proxies, accounts, or IPs to bypass controls.
10. Page-depth changes are driven by measured incremental workflow-distinct yield, not raw card counts alone.
11. One truthful configured user agent is pinned for a cycle and all of its retries; identity, proxy, account, and IP rotation are outside this design.
12. Logical work budgets and physical HTTP-attempt safety budgets are separate, and every initial request and retry consumes the physical budget.

## Proposed data model

### 1. Normalize page completion and extend coverage entries

Retain the existing bounded `ingestion_runs.page_coverage` array and add fields when available:

```json
{
  "page": 6,
  "start": 50,
  "elements": 10,
  "cards": 10,
  "new_source_ids": 9,
  "new_workflow_source_ids": 4,
  "known_source_ids": 5,
  "result": "complete",
  "request_attempts": 1,
  "cooldown_ms": 0,
  "elapsed_ms": 3180
}
```

Do not store full cards or descriptions in this JSON.

The JSON array is a bounded summary, not the correctness authority. Add `linkedin_ingestion_pages`, keyed by `(ingestion_run_id, page_number)`, with the offset, request timestamp, effective absolute window, counts, result classification, attempts, cooldown, elapsed time, classifier version, and a canonical response/source-position fingerprint. Add page-source membership keyed by `(ingestion_run_id, page_number, provider, source_job_id)` so retries can prove the same page membership and positions rather than trusting matching counts.

Add `linkedin_discovery_cycle_sources`, uniquely keyed by `(discovery_cycle_id, provider, source_job_id)`, with the first ingestion run/page/position. The page RPC's newly inserted cycle-source rows authoritatively determine `new_workflow_source_ids`; after a crash the set is reconstructed from this table, not an in-memory-only set. One bounded transactional discovery RPC must commit the page checkpoint, page-source membership, cycle-source rows, observations, monotonic source-state reduction, exact task requirements, and task upserts together. A retry replays the same logical page key idempotently; a crash resumes at the first absent page. Reject a response fingerprint or source membership that conflicts with an already committed page instead of silently replacing it.

The manifest freezes one absolute source window for each scope attempt. Before every initial, retried, or resumed page request, recompute `f_TPR` from that page's request time so the effective relative window still contains the manifest's earliest instant plus the indexing safety margin. A page whose supported maximum lookback can no longer contain the manifest window is not sent: fail the scope and create/retain explicit interval debt. Sealing verifies that every committed page contains the manifest window, preventing one cycle from combining silently disjoint sliding windows.

Metric definitions are intentionally orthogonal: `new_source_ids` means not seen on an earlier page of this scope run, `new_workflow_source_ids` means not seen in an earlier scope/page of this cycle, and `known_source_ids` means mapped to a canonical job before the cycle.

Add nullable `page_number`, `page_start`, `position_on_page`, and `position_in_scope` columns to append-only `listing_observations`. Attach these values to cards before per-query deduplication and retain the first observed position in that query run. Persist observations and a `listing_states` row for every valid source ID even when `posted_at` is unavailable.

`new_workflow_source_ids` may use one in-memory set shared across scopes as a cache, but `linkedin_discovery_cycle_sources` is authoritative and recoverable. It cannot be derived from the current query-local `scraped_cards` list.

Keep operational and coverage state separate. Retain the existing operational status only long enough to migrate readers, then constrain it to `running`, `complete`, and `failed`. Add `coverage_status` constrained to `unknown`, `exhausted`, `right_censored`, and `failed`, plus a constrained failure/error code. Do not silently redefine the existing `coverage_complete` boolean, which currently remains false because the guest search cannot prove absence; backfill and migrate every reader deliberately before removing it.

### 2. Per-scope coverage state

Add `linkedin_scope_coverage_state` keyed by stable query scope:

```text
scope_key                     text primary key
scope_definition_hash         text not null unique
scope_definition              jsonb not null
config_revision               bigint
config_content_hash           text not null
archetype                     text not null
query_id                      text not null
geography_id                  text not null
last_operational_success_at   timestamptz
last_exhausted_at             timestamptz
last_saturated_at             timestamptz
last_deep_sweep_at            timestamptz
consecutive_saturated_runs    integer not null default 0
recommended_pages             integer not null default 6
coverage_debt                 boolean not null default false
coverage_debt_since           timestamptz
latest_tail_new_ids           integer not null default 0
latest_tail_workflow_new_ids  integer not null default 0
updated_at                    timestamptz not null
```

Define `scope_key` as the full collision-resistant digest of a versioned canonical JSON serialization containing every stable inventory-defining parameter: provider, endpoint version, canonical lane, Unicode-normalized and whitespace-normalized query text, query language/type, effective location and geo ID plus mapping version, job type, sorted work types, and any partition filters. Exclude relative lookback, page cap, query sort order, and configuration revision. Store the canonical JSON beside the digest and reject a digest collision. The current generated `query_id` includes `sort_order` and omits effective filters, so it is not a durable scope key.

The current configuration revision is nullable. Use the nullable bigint plus a required content hash until the authoritative configuration owner migrates revision to a required monotonic value; do not stringify a missing revision.

Definitions:

- **Exhausted**: positive terminal evidence was reached. A tested no-results response is terminal. A short ranked page alone is not sufficient; require a confirming tested terminal response at the next offset unless an authorized source contract explicitly guarantees short-page termination.
- **Right-censored**: the attempt ended without positive terminal evidence because its adaptive target, soft/hard cap, or global logical budget was reached. **Saturated** is the measurable subtype whose final configured tail remained full or workflow-productive; both productive and unproductive nonterminal tails retain debt, but only productive tails raise recommended depth.
- **Failed**: request, challenge, parser, or persistence failure.

Here, exhausted means only that this response sequence terminated. It does not prove that LinkedIn's full matching inventory was exposed.

A nonempty response with no cards is terminal only when it matches a versioned, tested LinkedIn no-results classifier. The classifier must check expected URL, status, content type, and exact positive selectors/text signatures against archived positive and negative fixtures. Empty HTML, login/checkpoint/challenge content, and unfamiliar zero-card HTML are parser failures. A zero-card first page remains incomplete unless the empty-state structure is positively identified; this avoids interpreting a soft block as an empty market.

### 3. Durable coverage debt

A boolean on scope state is only a summary. Add a stateful `linkedin_coverage_debt` obligation table plus an append-only attempt ledger because a later exhausted 7-hour query cannot resolve an earlier saturated 96-hour query:

```text
id                         bigint generated always as identity primary key
scope_key                  text not null references linkedin_scope_coverage_state(scope_key)
origin_discovery_cycle_id  bigint not null
origin_ingestion_run_id    uuid not null references ingestion_runs(id)
debt_kind                  text not null
posting_date_filter        text not null
lookback_hours             integer not null
source_window_earliest_at  timestamptz not null
source_window_latest_at    timestamptz not null
page_cap                    integer not null
tail_new_source_ids        integer not null
status                     text not null
created_at                 timestamptz not null
last_attempted_at          timestamptz
resolved_at                timestamptz
resolution                 text
resolved_by_ingestion_run_id uuid references ingestion_runs(id)
accepted_at                timestamptz
accepted_by                text
acceptance_reason          text
unique (scope_key, origin_ingestion_run_id, debt_kind,
        source_window_earliest_at, source_window_latest_at)
```

Debt kinds include `search_right_censored`, `lookback_truncated`, `search_failed`, and `scope_unattempted_after_cycle_failure`; one ingestion run may create more than one kind for different intervals. Statuses are `pending`, `resolved`, `expired_unresolved`, and `accepted_boundary`. Represent recovery execution in an append-only `linkedin_coverage_debt_attempts` table keyed by debt and recovery ingestion run; do not put an unfenced `in_progress` state on the debt itself. Each attempt records requested parameters, per-page effective windows, outcome, and timestamps. The obligation fingerprint makes each distinct debt retry-idempotent without collapsing two gaps from one run.

Add a composite foreign key from `(origin_discovery_cycle_id, scope_key, origin_ingestion_run_id)` to the matching cycle-manifest row; do not allow a debt to pair an ingestion run with another scope. The manifest therefore also has a unique constraint on that triple.

Maintain scope-state summaries and debt rows through one reducer RPC that locks the scope row and affected debt rows. Derive `coverage_debt` and `coverage_debt_since` in that transaction or through a view; clients cannot set the summary independently. Older attempts cannot overwrite recommendation, streak, or success state from newer attempts. Add named checks for status values, positive lookback/page caps, nonnegative yields, ordered windows, and status-dependent resolution/acceptance fields.

A recovery can resolve debt only when every requested recovery page through positive terminal evidence contained the debt interval, or when an auditable reviewed partition-replacement manifest did so. Because `f_TPR` is relative and slides while requests run, compute each page's relative window from that page's request timestamp plus a safety margin and persist its effective absolute window. If the original interval can no longer fit inside the maximum lookback, retain `expired_unresolved`; do not relabel it resolved. `accepted_boundary` requires reviewer, timestamp, and reason. This is the explicit record of a potentially permanent coverage gap.

### 4. Top-level discovery cycles and publication barrier

Add one top-level discovery cycle ID shared by all per-query `ingestion_runs` in a scheduled execution. The cycle records the configuration revision, requested windows, required scopes, search status, durable-queue cutoff, and canonicalization status.

```text
linkedin_discovery_cycles
id                         bigint generated always as identity primary key
started_at                 timestamptz not null
search_completed_at        timestamptz
canonical_applied_at       timestamptz
config_revision            bigint
config_content_hash        text not null
required_scope_count       integer not null
completed_scope_count      integer not null default 0
search_status              text not null
canonical_status           text not null
coverage_debt_count        integer not null default 0
discovery_sequence         bigint not null unique
created_at                 timestamptz not null
```

Counts are cached audit values, never proof of completeness. Add `linkedin_discovery_cycle_scopes` as an immutable manifest populated before source requests begin:

```text
discovery_cycle_id         bigint not null references linkedin_discovery_cycles(id)
scope_key                  text not null references linkedin_scope_coverage_state(scope_key)
ingestion_run_id           uuid not null unique references ingestion_runs(id)
required                   boolean not null
posting_date_filter        text not null
request_anchor_at          timestamptz not null
source_window_earliest_at  timestamptz not null
source_window_latest_at    timestamptz not null
minimum_pages              integer not null
target_pages               integer not null
status                     text not null
enqueue_committed_at       timestamptz
primary key (discovery_cycle_id, scope_key)
unique (discovery_cycle_id, scope_key, ingestion_run_id)
```

Add `discovery_cycle_id` as an indexed foreign key on `ingestion_runs`. Also add a durable exact-requirement relation:

```text
linkedin_discovery_requirements
discovery_cycle_id         bigint not null references linkedin_discovery_cycles(id)
ingestion_run_id           uuid not null references ingestion_runs(id)
provider                   text not null
source_job_id              text not null
task_kind                  text not null
requirement_key            text not null
task_id                    bigint not null
required                   boolean not null
created_at                 timestamptz not null
primary key (discovery_cycle_id, ingestion_run_id, provider, source_job_id, task_kind, requirement_key)
```

Create the task table before this requirement table, or add the `task_id` foreign key in a later `ALTER TABLE`; the migration must never reference an object that does not yet exist.

The manifest proves required-scope set equality and the requirement rows prove the queue cutoff. A completed metadata or relist task cannot satisfy an `initial_detail / first` requirement. The cycle identity is correlation only; identity allocation order is not execution order. Create the cycle, one pre-created ingestion run per scope, its full manifest, and a monotonic `discovery_sequence` in one transaction under a singleton/advisory lock before source work begins. That sequence orders both successful and failed cycles and is the deterministic publication cutoff.

Add one idempotent `seal_linkedin_discovery_cycle(p_cycle_id)` RPC. In one transaction it locks the cycle, verifies set equality between required manifest rows and successfully `complete` ingestion runs whose coverage status is `exhausted` or `right_censored`, rejects every failed/running/missing required scope, verifies every committed discovery has an exact durable task requirement or conflict-free canonical mapping, derives all counts, records database completion time, and advances operational/per-scope watermarks monotonically. Replace `record_scrape_success(p_finished_at)` and remove its direct-write fallback before queue cutover; callers cannot supply an authoritative completion timestamp.

Record failed cycles in the same ordered ledger; do not discard their sequence or committed requirements. Add an immutable `linkedin_discovery_cycle_resolutions` row keyed by failed cycle with a resolving recovery cycle and resolution type. A database RPC may create it only after every committed requirement from the failed cycle is satisfied and every failed/missing scope-window has corresponding debt that the recovery resolved, or a reviewed acceptance with reviewer/reason exists. A later cutoff cannot pass a failed cycle without this explicit supersession relation.

The current global scrape watermark can advance when every required search scope completes and every valid discovery is durably observed and queued. Publication must not use that discovery watermark as proof that queued records are canonical. Add `source_discovery_cycle_id` and `source_discovery_sequence` to publication state and generation rows, with foreign keys to a sealed cycle and uniqueness on a published sequence. Migrate `finalize_freehire_publication` to accept only a cycle ID, derive the authoritative watermark/sequence from locked database state, and reject unsealed, stale, out-of-order, or predecessor-incomplete cycles. Every earlier discovery sequence must be sealed and canonically applied or have a valid failed-cycle resolution whose recovery is itself eligible. Deploy this versioned RPC beside the current timestamp-based contract before cutting the gate over. A failed cycle need not block forever after a later explicit recovery, but the cutoff cannot pass unresolved required observations or scope gaps from that failed cycle. Finalize only when:

- Every exact required occurrence through the cutoff satisfies exactly one allowed case: its linked task is `complete` and `listing_states` has a conflict-free canonical mapping; its linked task is confirmed `terminal_unavailable`, for which no canonical mapping is required; or it has a separate immutable reviewed acceptance record. `pending`, `leased`, `failed_retryable`, and unreviewed `failed_terminal` remain blocking.
- The single canonical writer has committed all successful details through the cutoff.
- The existing Freehire/publication integrity predicates pass for the rows selected by the service-role-only publication view.

Later tasks may complete before older tasks, but the published cutoff cannot skip an unresolved required occurrence. Terminal-unavailable records satisfy the barrier while remaining visible in audit metrics.

This requires changing both `publication_gate.py` and `finalize_freehire_publication`; the current gate validates only the global scrape timestamp and current Freehire rows. Enforce the single canonical writer with a transaction-level PostgreSQL advisory lock shared by canonical mutation and publication, not only workflow concurrency. Both paths acquire it before any other database lock. The fixed order is: canonical/publication advisory lock, cycle/publication singleton row, task rows by ascending task ID where needed, then canonical job rows by ascending job ID.

When search succeeds but the canonical cutoff is not ready, the publication RPC returns a typed `outcome = published | unchanged | deferred` result with requested cycle, eligible cycle, unchanged nullable prior-generation metadata, and reason. A deferred result inserts or updates no publication rows and exits the gate successfully; integrity failures, invalid transitions, and source-search failures remain workflow failures. The initial deferred case may have no positive generation. Expose the outcome and cycle IDs as workflow outputs. This prevents a legitimate detail backlog from converting every later GitHub run into a false source outage while still preventing an incomplete generation switch.

This barrier guarantees canonical handling of discoveries, not publication of every canonical row. Preserve the current per-row Freehire policy: historical `pending` and `failed` classifications remain excluded without blocking already-current rows. Making publication complete by discovery cohort would require a separate product decision and a downstream-classification cutoff; do not bundle that semantic change into discovery recovery.

### 5. Durable discovery queue

Add `linkedin_discovery_tasks`:

```text
id                       bigint generated always as identity primary key
provider                 text not null default 'linkedin'
source_job_id            text not null
task_kind                text not null
requirement_key          text not null
first_ingestion_run_id   uuid not null references ingestion_runs(id)
first_query_scope        text not null
first_observed_at        timestamptz not null
latest_observed_at       timestamptz not null
posted_at                date
search_card              jsonb not null
status                   text not null
priority                 integer not null
attempt_count            integer not null default 0
available_at             timestamptz not null
leased_by                text
leased_at                timestamptz
lease_expires_at         timestamptz
lease_token              uuid
max_attempts             integer not null
last_error_code          text
completed_at             timestamptz
canonical_job_id         text references jobs(job_id)
unique (provider, source_job_id, task_kind, requirement_key)
```

Statuses:

```text
pending, leased, complete, terminal_unavailable, failed_retryable, failed_terminal
```

Add named checks for the status vocabulary, nonempty keys, nonnegative priority/attempts, positive `max_attempts`, ordered observation times, object-shaped `search_card`, complete lease-field presence only while leased, and completion fields only in terminal states. `canonical_job_id` on the task is cached audit data; `listing_states` remains authoritative. Conflicting non-null mappings fail reconciliation. Keep the task foreign key as a merge guard and update the canonical merge RPC to remap task and source-state references to the survivor in the same transaction before deleting the loser.

Claim tasks in one `FOR UPDATE SKIP LOCKED` data-modifying statement. The claim RPC accepts a constrained `oldest` or `newest` initial-detail mode so the drain coordinator can enforce the reserved freshness quota: oldest orders `(priority desc, first_observed_at, id)`, newest orders `(priority desc, first_observed_at desc, id)`. Generate a fresh `lease_token` and increment `attempt_count` when the lease is granted. Claim, heartbeat, retry, terminal, and completion RPCs compare task ID, `status = 'leased'`, worker ID, unexpired lease, and exact token. A stale worker changes zero rows and cannot mutate canonical or mapping state. Use bounded backoff with database-derived `available_at`; exhausted attempts become review-blocking `failed_terminal`, never silently accepted.

Freeze both claim modes before choosing indexes. Test claimable-status partial indexes matching oldest and newest order plus a separate expired-lease partial index on `(lease_expires_at, id)`. If future-dated retries dominate, compare an `available_at`-first candidate index with `EXPLAIN (ANALYZE, BUFFERS)` rather than assuming one index serves filtering and both orderings.

Use task kinds and requirement keys so later work can be represented without reopening ambiguous completed rows:

```text
initial_detail / first
metadata_enrichment / <metadata-schema-version>
relist_validation / <candidate-posted-date>
availability_revalidation / <prior-terminal-task-id>:<reobserved-cycle-id>
```

Add append-only `linkedin_discovery_task_attempts` keyed by task and attempt number. Store lease token, request/response class, HTTP status when available, parser/schema version, bounded confirmation/error evidence, and start/finish timestamps; do not store full descriptions in this audit table. Add immutable `linkedin_discovery_requirement_acceptances` keyed by the exact requirement, with reviewer, nonempty reason, and database timestamp. Only this acceptance row can make an unreviewed `failed_terminal` requirement non-blocking.

The queue decouples discovery from the production configuration's observed 50-detail budget; the tracked contract itself permits other database-configured values. A discovered job is not lost merely because one query exceeds the immediate detail budget. Evaluate the publication barrier through `linkedin_discovery_requirements`, not a coarse source-ID-only join. Repeated observations share the same global `initial_detail / first` task but retain distinct cycle/run requirement rows.

Define enqueue conflict reduction explicitly: preserve immutable first-observation fields, advance `latest_observed_at` with `GREATEST`, keep the card associated with the resulting latest observation, and raise priority monotonically unless a versioned policy says otherwise. Enqueue never moves `leased` or terminal tasks back to `pending`. A later observation of a 404-derived `terminal_unavailable` ID creates a new versioned validation requirement rather than treating the old terminal result as permanent.

Canonical application uses one owner-checked transaction/RPC. The worker keeps the validated result in memory while it holds/heartbeats the lease; durable staging is optional and must be designed separately if result payloads cannot fit this transaction boundary. While holding the canonical advisory lock, the RPC validates the live lease token, applies or idempotently repairs the canonical job mutation, updates `listing_states`, records memberships/content evidence, and transitions the exact task to `complete`. A crash before commit changes none of those objects; a retry after an ambiguous response may refetch but repairs the same canonical mapping without creating a duplicate.

### 6. Normalized source map through `listing_states`

Prefer the minimum-schema path: use the existing `listing_states` primary key and nullable `canonical_job_id` as the authoritative source map rather than creating another table with the same identity.

```text
listing_states
provider          text not null
source_job_id     text not null
canonical_job_id  text references jobs(job_id)
first_seen_at     timestamptz not null
last_seen_at      timestamptz not null
primary key (provider, source_job_id)
```

Backfill missing historical mappings from canonical IDs, latest IDs, and listing instances while canonical mutation is paused or holds the canonical advisory lock. Report and reconcile any source ID that maps to multiple canonical rows instead of choosing one silently. The existing `idx_listing_states_canonical_job_id` supplies the foreign-key lookup index. After a canonical write succeeds, update this map immediately; retries must repair a missing map, and the publication barrier must not pass it. Introduce a separate `listing_source_map` only if production validation finds lifecycle semantics that prevent `listing_states` from being authoritative.

Discovery state reduction belongs inside the transactional page RPC. Use `LEAST` for `first_seen_at`, `GREATEST` for `last_seen_at`, preserve non-null values unless newer evidence legitimately replaces them, and never replace a non-null canonical mapping with null. Conflicting non-null canonical IDs abort and enter reconciliation. Content persistence must upsert or repair a missing state row rather than assuming it exists.

### 7. Security and queue indexes

All new `public` tables are internal service tables:

- Enable RLS.
- Revoke table access from `PUBLIC`, `anon`, and `authenticated`.
- Grant only the required operations to `service_role`; do not use `GRANT ALL`.
- Revoke `EXECUTE` from `PUBLIC`, `anon`, and `authenticated` on queue/batch RPCs and grant it explicitly to `service_role`.
- Atomicity does not require `SECURITY DEFINER`. Prefer invoker functions when `service_role` has exact underlying privileges. If direct state mutation is intentionally prohibited, use narrowly scoped definer transition functions with schema-qualified objects, `search_path` fixed to `pg_catalog` or empty, service-role JWT validation, and explicit execute revocations before the service-role grant.
- Revoke sequence access from public roles. If direct identity inserts remain, grant only the specific sequence privileges needed by `service_role`; prefer mutation RPCs with direct DML revoked.

Add and test indexes for every referencing foreign key and barrier path, including cycle IDs on ingestion/scope/requirement/publication rows; ingestion run on pages, observations, scope manifest, debt, tasks, and requirements; task ID on requirements; canonical job ID on tasks; and `(ingestion_run_id, provider, source_job_id)` for the barrier join. PostgreSQL does not create indexes automatically on referencing foreign-key columns. Validate claim, lease-recovery, barrier, debt-selection, and merge-remap plans at production-like distributions.

## Normal-run lookback strategy

The production configuration observed during validation used a 48-hour window; the tracked contract allows other database values. A large repeated window can keep the first pages saturated and waste requests on known jobs.

The existing configuration cannot produce the shorter windows proposed here without explicit changes:

- `config.LINKEDIN_LOOKBACK_HOURS` has a 24-hour floor.
- `ScrapeSettings.lookback_days` has a one-day floor and production is set to two days.
- The workflow exports `LINKEDIN_LOOKBACK_HOURS=48` for scheduled runs because the dispatch-input fallback is also used on schedules.
- `_run_database_configured_linkedin` takes the maximum of the database and environment values and reads one global success watermark.

Implementation prerequisites:

1. Add validated `minimum_recent_window_hours`, `indexing_overlap_hours`, `maximum_normal_window_hours`, and `outage_recovery_cap_hours` options to the database configuration contract.
2. Resolve elapsed time from per-scope success state, not only the global watermark.
3. Separate the manual recovery override from normal scheduled configuration. Scheduled runs must not export a default 48-hour override.
4. Retain `lookback_days` only as a compatibility/recovery fallback until a later migration removes it.

After a measured canary, change normal lookback calculation to:

```text
normal_window = clamp(
    elapsed_since_scope_success + indexing_overlap,
    minimum_recent_window_hours,
    maximum_normal_window_hours
)
```

Initial candidate configuration:

```text
minimum_recent_window_hours = 3
indexing_overlap_hours = 6
maximum_normal_window_hours = 24
outage_recovery_cap_hours = 168
```

Expected behavior:

- Hourly daytime run: approximately 7 hours, pending measurement.
- Nine-hour overnight gap: approximately 15 hours.
- Delayed workflow within the cap: elapsed duration plus overlap.
- Explicit recovery: 48, 96, or 168 hours.

These values are hypotheses and require canary data. Compute integer seconds as `ceil(max(0, request_anchor_at - last_operational_success_at)) + indexing_overlap_seconds`, then clamp in seconds. Use a database timestamp captured before requests as the anchor. A missing/invalid/future scope timestamp uses the compatibility/recovery fallback and creates a diagnostic event; it never produces a negative interval.

When elapsed plus overlap exceeds `maximum_normal_window_hours`, create interval debt for the truncated portion in the same transaction that seals the scope and schedule the smallest recovery window that contains it, bounded by `outage_recovery_cap_hours`. If elapsed exceeds that recovery cap, record the unreachable portion as `expired_unresolved`; never silently advance past an unrepresented gap. A shorter relative window is supported by `f_TPR=r<seconds>` and reduces competition for recently posted jobs. The guest endpoint does not provide a verified arbitrary historical start/end interval, so the design must not depend on disjoint date slicing.

Before reducing the normal 48-hour baseline, measure delayed-indexing behavior. Keep periodic recovery sweeps until evidence shows the shorter window is safe.

## Adaptive search algorithm

### Configuration

```text
min_pages_per_query = 6
soft_max_pages_per_query = 10
hard_max_pages_per_query = 20
deep_sweep_max_pages = 20
deep_sweep_lookback_hours = 96
max_adaptive_extra_requests = bounded logical-page budget
max_source_http_attempts_per_run = absolute physical-attempt budget
```

Initial deployment must keep six as the minimum allocation cap, except when positive terminal evidence ends a scope earlier. Enforce `1 <= min_pages_per_query <= soft_max_pages_per_query <= hard_max_pages_per_query` and `deep_sweep_max_pages <= hard_max_pages_per_query`. Normal runs never exceed the soft maximum; pages above it are recovery-only. Map the legacy exact `max_pages_per_query` deliberately during migration and reject ambiguous/conflicting settings. Higher values require source-policy review and canary approval.

Define the logical normal search allocation as `required_scope_count * min_pages + max_adaptive_extra_requests`. Keep `max_adaptive_extra_requests=0` in shadow mode, use no more than 20 extra logical pages in the first production canary, and raise it only after reviewing runtime and denial/challenge metrics. Independently cap physical HTTP attempts; every initial request and retry consumes that cap. If retry consumption prevents required logical pages from completing, fail the cycle rather than starving later scopes. Define a separate physical/logical detail budget for queue drain; replace the per-query cap only after measuring equivalent source pressure.

### Pseudocode

```text
build the immutable required-scope manifest

for page_number in 1..min_pages:
    for each non-terminal required scope in rotating deterministic order:
        acquire global limiter
        consume one physical HTTP attempt per request/retry
        response = fetch(scope, page_number)

        if explicit denial/challenge:
            open durable source circuit; stop entire source run
        if request failure, physical-attempt exhaustion, or unrecognized response:
            mark failed; stop entire source run
        if positively classified no-results response:
            mark exhausted; resolve only comparable scope/window debt; break

        transactionally persist page, cards, source states, requirements, and tasks

        if short page:
            mark terminal confirmation required
            request the next offset on this scope's next turn
            only positive terminal evidence marks exhausted

    if min_pages reached without positive terminal evidence:
        extension_target = min(
            soft_max_pages,
            max(min_pages + 1, scope_state.recommended_pages)
        )
        add scope to adaptive extension queue

while logical extension budget remains and eligible scopes exist:
    order a round by oldest debt, then descending tail workflow-distinct yield,
    then a rotating cycle cursor, then scope_key
    for each eligible scope, request at most one next page in this round:
        apply the same limiter, attempt-budget, classifier, and page transaction
        if positive terminal evidence:
            mark exhausted; resolve only comparable scope/window debt; remove scope
            decay recommended_pages slowly toward min_pages
        elif extension_target reached:
            mark right-censored; idempotently create debt; remove scope
            increase recommended_pages by one only when the measured tail is productive,
            bounded by soft_max_pages
        else:
            keep scope eligible for the next round

for any eligible scope deferred by the global budget:
    mark right-censored; create/retain coverage debt
```

Define `short` as fewer than the expected ten valid parsed cards; it is evidence requiring confirmation, not termination by itself. Define tail yield as the sum of `new_workflow_source_ids` over the last two committed pages, ordered descending after debt age; persist the allocation inputs and decision. New scopes use zero prior yield. Do not stop because one or more pages contain only IDs already seen in the query. The current scraper deliberately continues duplicate-only pages, and unstable offset pagination allows a later page to contain new IDs. Duplicate-only yield lowers extension priority but cannot establish exhaustion.

The minimum six-page pass is required work. Adaptive extensions that are not started because the global budget is exhausted become explicit debt. Once an extension request starts, request/parser/persistence failure fails the operational source run rather than silently downgrading that request to optional work.

### Request-budget fairness

If the global search budget cannot serve every saturated scope to its recommended depth:

1. Always provide the minimum page allocation to all enabled scopes before adaptive extension requests.
2. Allocate extra pages round-robin by coverage-debt age and tail yield.
3. Never let one broad query consume the entire source budget.
4. Persist deferred extra-page work rather than treating it as exhausted.

## Detail processing algorithm

Search completion must not depend on immediate detail capacity.

1. Persist every valid card observation, page position, and required task in one bounded transactional discovery RPC.
2. Upsert every newly discovered or enrichment-required ID into `linkedin_discovery_tasks`; fail the source run if durable enqueue fails.
3. Complete the search phase for all scopes.
4. Drain detail tasks by priority under the same request limiter:
   - Newly discovered IDs first.
   - Incomplete metadata second.
   - Conservative relist refreshes third.
5. Pass validated detail results to the one canonical writer under the atomic task-application contract while the fenced lease remains live. Do not add staging unless payload size/runtime evidence requires a separately specified durable handoff.
6. Apply the versioned outcome matrix below; every retry reacquires the source-wide limiter and physical-attempt grant.
7. Treat 410 as terminal unavailable. Treat a first 404 as retryable after a bounded delay; mark it terminal unavailable only after configured confirmation, and create a new validation requirement if a later card re-observes the ID.
8. Leave unprocessed tasks pending for the next run.

The implementation must freeze an outcome matrix before queue cutover:

| Outcome | Task transition | Circuit/run effect |
|---|---|---|
| Network error, 408, 429, transient 5xx | `failed_retryable` with bounded exponential backoff and honored `Retry-After` | Continue queue unless physical budget is exhausted |
| 403, 999, checkpoint, or challenge | `failed_retryable` without consuming acceptance | Open source circuit and stop every producer |
| Confirmed 404 or 410 | `terminal_unavailable` with response class and confirmation evidence | Satisfies the exact requirement; continue |
| Unexpected 4xx | `failed_retryable`, then review-blocking `failed_terminal` at max attempts | Continue unless policy opens the circuit |
| Malformed/unknown 2xx or missing task-required fields | `failed_retryable` with parser/schema version, then review-blocking `failed_terminal` | Continue; never mark complete |
| Validated detail plus atomic canonical apply | `complete` with authoritative source mapping | Continue |

Define required fields separately for initial detail, metadata enrichment, and relist validation, including the `fetch_descriptions=false` mode. Lease loss during or after a request cannot commit a transition; the replacement worker owns the task.

Suggested priorities:

```text
100: initial detail, including recovery discoveries
60: incomplete required metadata
30: relist refresh
```

Order initial-detail tasks oldest-first to prevent starvation. Reserve at least one slot and 20% of the remaining initial-detail budget for newest-first work, with deterministic ID tie-breakers; unused quota spills to oldest-first work. Enrichment and relist work cannot consume the reserved initial-detail quota. Revisit the percentage only through measured configuration changes.

Finalize the discovery cycle and advance its operational watermark transactionally after all required search scopes finish and all discoveries are durably queued, before draining details. It must not require the entire detail backlog to drain in the same workflow. A detail failure keeps its task retryable and blocks the canonical cutoff without erasing completed discovery. Explicit denial/challenge still opens the source circuit and stops all remaining requests. Publication readiness is separate and must require canonical processing through the cycle cutoff described above.

## Recovery and deep sweeps

### Trigger conditions

Schedule or enqueue a deep sweep when:

- A normal scope reaches its adaptive target or soft cap with a full/productive final page.
- `consecutive_saturated_runs` exceeds a threshold.
- Coverage debt exceeds an age target.
- A debt window is approaching the point where the maximum relative lookback can no longer contain it.
- A scheduled run was missed or failed.
- Query configuration changed materially.

### Strategy

- Run off-peak under the same global limiter.
- Select the smallest allowed lookback that still contains the debt's absolute inferred window, up to `outage_recovery_cap_hours`, and use a bounded deeper page cap that never exceeds the absolute hard maximum. Recompute the relative lookback at each page request so its effective absolute window still contains the debt interval.
- Prioritize scopes with the oldest debt and highest tail yield.
- Do not assume a wider window improves recall; it can increase ranking competition. Compare deep-page yield against narrower normal runs.
- Mark debt `expired_unresolved` when no supported relative window can contain its original interval; this is not a successful recovery.
- Stop on explicit source denial or challenge.
- Keep recovery idempotent through source IDs, observations, and discovery-task keys.

### Debt clearing

Clear coverage debt only when one of these is true:

- A comparable or broader recovery attempt reaches positive terminal evidence while every page through that evidence contains the debt interval.
- A reviewed, versioned parent-to-partition replacement manifest proves every child covered the debt interval and reached positive terminal evidence.
- A product decision accepts a documented right-censored boundary.

An ordinary capped success must not clear prior debt.

## Scope partitioning

Persistent saturation should first be addressed by legitimate search partitions rather than unbounded depth.

Candidate partitions:

- Province or approved metro geography.
- Remote versus on-site/hybrid work type.
- Precision title families within a broad recall query.
- Language where the lane already supports multiple languages.
- Distinct role families that currently share one broad Boolean query.

Partition requirements:

1. Each partition has a stable scope key and independent coverage state.
2. The union preserves the original product intent.
3. Cross-partition source IDs are deduplicated before detail work.
4. Partitioning is not used to evade denial, challenge, rate, account, or IP controls.
5. New partitions run in shadow mode before replacing the original scope.
6. Parent-to-partition replacement records exact child scope definitions, covered windows, reviewer, review time, and product-intent rationale; debt cannot clear from an implicit current configuration.

## Supabase efficiency work

### Phase A: authoritative `listing_states` map and compact canonical lookup

The observed production sample loaded about 49.3 MB of logical JSON for 5,840 LinkedIn jobs; a compact projection was about 2.39 MB. Link these values to the Phase 0 evidence artifact before using them as acceptance baselines.

Implementation:

1. Backfill `listing_states.canonical_job_id` from canonical ID, latest ID, and listing instances; report and reconcile any source ID mapped to multiple canonical rows instead of choosing one silently.
2. Maintain `listing_states` mappings immediately after canonical inserts/reposts/merges, with idempotent repair on retry and publication blocked until the mapping exists.
3. Resolve exact source IDs with targeted indexed queries.
4. Load one compact all-canonical projection per run, build the same normalized-company buckets in memory, and preserve the current greater-than-200 rejection rule.
5. Resolve equal body fingerprints from compact fields; fetch full descriptions only for same-role/location candidates that require fuzzy body comparison.
6. Fetch the full row only for the chosen match before building its update/CAS payload.
7. Shadow old and new canonical decisions before cutover.

Acceptance:

- 100% source-ID resolution equivalence.
- 100% canonical-decision equivalence or reviewed exceptions.
- At least 80% reduction in logical candidate-read bytes.
- No increase in duplicate canonical rows.

### Phase B: batched membership RPC

Current code issues one `record_job_archetype_membership` call per observed canonical/query relationship, not six archetypes per new job. The observed six-page production sample produced 2,121 calls; link that count to the Phase 0 evidence artifact.

Before adding the plural RPC, pin and document the authoritative `zeroluck/job-scraper-web` migration revision for the existing membership table and singular RPC, which are consumed here but not created under `supabase_setup/`. The cross-repository clean database baseline must support the singular path before plural-path equivalence can be tested.

Implementation:

1. Add `record_job_archetype_memberships(p_records jsonb)`.
2. Batch per query in bounded chunks.
3. Preserve service-role checks, provenance union/deduplication, stable ordering, and first/last timestamps.
4. Reject an invalid batch atomically and return only a processed count; split/retry in the caller rather than introducing partial-commit semantics.

Acceptance:

- Final membership rows and `matched_queries` equal the existing path.
- Retry idempotency.
- At least 95% fewer membership HTTP calls.

### Phase C: batched content versions

Current detail persistence performs a read-before-write for each `(provider, source_job_id, content_hash)` and may patch listing state separately.

Implementation:

1. Add a 50-100-record transactional array RPC.
2. Preserve first/last timestamps and canonical linkage.
3. Add `listing_content_version_observations`, keyed by `(provider, source_job_id, content_hash, ingestion_run_id)`. Insert with `ON CONFLICT DO NOTHING` and increment the parent count only when that insert returns a row in the same transaction, or derive the count from the ledger. `last_ingestion_run_id` is audit metadata, not an idempotency key.
4. Lock/update records in deterministic provider/source/hash order.
5. Update related listing state in the same transaction where valid.

Acceptance:

- Identical content-version keys and descriptions.
- No duplicate observation increment after retries.
- At least 80% fewer version/state calls.

### Phase D: Freehire CAS token

Replace repeated full source snapshots in claim and persist RPCs with a database-generated, versioned token covering every classification input.

Acceptance:

- A source mutation between claim and persist rejects the stale token.
- Classification results and retry behavior remain equivalent.
- Material reduction in claim/persist request bytes.

### Phase E: run-scoped listing-state cache

Reuse state reads for source IDs repeated across queries while continuing to write immutable observations for provenance. Update the cache only after successful writes. This remains safe only with one listing writer or explicit row-version CAS.

### Not planned: custom compression transport

The existing Supabase/PostgREST client already negotiates gzip and production responses were observed with `Content-Encoding: gzip`. Do not add a custom HTTP transport merely to set `Accept-Encoding`; reduce selected fields and request count instead. Supabase billing documentation does not establish whether Database Egress is metered before or after compression, so report logical payload and observed wire bytes separately.

## Migration and verification discipline

This repository uses hand-authored SQL under `supabase_setup/`, not a declarative `supabase/schemas/` layout, but it does not own every consumed database object. Freeze this ownership and deployment order before Phase 1:

| Object | Authoritative owner and order |
|---|---|
| `scrape_settings`, configuration revision, `get_scraper_configuration()`, existing membership table, and singular membership RPC | Companion `zeroluck/job-scraper-web` migrations; pin the exact authoritative migration revision, deploy adaptive options there first, and preserve singular behavior during cutover |
| LinkedIn source-policy row, grant ledger, durable circuit, and grant/consume/open RPCs | This repository under `supabase_setup/`; deploy and migrate every producer before task/cycle rollout |
| Discovery tasks/attempts, then cycles/scopes/pages/sources/requirements/debt and seal/queue RPCs | This repository under `supabase_setup/`; create task targets before requirement foreign keys |
| Publication state/generations/finalizer v2 | This repository, after cycle schema and before Python gate cutover |

For every database phase:

1. Inspect the current remote/local tables before authoring DDL.
2. Add a focused migration under `supabase_setup/`, keep applied migrations immutable, and align `supabase_setup/init.sql` as a clean-install snapshot plus `supabase_setup/cleanup.sql` as its teardown counterpart.
3. Use bounded lock/statement timeouts and avoid combining a potentially slow backfill with table creation when separate deployment is safer.
4. Test locally or on a development branch before production.
5. Run schema tests, retry/idempotency tests, and Supabase security/performance advisors after DDL.
6. Verify RLS with actual `anon`, `authenticated`, and `service_role` roles; inspect table, sequence, and routine ACLs; and verify foreign-key indexes, claim/barrier query plans, and rollback behavior explicitly.
7. Add columns nullable first, backfill in bounded batches, add scan-heavy checks/FKs as `NOT VALID`, validate later, then enforce `NOT NULL`. Put `CREATE INDEX CONCURRENTLY` in a migration that is not transaction-wrapped and clean up invalid indexes after a failed build.
8. Test both clean bootstrap and ordered upgrade against the same expected schema, including actual SQL compilation and RPC execution rather than static SQL-string assertions alone.

Supabase is changing Data API auto-exposure defaults in 2026. Do not depend on those defaults; use explicit revocations and service-role grants for these internal tables and RPCs.

## Rollout phases

### Phase 0: observe the restored six-page baseline

Duration: at least 14 days covering all schedule hours.

Collect:

- Tail fullness and incremental source IDs by scope/page.
- Workflow-distinct incremental IDs.
- Delayed discovery age.
- IDs selected and deferred by the current 50-detail budget.
- 429/403/challenge/5xx rates.
- Search/detail request counts and duration.

First serialize every LinkedIn-producing workflow under one non-cancelling concurrency group, pin one configured user agent per cycle, and inventory every direct LinkedIn caller. Add only bounded telemetry needed for these measurements; make no automatic depth or lookback change. Check in the sanitized evidence artifact and migration ownership matrix before Phase 1.

Before Phase 1, replace workflow-only serialization with the durable source-wide limiter/circuit. Permit at most one pending grant per source. An atomic acquire RPC locks the source-policy row and issues a short-lived grant only when database time is at or after `next_allowed_at`; otherwise it returns the next eligible time without reserving a future slot. The producer waits and retries acquisition. Immediately before one physical attempt, a consume RPC locks the same row, rejects expired grants or a changed/open circuit, marks the grant consumed, records `started_at = database_now`, and advances `next_allowed_at` from that actual start by the configured interval plus bounded jitter. An unused/rejected grant expires and is never reused.

A denial/challenge increments the circuit generation and opens the same row transactionally, invalidating every pending grant. A consumed grant is considered in flight; because a process can pause between database commit and socket send, the protocol cannot retract it, so the worker must send immediately and may not perform other work in that boundary. No later grant can be consumed while the circuit is open. Verify all known callers fail closed when either RPC is absent and test process-delay behavior against the documented in-flight boundary. Adaptive pages and queue workers remain disabled until this cross-process contract is active.

### Phase 1: schema foundation and shadow queue

- Add cycle, page-position, scope-state, coverage-debt, and task schema through reviewed migrations; backfill the existing `listing_states` source map.
- Enable RLS, explicit service-role grants, claim indexes, and foreign-key indexes in the same migrations.
- Add idempotent transactional observation/enqueue.
- Continue immediate detail processing initially, but make shadow tasks non-claimable and reconcile each immediate outcome to the corresponding shadow task state/mapping. Do not allow queue drain until metrics prove zero tasks would refetch an already-applied detail.
- Verify every eligible source ID, including cards without `posted_at`, has the expected task or existing canonical source mapping.

### Phase 2: queue drain and publication barrier

- Switch initial detail processing to durable queue drain with one canonical writer.
- Enable fenced lease expiry/recovery and task-kind requirement keys.
- Cut canonical task application over to the advisory-locked atomic RPC before enabling workers.
- Shadow the canonical-applied cycle cutoff against current publication.
- Migrate publication finalization only after cutoff equivalence and crash-recovery tests pass.

### Phase 3: database hot-path reductions

- Cut over the authoritative `listing_states`/compact canonical lookup after decision equivalence.
- Batch membership writes before adding more search pages.
- Batch content-version/state writes.
- Measure request count and logical/wire payload after each independent cutover.

### Phase 4: adaptive depth in shadow mode

- Compute recommended depth and saturation without changing requests.
- Compare recommendations against observed six-page results.
- Enable extra pages for a small set of saturated scopes with a conservative hard cap.

### Phase 5: shorter normal lookback canary

- Implement the per-scope options and remove the scheduled 48-hour environment override first.
- Compare current 48-hour scopes with elapsed-plus-overlap scopes at matched schedule hours.
- Measure first-observation age and standardized recovery-only discoveries over subsequent sweeps.
- Do not cut over if delayed discoveries increase beyond the agreed threshold.

### Phase 6: scope partitioning

- Shadow candidate partitions for the most saturated scopes.
- Compare union yield, relevance, overlap, and request cost.
- Replace broad scopes only after product review.

### Phase 7: remaining payload reductions

- Freehire CAS token.
- Listing-state cache.

### Phase 8: parallel fetch workers only if needed

Use `stage2-parallelworkersplan.md` only if the optimized single worker cannot meet the runtime SLO. Parallel workers still share a source-wide limiter and feed one canonical writer.

## Tests

### Unit tests

- Full productive tail marks a scope saturated.
- A short page requests terminal confirmation; only positive terminal evidence marks exhausted.
- Unknown nonempty zero-card HTML fails instead of masquerading as no results.
- A positively identified no-results structure terminates the scope.
- Nonempty unparsable page fails.
- Any number of duplicate-only pages continues to the configured/adaptive bound.
- Adaptive depth respects minimum, soft maximum, and hard maximum.
- Search budget allocation is fair and deterministic.
- A narrow terminal attempt does not clear broader unresolved coverage debt.
- Stable scope identity does not change with lookback, page cap, sort order, formatting-only query edits, or config revision, but does change with effective job/work/geography/partition filters.
- Every eligible source ID is enqueued once per task requirement, including cards without `posted_at`.
- Page and rank fields survive query-local deduplication.
- Detail retry and fenced lease recovery are idempotent; a stale token cannot heartbeat, retry, complete, or write canonical state.
- A first 404 retries, a confirmed 404/410 becomes terminal unavailable, and a later observation creates a new validation requirement.
- Lookback truncation creates interval debt, including the portion already beyond the recovery cap.
- Duplicate scope completion cannot compensate for a missing required scope when sealing a cycle.
- Logical page allocation and physical HTTP-attempt caps account for every retry independently.
- Opening the source circuit invalidates every pending grant; no unconsumed request sends after the circuit generation changes.
- Stale Freehire CAS token is rejected.

### Integration tests

- Mock unstable pagination with duplicates moving between offsets.
- Simulate more than 50 new IDs in one query and verify all are queued.
- Retry an entire query and verify observations/tasks are not duplicated.
- Inject failures within the page RPC and verify page, observation, state, requirement, and task writes are all-or-nothing.
- Resume a stale running cycle at the first absent page checkpoint.
- Compare old and compact canonical match decisions.
- Retry membership and content-version batches; verify content observation sequences A/B/A count two distinct ingestion runs, concurrent duplicate A counts once, and rollback changes neither ledger nor parent count.
- Verify a failed scope prevents operational watermark advancement.
- Verify saturated success advances only the operational watermark and retains debt.
- Verify a retryable detail failure preserves the completed discovery watermark but blocks the canonical/publication cutoff.
- Verify publication cannot pass an unresolved exact requirement, cannot satisfy initial detail with an unrelated completed task, and can pass terminal-unavailable or immutable reviewed acceptance.
- Verify deferred publication changes no publication row, supports no prior generation, and reports a typed outcome.
- Create and finish two cycles out of order; verify monotonic discovery sequence/watermarks and idempotent resealing.
- Verify a later publication cutoff cannot pass a failed-cycle sequence until exact committed requirements and every missing scope-window are recovered or explicitly accepted.
- Use multiple real database sessions to claim disjoint task sets, expire/reassign a lease, reject the stale token, and allow exactly one concurrent completion.
- Race canonical completion against publication repeatedly under the shared advisory lock; verify no deadlock or premature generation.
- Terminate the queue worker after claim and around canonical application; verify lease recovery and mapping repair without duplicate canonical writes.
- Verify no-date cards create observations, source states, tasks, canonical mappings, and publication-eligible requirements.
- Verify canonical merges remap completed task/source-state foreign keys before deleting the losing row.
- Run clean-bootstrap and ordered-upgrade schema-equivalence tests, role-level RLS/ACL tests, actual RPC calls, and production-scale `EXPLAIN (ANALYZE, BUFFERS)` checks.

### Production canaries

- One or two scopes per lane before global rollout.
- No increase in denial/challenge rate.
- No violation of source-wide pacing or durable circuit behavior across concurrent producers.
- No canonical mismatch or duplicate increase.
- Recovery sweeps show decreasing aged coverage debt.

Run focused repository verification with:

```bash
PYTHONPATH=. python -m pytest tests
python -m py_compile scraper.py supabase_utils.py scrape_configuration.py publication_gate.py
git diff --check
```

Do not use unscoped repository-root test collection because vendored `docs/autthrottle-lab/Scrapling/tests` has unrelated optional dependencies.

## Metrics and SLOs

Track:

```text
source_http_attempts_total by endpoint/outcome
source_retries_total by endpoint/outcome
source_retry_after_seconds
source_backoff_seconds
source_wait_seconds
source_circuit_open_total
source_circuit_open_seconds
logical_search_pages_allocated
logical_search_pages_deferred
physical_http_attempt_budget_remaining
scope_saturated_total
scope_exhausted_total
coverage_debt_age_hours
expired_unresolved_coverage_debt_total
tail_new_source_ids
tail_new_workflow_source_ids
first_observation_age_days
recovery_only_discovery_rate
discovery_queue_depth
oldest_discovery_task_age_hours
task_lease_expired_total
task_lease_reclaimed_total
stale_cycle_total
discovery_to_detail_latency
detail_to_canonical_latency
canonical_match_disagreement_total
membership_calls_per_observation
database_logical_bytes_loaded
database_wire_bytes_observed
publication_cycle_lag
unresolved_required_occurrences
allocation_decisions by cycle/scope/reason
```

Keep metric dimensions bounded to endpoint, response class, lane, geography class, and query kind; put high-cardinality cycle/scope IDs in structured audit records. Request totals include retries. Define `recovery_only_discovery_rate` as workflow-distinct IDs first observed in a standardized recovery sweep and absent from matched preceding normal cycles, divided by workflow-distinct IDs in that sweep. Report missing-date IDs separately and require a predeclared minimum sample size/confidence method before a non-inferiority decision.

Initial acceptance targets:

- Zero lost discovered IDs between search and durable queue.
- p95 discovery-to-canonical latency below one scheduled interval for recent jobs, excluding source failures.
- No coverage debt older than 24 hours without a recorded recovery attempt.
- Zero challenge/denial responses in a canary; any occurrence stops the canary.
- At least 95% reduction in membership calls after batching.
- At least 80% reduction in canonical candidate-read bytes.
- For lookback changes, the standardized recovery-only discovery rate must remain within a non-inferiority margin of `max(0.5 percentage point, 10% of the Phase 0 baseline rate)` over matched scopes and schedule hours.

## Rollback criteria

Rollback an adaptive or partition change if any occurs:

- Increased 403/429/challenge frequency.
- Source work exceeds workflow/runtime SLO.
- Any canonical-decision mismatch outside an approved exception.
- Duplicate canonical rows increase.
- Discovery tasks are lost or remain unleased past the SLO.
- Delayed first-observation rate materially worsens.
- Coverage debt grows for seven consecutive successful schedules.

Rollback actions:

1. Return to the reviewed fixed six-page baseline and configured recovery-safe lookback.
2. Keep the discovery queue and observations; do not delete evidence.
3. Disable extra-page/deep-sweep scheduling.
4. Drain already discovered tasks under the normal limiter.
5. Preserve one canonical writer.

## Definition of success

The system succeeds when it can state, for each configured scope and run:

- What relative time window was requested.
- How many pages were attempted and completed.
- Whether the result set was exhausted, right-censored, or failed.
- How many distinct IDs each page added.
- Whether every discovered ID was durably queued and resolved.
- What unresolved coverage debt remains.

It must not state that every LinkedIn job was captured. Exhaustive coverage requires an authorized source with stable inventory pagination or a contractual feed; the guest ranked endpoint cannot provide that proof.
