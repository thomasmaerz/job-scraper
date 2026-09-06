BEGIN;

SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '10min';
SELECT pg_catalog.set_config('search_path', '', true);

DO $prerequisite$
DECLARE
    membership_table regclass := pg_catalog.to_regclass('public.job_archetype_memberships');
    jobs_table regclass := pg_catalog.to_regclass('public.jobs');
BEGIN
    IF pg_catalog.current_setting('job_scraper.install_mode', true) = 'base_init' THEN
        RETURN;
    END IF;
    IF membership_table IS NULL THEN
        RAISE EXCEPTION 'adaptive LinkedIn discovery requires externally owned public.job_archetype_memberships'
            USING ERRCODE = '55000';
    END IF;
    IF jobs_table IS NULL THEN
        RAISE EXCEPTION 'adaptive LinkedIn discovery requires public.jobs'
            USING ERRCODE = '55000';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM (VALUES
            ('job_id', 'text'::regtype, true),
            ('archetype', 'text'::regtype, true),
            ('matched_queries', 'jsonb'::regtype, true),
            ('first_matched_at', 'timestamptz'::regtype, true),
            ('last_matched_at', 'timestamptz'::regtype, true),
            ('filter_status', 'text'::regtype, true),
            ('is_filtered', 'boolean'::regtype, true),
            ('filter_reason', 'text'::regtype, false),
            ('insights', 'jsonb'::regtype, true),
            ('updated_at', 'timestamptz'::regtype, true)
        ) AS required(column_name, type_oid, is_not_null)
        LEFT JOIN pg_catalog.pg_attribute attribute
          ON attribute.attrelid = membership_table
         AND attribute.attname = required.column_name
         AND attribute.attnum > 0
         AND NOT attribute.attisdropped
        WHERE attribute.attname IS NULL
           OR attribute.atttypid <> required.type_oid
           OR (required.is_not_null AND NOT attribute.attnotnull)
           OR (NOT required.is_not_null AND attribute.attnotnull)
    ) THEN
        RAISE EXCEPTION 'public.job_archetype_memberships has incompatible required columns'
            USING ERRCODE = '55000';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint constraint_row
        WHERE constraint_row.conrelid = membership_table
          AND constraint_row.contype = 'p'
          AND ARRAY(
              SELECT attribute.attname::text
              FROM pg_catalog.unnest(constraint_row.conkey) WITH ORDINALITY key_column(attnum, position)
              JOIN pg_catalog.pg_attribute attribute
                ON attribute.attrelid = membership_table
               AND attribute.attnum = key_column.attnum
              ORDER BY key_column.position
          ) = ARRAY['job_id', 'archetype']
    ) THEN
        RAISE EXCEPTION 'public.job_archetype_memberships requires primary key (job_id, archetype)'
            USING ERRCODE = '55000';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint constraint_row
        WHERE constraint_row.conrelid = membership_table
          AND constraint_row.contype = 'f'
          AND constraint_row.confrelid = jobs_table
          AND constraint_row.confdeltype = 'c'
          AND ARRAY(
              SELECT attribute.attname::text
              FROM pg_catalog.unnest(constraint_row.conkey) WITH ORDINALITY key_column(attnum, position)
              JOIN pg_catalog.pg_attribute attribute
                ON attribute.attrelid = membership_table
               AND attribute.attnum = key_column.attnum
              ORDER BY key_column.position
          ) = ARRAY['job_id']
          AND ARRAY(
              SELECT attribute.attname::text
              FROM pg_catalog.unnest(constraint_row.confkey) WITH ORDINALITY key_column(attnum, position)
              JOIN pg_catalog.pg_attribute attribute
                ON attribute.attrelid = jobs_table
               AND attribute.attnum = key_column.attnum
              ORDER BY key_column.position
          ) = ARRAY['job_id']
    ) THEN
        RAISE EXCEPTION 'public.job_archetype_memberships requires cascading jobs(job_id) foreign key'
            USING ERRCODE = '55000';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint constraint_row
        WHERE constraint_row.conrelid = membership_table
          AND constraint_row.contype = 'c'
           AND pg_catalog.replace(pg_catalog.replace(
               pg_catalog.lower(pg_catalog.pg_get_constraintdef(constraint_row.oid)), '"', ''
           ), ' ', '') LIKE ANY (ARRAY[
               '%jsonb_typeof(matched_queries)=''array''%',
               '%is_jsonb_object_array(matched_queries)%'
           ])
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint constraint_row
        WHERE constraint_row.conrelid = membership_table
          AND constraint_row.contype = 'c'
           AND pg_catalog.replace(pg_catalog.replace(
               pg_catalog.lower(pg_catalog.pg_get_constraintdef(constraint_row.oid)), '"', ''
           ), ' ', '') LIKE '%jsonb_typeof(insights)=''object''%'
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_constraint constraint_row
        WHERE constraint_row.conrelid = membership_table
          AND constraint_row.contype = 'c'
          AND pg_catalog.replace(pg_catalog.replace(
              pg_catalog.lower(pg_catalog.pg_get_constraintdef(constraint_row.oid)), '"', ''
          ), ' ', '') LIKE '%filter_status=any(array[%'
          AND ARRAY(
              SELECT captures[1]
              FROM pg_catalog.regexp_matches(
                  pg_catalog.lower(pg_catalog.pg_get_constraintdef(constraint_row.oid)),
                  '''([^'']+)''(?:::text)?', 'g'
              ) AS captures
              ORDER BY captures[1]
          ) = ARRAY['filtered', 'included', 'pending', 'review']
    ) THEN
        RAISE EXCEPTION 'public.job_archetype_memberships requires JSON and filter-status checks'
            USING ERRCODE = '55000';
    END IF;
END;
$prerequisite$;

ALTER TABLE public.jobs
    ADD COLUMN IF NOT EXISTS canonical_revision bigint NOT NULL DEFAULT 0;
ALTER TABLE public.jobs
    DROP CONSTRAINT IF EXISTS jobs_canonical_revision_check;
ALTER TABLE public.jobs
    ADD CONSTRAINT jobs_canonical_revision_check CHECK (canonical_revision >= 0) NOT VALID;
ALTER TABLE public.jobs
    VALIDATE CONSTRAINT jobs_canonical_revision_check;

CREATE OR REPLACE FUNCTION public.increment_job_canonical_revision()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $$
BEGIN
    NEW.canonical_revision := OLD.canonical_revision + 1;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS maintain_job_canonical_revision ON public.jobs;
CREATE TRIGGER maintain_job_canonical_revision
BEFORE UPDATE ON public.jobs
FOR EACH ROW EXECUTE FUNCTION public.increment_job_canonical_revision();

CREATE TABLE IF NOT EXISTS public.canonical_provider_revisions (
    provider text PRIMARY KEY,
    revision bigint NOT NULL DEFAULT 0 CHECK (revision >= 0),
    updated_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp()
);

INSERT INTO public.canonical_provider_revisions (provider)
VALUES ('linkedin')
ON CONFLICT (provider) DO NOTHING;

INSERT INTO public.canonical_provider_revisions (provider)
SELECT DISTINCT job.provider
FROM public.jobs job
WHERE job.provider IS NOT NULL
ON CONFLICT (provider) DO NOTHING;

CREATE OR REPLACE FUNCTION public.bump_canonical_provider_revision()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = pg_catalog
AS $$
BEGIN
    INSERT INTO public.canonical_provider_revisions AS provider_revision (
        provider, revision, updated_at
    ) VALUES ('linkedin', 1, pg_catalog.clock_timestamp())
    ON CONFLICT (provider) DO UPDATE SET
        revision = provider_revision.revision + 1,
        updated_at = EXCLUDED.updated_at;
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS maintain_canonical_provider_revision ON public.jobs;
CREATE TRIGGER maintain_canonical_provider_revision
AFTER INSERT OR UPDATE OR DELETE ON public.jobs
FOR EACH STATEMENT EXECUTE FUNCTION public.bump_canonical_provider_revision();

CREATE OR REPLACE FUNCTION public.get_canonical_provider_revision(p_provider text)
RETURNS text
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog AS $$
SELECT pg_catalog.lpad(pg_catalog.to_hex(COALESCE((
    SELECT provider_revision.revision
    FROM public.canonical_provider_revisions provider_revision
    WHERE provider_revision.provider = p_provider
), 0)), 64, '0');
$$;

CREATE TABLE IF NOT EXISTS public.linkedin_source_request_policy (
    source text PRIMARY KEY CHECK (source = 'linkedin'),
    minimum_interval_ms integer NOT NULL CHECK (minimum_interval_ms BETWEEN 2500 AND 60000),
    grant_ttl_ms integer NOT NULL DEFAULT 30000 CHECK (grant_ttl_ms BETWEEN 1000 AND 60000),
    next_allowed_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    circuit_generation bigint NOT NULL DEFAULT 0 CHECK (circuit_generation >= 0),
    circuit_state text NOT NULL DEFAULT 'closed' CHECK (circuit_state IN ('closed', 'open')),
    circuit_reason text,
    opened_at timestamptz,
    open_until timestamptz,
    last_reset_at timestamptz,
    last_reset_actor text,
    last_reset_reason text,
    updated_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK (
        (circuit_state = 'closed' AND circuit_reason IS NULL AND opened_at IS NULL AND open_until IS NULL)
        OR (
            circuit_state = 'open'
            AND circuit_reason IS NOT NULL
            AND opened_at IS NOT NULL
            AND open_until IS NOT NULL
            AND open_until >= opened_at
        )
    )
);

INSERT INTO public.linkedin_source_request_policy (source, minimum_interval_ms)
VALUES ('linkedin', 2500)
ON CONFLICT (source) DO NOTHING;

CREATE TABLE IF NOT EXISTS public.linkedin_source_request_grants (
    id uuid PRIMARY KEY DEFAULT extensions.gen_random_uuid(),
    source text NOT NULL REFERENCES public.linkedin_source_request_policy(source) ON DELETE RESTRICT,
    producer text NOT NULL CHECK (pg_catalog.btrim(producer) <> ''),
    request_kind text NOT NULL CHECK (request_kind IN ('search', 'detail', 'activity_check', 'backfill')),
    request_key text NOT NULL CHECK (pg_catalog.btrim(request_key) <> ''),
    requested_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    circuit_generation bigint NOT NULL,
    status text NOT NULL CHECK (status IN ('pending', 'consumed', 'finished', 'expired', 'invalidated')),
    started_at timestamptz,
    finished_at timestamptz,
    response_class text,
    http_status integer CHECK (http_status IS NULL OR http_status BETWEEN 100 AND 599 OR http_status = 999),
    CHECK (expires_at > requested_at),
    CHECK ((status = 'pending') = (started_at IS NULL AND finished_at IS NULL)),
    CHECK (status <> 'consumed' OR (started_at IS NOT NULL AND finished_at IS NULL)),
    CHECK (status NOT IN ('finished', 'expired', 'invalidated') OR finished_at IS NOT NULL)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_linkedin_source_request_pending
    ON public.linkedin_source_request_grants (source) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_linkedin_source_request_grants_requested
    ON public.linkedin_source_request_grants (source, requested_at DESC);

CREATE TABLE IF NOT EXISTS public.linkedin_scope_coverage_state (
    scope_key text PRIMARY KEY,
    scope_definition_hash text NOT NULL UNIQUE CHECK (scope_definition_hash ~ '^[0-9a-f]{64}$'),
    scope_definition jsonb NOT NULL CHECK (pg_catalog.jsonb_typeof(scope_definition) = 'object'),
    config_revision bigint,
    config_content_hash text NOT NULL CHECK (config_content_hash ~ '^[0-9a-f]{64}$'),
    archetype text NOT NULL,
    query_id text NOT NULL,
    geography_id text NOT NULL,
    last_operational_success_at timestamptz,
    last_exhausted_at timestamptz,
    last_saturated_at timestamptz,
    consecutive_saturated_runs integer NOT NULL DEFAULT 0 CHECK (consecutive_saturated_runs >= 0),
    recommended_pages integer NOT NULL DEFAULT 6 CHECK (recommended_pages BETWEEN 1 AND 100),
    coverage_debt boolean NOT NULL DEFAULT false,
    coverage_debt_since timestamptz,
    latest_tail_workflow_new_ids integer NOT NULL DEFAULT 0 CHECK (latest_tail_workflow_new_ids >= 0),
    last_deep_sweep_at timestamptz,
    last_operational_discovery_sequence bigint,
    updated_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK (coverage_debt = (coverage_debt_since IS NOT NULL))
);
ALTER TABLE public.linkedin_scope_coverage_state
    ADD COLUMN IF NOT EXISTS last_deep_sweep_at timestamptz;

CREATE TABLE IF NOT EXISTS public.linkedin_discovery_cycles (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    execution_id uuid NOT NULL UNIQUE,
    discovery_sequence bigint NOT NULL UNIQUE,
    started_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    search_completed_at timestamptz,
    config_revision bigint,
    config_content_hash text NOT NULL CHECK (config_content_hash ~ '^[0-9a-f]{64}$'),
    required_scope_count integer NOT NULL CHECK (required_scope_count > 0),
    completed_scope_count integer NOT NULL DEFAULT 0 CHECK (completed_scope_count >= 0),
    search_status text NOT NULL DEFAULT 'running' CHECK (search_status IN ('running', 'sealed', 'failed')),
    canonical_status text NOT NULL DEFAULT 'pending' CHECK (canonical_status IN ('pending', 'applied')),
    coverage_debt_count integer NOT NULL DEFAULT 0 CHECK (coverage_debt_count >= 0),
    operational_watermark_eligible boolean NOT NULL DEFAULT false,
    pinned_user_agent text NOT NULL CHECK (pg_catalog.btrim(pinned_user_agent) <> ''),
    failure_reason text,
    created_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
ALTER TABLE public.linkedin_discovery_cycles
    ADD COLUMN IF NOT EXISTS operational_watermark_eligible boolean NOT NULL DEFAULT false;

ALTER TABLE public.ingestion_runs
    ADD COLUMN IF NOT EXISTS discovery_cycle_id bigint REFERENCES public.linkedin_discovery_cycles(id) ON DELETE RESTRICT,
    ADD COLUMN IF NOT EXISTS coverage_status text NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS failure_code text;

ALTER TABLE public.ingestion_runs DROP CONSTRAINT IF EXISTS ingestion_runs_coverage_status_check;
ALTER TABLE public.ingestion_runs ADD CONSTRAINT ingestion_runs_coverage_status_check
    CHECK (coverage_status IN ('unknown', 'exhausted', 'right_censored', 'failed')) NOT VALID;

ALTER TABLE public.listing_observations
    ADD COLUMN IF NOT EXISTS page_number integer,
    ADD COLUMN IF NOT EXISTS page_start integer,
    ADD COLUMN IF NOT EXISTS position_on_page integer,
    ADD COLUMN IF NOT EXISTS position_in_scope integer;

ALTER TABLE public.listing_observations DROP CONSTRAINT IF EXISTS listing_observations_page_position_check;
ALTER TABLE public.listing_observations ADD CONSTRAINT listing_observations_page_position_check CHECK (
    (page_number IS NULL AND page_start IS NULL AND position_on_page IS NULL AND position_in_scope IS NULL)
    OR (
        page_number IS NOT NULL AND page_number >= 1
        AND page_start IS NOT NULL AND page_start >= 0
        AND position_on_page IS NOT NULL AND position_on_page >= 0
        AND position_in_scope IS NOT NULL AND position_in_scope >= 0
    )
) NOT VALID;

CREATE TABLE IF NOT EXISTS public.linkedin_discovery_cycle_scopes (
    discovery_cycle_id bigint NOT NULL REFERENCES public.linkedin_discovery_cycles(id) ON DELETE RESTRICT,
    scope_key text NOT NULL REFERENCES public.linkedin_scope_coverage_state(scope_key) ON DELETE RESTRICT,
    ingestion_run_id uuid NOT NULL UNIQUE REFERENCES public.ingestion_runs(id) ON DELETE RESTRICT,
    query_scope text NOT NULL,
    required boolean NOT NULL DEFAULT true,
    request_anchor_at timestamptz NOT NULL,
    source_window_earliest_at timestamptz NOT NULL,
    source_window_latest_at timestamptz NOT NULL,
    truncated_window_earliest_at timestamptz,
    truncated_window_latest_at timestamptz,
    expired_window_earliest_at timestamptz,
    expired_window_latest_at timestamptz,
    minimum_pages integer NOT NULL CHECK (minimum_pages BETWEEN 1 AND 100),
    target_pages integer NOT NULL CHECK (target_pages BETWEEN 1 AND 100),
    status text NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'complete', 'failed')),
    enqueue_committed_at timestamptz,
    PRIMARY KEY (discovery_cycle_id, scope_key),
    UNIQUE (discovery_cycle_id, scope_key, ingestion_run_id),
    CHECK (source_window_earliest_at <= source_window_latest_at),
    CHECK ((truncated_window_earliest_at IS NULL) = (truncated_window_latest_at IS NULL)),
    CHECK (truncated_window_earliest_at IS NULL OR truncated_window_earliest_at < truncated_window_latest_at),
    CHECK ((expired_window_earliest_at IS NULL) = (expired_window_latest_at IS NULL)),
    CHECK (expired_window_earliest_at IS NULL OR expired_window_earliest_at < expired_window_latest_at),
    CHECK (minimum_pages <= target_pages)
);
ALTER TABLE public.linkedin_discovery_cycle_scopes
    ADD COLUMN IF NOT EXISTS truncated_window_earliest_at timestamptz,
    ADD COLUMN IF NOT EXISTS truncated_window_latest_at timestamptz,
    ADD COLUMN IF NOT EXISTS expired_window_earliest_at timestamptz,
    ADD COLUMN IF NOT EXISTS expired_window_latest_at timestamptz;

CREATE TABLE IF NOT EXISTS public.linkedin_ingestion_pages (
    ingestion_run_id uuid NOT NULL REFERENCES public.ingestion_runs(id) ON DELETE RESTRICT,
    page_number integer NOT NULL CHECK (page_number >= 1),
    page_start integer NOT NULL CHECK (page_start >= 0),
    requested_at timestamptz NOT NULL,
    source_window_earliest_at timestamptz NOT NULL,
    source_window_latest_at timestamptz NOT NULL,
    element_count integer NOT NULL CHECK (element_count >= 0),
    card_count integer NOT NULL CHECK (card_count >= 0),
    new_source_ids integer NOT NULL DEFAULT 0 CHECK (new_source_ids >= 0),
    new_workflow_source_ids integer NOT NULL DEFAULT 0 CHECK (new_workflow_source_ids >= 0),
    known_source_ids integer NOT NULL DEFAULT 0 CHECK (known_source_ids >= 0),
    result text NOT NULL CHECK (result IN ('cards', 'no_results')),
    request_attempts integer NOT NULL CHECK (request_attempts >= 1),
    elapsed_ms integer NOT NULL CHECK (elapsed_ms >= 0),
    classifier_version text NOT NULL,
    response_fingerprint text NOT NULL CHECK (response_fingerprint ~ '^[0-9a-f]{64}$'),
    membership_fingerprint text NOT NULL CHECK (membership_fingerprint ~ '^[0-9a-f]{64}$'),
    committed_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (ingestion_run_id, page_number),
    UNIQUE (ingestion_run_id, page_start),
    CHECK (source_window_earliest_at <= source_window_latest_at),
    CHECK (card_count <= element_count)
);

CREATE TABLE IF NOT EXISTS public.linkedin_ingestion_page_sources (
    ingestion_run_id uuid NOT NULL,
    page_number integer NOT NULL,
    provider text NOT NULL DEFAULT 'linkedin' CHECK (provider = 'linkedin'),
    source_job_id text NOT NULL,
    position_on_page integer NOT NULL CHECK (position_on_page >= 0),
    position_in_scope integer NOT NULL CHECK (position_in_scope >= 0),
    PRIMARY KEY (ingestion_run_id, page_number, provider, source_job_id),
    UNIQUE (ingestion_run_id, page_number, position_on_page),
    FOREIGN KEY (ingestion_run_id, page_number)
        REFERENCES public.linkedin_ingestion_pages(ingestion_run_id, page_number) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS public.linkedin_discovery_cycle_sources (
    discovery_cycle_id bigint NOT NULL REFERENCES public.linkedin_discovery_cycles(id) ON DELETE RESTRICT,
    provider text NOT NULL DEFAULT 'linkedin' CHECK (provider = 'linkedin'),
    source_job_id text NOT NULL,
    first_ingestion_run_id uuid NOT NULL REFERENCES public.ingestion_runs(id) ON DELETE RESTRICT,
    first_page_number integer NOT NULL,
    first_position_on_page integer NOT NULL,
    PRIMARY KEY (discovery_cycle_id, provider, source_job_id)
);

CREATE TABLE IF NOT EXISTS public.linkedin_discovery_tasks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    provider text NOT NULL DEFAULT 'linkedin' CHECK (provider = 'linkedin'),
    source_job_id text NOT NULL,
    task_kind text NOT NULL DEFAULT 'initial_detail' CHECK (task_kind IN ('initial_detail', 'availability_revalidation')),
    requirement_key text NOT NULL,
    first_ingestion_run_id uuid NOT NULL REFERENCES public.ingestion_runs(id) ON DELETE RESTRICT,
    first_query_scope text NOT NULL,
    first_observed_at timestamptz NOT NULL,
    latest_observed_at timestamptz NOT NULL,
    posted_at date,
    search_card jsonb NOT NULL CHECK (pg_catalog.jsonb_typeof(search_card) = 'object'),
    provenance jsonb NOT NULL DEFAULT '{}'::jsonb CHECK (pg_catalog.jsonb_typeof(provenance) = 'object'),
    membership_provenances jsonb NOT NULL DEFAULT '[]'::jsonb,
    membership_provenance_revision bigint NOT NULL DEFAULT 0 CHECK (membership_provenance_revision >= 0),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'leased', 'complete', 'terminal_unavailable', 'failed_retryable', 'failed_terminal')),
    priority integer NOT NULL DEFAULT 100 CHECK (priority >= 0),
    attempt_count integer NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    max_attempts integer NOT NULL DEFAULT 5 CHECK (max_attempts > 0),
    available_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    leased_by text,
    leased_at timestamptz,
    lease_expires_at timestamptz,
    lease_token uuid,
    last_error_code text,
    completed_at timestamptz,
    canonical_job_id text REFERENCES public.jobs(job_id) ON DELETE RESTRICT,
    canonical_applied_lease_token uuid,
    canonical_application_hash text,
    UNIQUE (provider, source_job_id, task_kind, requirement_key),
    UNIQUE (id, provider, source_job_id, task_kind, requirement_key),
    CHECK (first_observed_at <= latest_observed_at),
    CHECK ((status = 'leased') = (leased_by IS NOT NULL AND leased_at IS NOT NULL AND lease_expires_at IS NOT NULL AND lease_token IS NOT NULL)),
    CHECK (attempt_count <= max_attempts),
    CHECK (status <> 'complete' OR (canonical_job_id IS NOT NULL AND completed_at IS NOT NULL)),
    CHECK (status NOT IN ('terminal_unavailable', 'failed_terminal') OR completed_at IS NOT NULL),
    CHECK (status NOT IN ('pending', 'leased', 'failed_retryable') OR completed_at IS NULL)
);
ALTER TABLE public.linkedin_discovery_tasks
    ADD COLUMN IF NOT EXISTS membership_provenances jsonb,
    ADD COLUMN IF NOT EXISTS membership_provenance_revision bigint NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS canonical_applied_lease_token uuid,
    ADD COLUMN IF NOT EXISTS canonical_application_hash text;
ALTER TABLE public.linkedin_discovery_tasks
    DROP CONSTRAINT IF EXISTS linkedin_discovery_tasks_membership_provenance_revision_check;
ALTER TABLE public.linkedin_discovery_tasks
    ADD CONSTRAINT linkedin_discovery_tasks_membership_provenance_revision_check
    CHECK (membership_provenance_revision >= 0) NOT VALID;
ALTER TABLE public.linkedin_discovery_tasks
    VALIDATE CONSTRAINT linkedin_discovery_tasks_membership_provenance_revision_check;
ALTER TABLE public.linkedin_discovery_tasks
    DROP CONSTRAINT IF EXISTS linkedin_discovery_tasks_canonical_receipt_check;
ALTER TABLE public.linkedin_discovery_tasks
    ADD CONSTRAINT linkedin_discovery_tasks_canonical_receipt_check CHECK (
        (canonical_applied_lease_token IS NULL) = (canonical_application_hash IS NULL)
        AND (
            canonical_application_hash IS NULL
            OR canonical_application_hash ~ '^[0-9a-f]{64}$'
        )
    ) NOT VALID;
ALTER TABLE public.linkedin_discovery_tasks
    VALIDATE CONSTRAINT linkedin_discovery_tasks_canonical_receipt_check;

CREATE TABLE IF NOT EXISTS public.linkedin_discovery_requirements (
    discovery_cycle_id bigint NOT NULL REFERENCES public.linkedin_discovery_cycles(id) ON DELETE RESTRICT,
    ingestion_run_id uuid NOT NULL REFERENCES public.ingestion_runs(id) ON DELETE RESTRICT,
    provider text NOT NULL,
    source_job_id text NOT NULL,
    task_kind text NOT NULL,
    requirement_key text NOT NULL,
    task_id bigint NOT NULL,
    membership_provenance jsonb NOT NULL,
    required boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (discovery_cycle_id, ingestion_run_id, provider, source_job_id, task_kind, requirement_key),
    FOREIGN KEY (task_id, provider, source_job_id, task_kind, requirement_key)
        REFERENCES public.linkedin_discovery_tasks(id, provider, source_job_id, task_kind, requirement_key) ON DELETE RESTRICT
);

ALTER TABLE public.linkedin_discovery_requirements
    ADD COLUMN IF NOT EXISTS membership_provenance jsonb;

WITH requirement_evidence AS (
    SELECT requirement.discovery_cycle_id, requirement.ingestion_run_id,
           requirement.provider, requirement.source_job_id,
           requirement.task_kind, requirement.requirement_key,
           pg_catalog.jsonb_strip_nulls(pg_catalog.jsonb_build_object(
               'scope_key', scope.scope_key,
               'lane', COALESCE(
                   scope_state.archetype,
                   scope_state.scope_definition->>'lane',
                   task.provenance->>'lane',
                   task.provenance->>'archetype',
                   ingestion.archetype
               ),
               'archetype', COALESCE(
                   scope_state.archetype,
                   scope_state.scope_definition->>'lane',
                   task.provenance->>'archetype',
                   task.provenance->>'lane',
                   ingestion.archetype
               ),
               'query_id', COALESCE(
                   scope_state.query_id,
                   task.provenance->>'search_query_id',
                   task.provenance->>'query_id'
               ),
               'query', COALESCE(
                   scope_state.scope_definition->>'query',
                   task.provenance->>'search_query',
                   task.provenance->>'query',
                   ingestion.search_query
               ),
               'query_type', COALESCE(
                   scope_state.scope_definition->>'query_type',
                   task.provenance->>'search_query_type',
                   task.provenance->>'query_type'
               ),
               'language', COALESCE(
                   scope_state.scope_definition->>'language',
                   task.provenance->>'search_query_language',
                   task.provenance->>'language'
               ),
               'location_scope', COALESCE(
                   scope_state.scope_definition->>'location_scope',
                   task.provenance->>'search_location_scope',
                   task.provenance->>'location_scope'
               ),
               'geography_id', COALESCE(
                   scope_state.geography_id,
                   scope_state.scope_definition->>'geography_id',
                   task.provenance->>'geography_id'
               ),
               'observed_at', pg_catalog.to_jsonb(COALESCE(
                   (
                       SELECT MIN(page.requested_at)
                       FROM public.linkedin_ingestion_pages page
                       JOIN public.linkedin_ingestion_page_sources page_source
                         ON page_source.ingestion_run_id = page.ingestion_run_id
                        AND page_source.page_number = page.page_number
                       WHERE page.ingestion_run_id = requirement.ingestion_run_id
                         AND page_source.provider = requirement.provider
                         AND page_source.source_job_id = requirement.source_job_id
                   ),
                   task.first_observed_at
               ))
           )) AS membership_provenance
    FROM public.linkedin_discovery_requirements requirement
    JOIN public.linkedin_discovery_tasks task ON task.id = requirement.task_id
    LEFT JOIN public.linkedin_discovery_cycle_scopes scope
      ON scope.discovery_cycle_id = requirement.discovery_cycle_id
     AND scope.ingestion_run_id = requirement.ingestion_run_id
    LEFT JOIN public.linkedin_scope_coverage_state scope_state
      ON scope_state.scope_key = scope.scope_key
    LEFT JOIN public.ingestion_runs ingestion ON ingestion.id = requirement.ingestion_run_id
)
UPDATE public.linkedin_discovery_requirements requirement
SET membership_provenance = evidence.membership_provenance
FROM requirement_evidence evidence
WHERE requirement.discovery_cycle_id = evidence.discovery_cycle_id
  AND requirement.ingestion_run_id = evidence.ingestion_run_id
  AND requirement.provider = evidence.provider
  AND requirement.source_job_id = evidence.source_job_id
  AND requirement.task_kind = evidence.task_kind
  AND requirement.requirement_key = evidence.requirement_key
  AND requirement.membership_provenance IS NULL;

WITH task_evidence AS (
    SELECT task.id,
           COALESCE(
               (
                   SELECT pg_catalog.jsonb_agg(distinct_provenance.value ORDER BY distinct_provenance.value::text)
                   FROM (
                       SELECT DISTINCT requirement.membership_provenance AS value
                       FROM public.linkedin_discovery_requirements requirement
                       WHERE requirement.task_id = task.id
                   ) distinct_provenance
               ),
               pg_catalog.jsonb_build_array(pg_catalog.jsonb_strip_nulls(
                   task.provenance || pg_catalog.jsonb_build_object(
                       'lane', COALESCE(
                           task.provenance->>'lane', task.provenance->>'archetype', ingestion.archetype
                       ),
                       'archetype', COALESCE(
                           task.provenance->>'archetype', task.provenance->>'lane', ingestion.archetype
                       ),
                       'query_id', COALESCE(
                           task.provenance->>'search_query_id', task.provenance->>'query_id'
                       ),
                       'query', COALESCE(
                           task.provenance->>'search_query', task.provenance->>'query', ingestion.search_query
                       ),
                       'query_type', COALESCE(
                           task.provenance->>'search_query_type', task.provenance->>'query_type'
                       ),
                       'language', COALESCE(
                           task.provenance->>'search_query_language', task.provenance->>'language'
                       ),
                       'location_scope', COALESCE(
                           task.provenance->>'search_location_scope', task.provenance->>'location_scope'
                       ),
                       'geography_id', task.provenance->>'geography_id',
                       'observed_at', pg_catalog.to_jsonb(task.first_observed_at)
                   )
               ))
           ) AS membership_provenances
    FROM public.linkedin_discovery_tasks task
    LEFT JOIN public.ingestion_runs ingestion ON ingestion.id = task.first_ingestion_run_id
)
UPDATE public.linkedin_discovery_tasks task
SET membership_provenances = evidence.membership_provenances
FROM task_evidence evidence
WHERE task.id = evidence.id
  AND task.membership_provenances IS NULL;

ALTER TABLE public.linkedin_discovery_tasks
    ALTER COLUMN membership_provenances SET DEFAULT '[]'::jsonb,
    ALTER COLUMN membership_provenances SET NOT NULL;
ALTER TABLE public.linkedin_discovery_tasks
    DROP CONSTRAINT IF EXISTS linkedin_discovery_tasks_membership_provenances_check;
ALTER TABLE public.linkedin_discovery_tasks
    ADD CONSTRAINT linkedin_discovery_tasks_membership_provenances_check CHECK (
        pg_catalog.jsonb_typeof(membership_provenances) = 'array'
        AND NOT pg_catalog.jsonb_path_exists(
            membership_provenances, '$[*] ? (@.type() != "object")'
        )
    ) NOT VALID;
ALTER TABLE public.linkedin_discovery_tasks
    VALIDATE CONSTRAINT linkedin_discovery_tasks_membership_provenances_check;

ALTER TABLE public.linkedin_discovery_requirements
    ALTER COLUMN membership_provenance SET NOT NULL;
ALTER TABLE public.linkedin_discovery_requirements
    DROP CONSTRAINT IF EXISTS linkedin_discovery_requirements_membership_provenance_check;
ALTER TABLE public.linkedin_discovery_requirements
    ADD CONSTRAINT linkedin_discovery_requirements_membership_provenance_check CHECK (
        pg_catalog.jsonb_typeof(membership_provenance) = 'object'
    ) NOT VALID;
ALTER TABLE public.linkedin_discovery_requirements
    VALIDATE CONSTRAINT linkedin_discovery_requirements_membership_provenance_check;

CREATE OR REPLACE FUNCTION public.reject_linkedin_requirement_provenance_change()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $$
BEGIN
    IF NEW.membership_provenance IS DISTINCT FROM OLD.membership_provenance THEN
        RAISE EXCEPTION 'discovery requirement membership provenance is immutable'
            USING ERRCODE = '23000';
    END IF;
    RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS reject_linkedin_requirement_provenance_change
    ON public.linkedin_discovery_requirements;
CREATE TRIGGER reject_linkedin_requirement_provenance_change
BEFORE UPDATE OF membership_provenance ON public.linkedin_discovery_requirements
FOR EACH ROW EXECUTE FUNCTION public.reject_linkedin_requirement_provenance_change();
REVOKE ALL ON FUNCTION public.reject_linkedin_requirement_provenance_change() FROM PUBLIC;

CREATE TABLE IF NOT EXISTS public.linkedin_coverage_debt (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    scope_key text NOT NULL REFERENCES public.linkedin_scope_coverage_state(scope_key) ON DELETE RESTRICT,
    origin_discovery_cycle_id bigint NOT NULL,
    origin_ingestion_run_id uuid NOT NULL,
    debt_kind text NOT NULL CHECK (debt_kind IN ('search_right_censored', 'lookback_truncated', 'search_failed', 'scope_unattempted_after_cycle_failure')),
    source_window_earliest_at timestamptz NOT NULL,
    source_window_latest_at timestamptz NOT NULL,
    page_cap integer NOT NULL CHECK (page_cap > 0),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'resolved', 'expired_unresolved', 'accepted_boundary')),
    created_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    resolved_at timestamptz,
    resolution text,
    reviewer text,
    resolved_by_discovery_cycle_id bigint REFERENCES public.linkedin_discovery_cycles(id) ON DELETE RESTRICT,
    resolved_by_ingestion_run_id uuid REFERENCES public.ingestion_runs(id) ON DELETE RESTRICT,
    UNIQUE (scope_key, origin_ingestion_run_id, debt_kind, source_window_earliest_at, source_window_latest_at),
    FOREIGN KEY (origin_discovery_cycle_id, scope_key, origin_ingestion_run_id)
        REFERENCES public.linkedin_discovery_cycle_scopes(discovery_cycle_id, scope_key, ingestion_run_id) ON DELETE RESTRICT
);
ALTER TABLE public.linkedin_coverage_debt
    ADD COLUMN IF NOT EXISTS reviewer text,
    ADD COLUMN IF NOT EXISTS resolved_by_discovery_cycle_id bigint REFERENCES public.linkedin_discovery_cycles(id) ON DELETE RESTRICT,
    ADD COLUMN IF NOT EXISTS resolved_by_ingestion_run_id uuid REFERENCES public.ingestion_runs(id) ON DELETE RESTRICT;
ALTER TABLE public.linkedin_coverage_debt
    DROP CONSTRAINT IF EXISTS linkedin_coverage_debt_debt_kind_check;
ALTER TABLE public.linkedin_coverage_debt
    ADD CONSTRAINT linkedin_coverage_debt_debt_kind_check
    CHECK (debt_kind IN (
        'search_right_censored', 'lookback_truncated', 'search_failed',
        'scope_unattempted_after_cycle_failure'
    )) NOT VALID;

CREATE TABLE IF NOT EXISTS public.linkedin_discovery_cycle_resolutions (
    failed_discovery_cycle_id bigint PRIMARY KEY REFERENCES public.linkedin_discovery_cycles(id) ON DELETE RESTRICT,
    resolving_discovery_cycle_id bigint NOT NULL REFERENCES public.linkedin_discovery_cycles(id) ON DELETE RESTRICT,
    resolution_type text NOT NULL CHECK (resolution_type IN ('recovered', 'reviewed_acceptance')),
    reviewer text,
    reason text NOT NULL CHECK (pg_catalog.btrim(reason) <> ''),
    created_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK (failed_discovery_cycle_id <> resolving_discovery_cycle_id),
    CHECK (resolution_type <> 'reviewed_acceptance' OR (reviewer IS NOT NULL AND pg_catalog.btrim(reviewer) <> ''))
);

CREATE TABLE IF NOT EXISTS public.linkedin_coverage_debt_attempts (
    debt_id bigint NOT NULL REFERENCES public.linkedin_coverage_debt(id) ON DELETE RESTRICT,
    recovery_ingestion_run_id uuid NOT NULL REFERENCES public.ingestion_runs(id) ON DELETE RESTRICT,
    recovery_discovery_cycle_id bigint NOT NULL REFERENCES public.linkedin_discovery_cycles(id) ON DELETE RESTRICT,
    requested_window_earliest_at timestamptz NOT NULL,
    requested_window_latest_at timestamptz NOT NULL,
    requested_page_cap integer NOT NULL CHECK (requested_page_cap > 0),
    outcome text NOT NULL CHECK (outcome IN ('resolved', 'right_censored', 'not_contained', 'failed')),
    created_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (debt_id, recovery_ingestion_run_id),
    CHECK (requested_window_earliest_at <= requested_window_latest_at)
);
ALTER TABLE public.linkedin_coverage_debt_attempts
    DROP CONSTRAINT IF EXISTS linkedin_coverage_debt_attempts_outcome_check;
ALTER TABLE public.linkedin_coverage_debt_attempts
    ADD CONSTRAINT linkedin_coverage_debt_attempts_outcome_check
    CHECK (outcome IN ('resolved', 'right_censored', 'not_contained', 'failed')) NOT VALID;
ALTER TABLE public.linkedin_coverage_debt_attempts
    VALIDATE CONSTRAINT linkedin_coverage_debt_attempts_outcome_check;

CREATE TABLE IF NOT EXISTS public.linkedin_discovery_requirement_acceptances (
    discovery_cycle_id bigint NOT NULL,
    ingestion_run_id uuid NOT NULL,
    provider text NOT NULL,
    source_job_id text NOT NULL,
    task_kind text NOT NULL,
    requirement_key text NOT NULL,
    reviewer text NOT NULL CHECK (pg_catalog.btrim(reviewer) <> ''),
    reason text NOT NULL CHECK (pg_catalog.btrim(reason) <> ''),
    created_at timestamptz NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    PRIMARY KEY (
        discovery_cycle_id, ingestion_run_id, provider, source_job_id,
        task_kind, requirement_key
    ),
    FOREIGN KEY (
        discovery_cycle_id, ingestion_run_id, provider, source_job_id,
        task_kind, requirement_key
    ) REFERENCES public.linkedin_discovery_requirements (
        discovery_cycle_id, ingestion_run_id, provider, source_job_id,
        task_kind, requirement_key
    ) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS public.linkedin_discovery_task_attempts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    task_id bigint NOT NULL REFERENCES public.linkedin_discovery_tasks(id) ON DELETE RESTRICT,
    lease_token uuid NOT NULL,
    request_attempt integer NOT NULL CHECK (request_attempt >= 0),
    request_grant_id uuid NOT NULL UNIQUE REFERENCES public.linkedin_source_request_grants(id) ON DELETE RESTRICT,
    response_class text NOT NULL DEFAULT 'started',
    http_status integer,
    parser_version text,
    started_at timestamptz NOT NULL,
    finished_at timestamptz,
    UNIQUE (task_id, lease_token, request_attempt),
    CHECK (finished_at IS NULL OR started_at <= finished_at)
);
ALTER TABLE public.linkedin_discovery_task_attempts
    ALTER COLUMN response_class SET DEFAULT 'started',
    ALTER COLUMN finished_at DROP NOT NULL;

ALTER TABLE public.scrape_run_state
    ADD COLUMN IF NOT EXISTS last_successful_discovery_cycle_id bigint REFERENCES public.linkedin_discovery_cycles(id) ON DELETE RESTRICT,
    ADD COLUMN IF NOT EXISTS last_successful_discovery_sequence bigint;

ALTER TABLE public.freehire_publication_state
    ADD COLUMN IF NOT EXISTS source_discovery_cycle_id bigint REFERENCES public.linkedin_discovery_cycles(id) ON DELETE RESTRICT,
    ADD COLUMN IF NOT EXISTS source_discovery_sequence bigint;
ALTER TABLE public.freehire_publication_generations
    ADD COLUMN IF NOT EXISTS source_discovery_cycle_id bigint REFERENCES public.linkedin_discovery_cycles(id) ON DELETE RESTRICT,
    ADD COLUMN IF NOT EXISTS source_discovery_sequence bigint;

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_discovery_cycle ON public.ingestion_runs(discovery_cycle_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_linkedin_cycles_id_sequence
    ON public.linkedin_discovery_cycles(id, discovery_sequence);
CREATE UNIQUE INDEX IF NOT EXISTS uq_linkedin_cycle_scopes_cycle_run
    ON public.linkedin_discovery_cycle_scopes(discovery_cycle_id, ingestion_run_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_ingestion_runs_discovery_cycle_id
    ON public.ingestion_runs(discovery_cycle_id, id);
CREATE INDEX IF NOT EXISTS idx_listing_observations_barrier ON public.listing_observations(ingestion_run_id, provider, source_job_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_page_sources_source ON public.linkedin_ingestion_page_sources(provider, source_job_id, ingestion_run_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_cycle_sources_source ON public.linkedin_discovery_cycle_sources(provider, source_job_id, discovery_cycle_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_cycle_scopes_scope ON public.linkedin_discovery_cycle_scopes(scope_key);
CREATE INDEX IF NOT EXISTS idx_linkedin_cycle_sources_run ON public.linkedin_discovery_cycle_sources(first_ingestion_run_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_cycle_sources_cycle_run
    ON public.linkedin_discovery_cycle_sources(discovery_cycle_id, first_ingestion_run_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_requirements_task ON public.linkedin_discovery_requirements(task_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_requirements_run ON public.linkedin_discovery_requirements(ingestion_run_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_requirements_cycle_run
    ON public.linkedin_discovery_requirements(discovery_cycle_id, ingestion_run_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_requirements_barrier ON public.linkedin_discovery_requirements(discovery_cycle_id, task_id) WHERE required;
CREATE INDEX IF NOT EXISTS idx_linkedin_tasks_claim ON public.linkedin_discovery_tasks(priority DESC, first_observed_at, id) WHERE status IN ('pending', 'failed_retryable');
CREATE INDEX IF NOT EXISTS idx_linkedin_tasks_claim_newest ON public.linkedin_discovery_tasks(priority DESC, first_observed_at DESC, id) WHERE status IN ('pending', 'failed_retryable');
CREATE INDEX IF NOT EXISTS idx_linkedin_tasks_expired ON public.linkedin_discovery_tasks(lease_expires_at, id) WHERE status = 'leased';
CREATE INDEX IF NOT EXISTS idx_linkedin_tasks_run ON public.linkedin_discovery_tasks(first_ingestion_run_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_tasks_canonical ON public.linkedin_discovery_tasks(canonical_job_id) WHERE canonical_job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_linkedin_debt_pending ON public.linkedin_coverage_debt(created_at, scope_key, id) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_linkedin_debt_origin ON public.linkedin_coverage_debt(origin_discovery_cycle_id, scope_key, origin_ingestion_run_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_debt_resolving_cycle
    ON public.linkedin_coverage_debt(resolved_by_discovery_cycle_id)
    WHERE resolved_by_discovery_cycle_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_linkedin_debt_resolving_run
    ON public.linkedin_coverage_debt(resolved_by_ingestion_run_id)
    WHERE resolved_by_ingestion_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_linkedin_debt_attempts_recovery_run
    ON public.linkedin_coverage_debt_attempts(recovery_ingestion_run_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_debt_attempts_recovery_cycle
    ON public.linkedin_coverage_debt_attempts(recovery_discovery_cycle_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_cycle_resolutions_resolving
    ON public.linkedin_discovery_cycle_resolutions(resolving_discovery_cycle_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_requirement_acceptances_run
    ON public.linkedin_discovery_requirement_acceptances(ingestion_run_id);
CREATE INDEX IF NOT EXISTS idx_linkedin_task_attempts_task
    ON public.linkedin_discovery_task_attempts(task_id, started_at);
CREATE INDEX IF NOT EXISTS idx_freehire_generations_discovery_cycle ON public.freehire_publication_generations(source_discovery_cycle_id);
CREATE UNIQUE INDEX IF NOT EXISTS uq_freehire_generations_discovery_sequence
    ON public.freehire_publication_generations(source_discovery_sequence)
    WHERE source_discovery_sequence IS NOT NULL;

ALTER TABLE public.linkedin_discovery_cycle_sources
    DROP CONSTRAINT IF EXISTS linkedin_cycle_sources_scope_run_fk;
ALTER TABLE public.linkedin_discovery_cycle_sources
    ADD CONSTRAINT linkedin_cycle_sources_scope_run_fk
    FOREIGN KEY (discovery_cycle_id, first_ingestion_run_id)
    REFERENCES public.linkedin_discovery_cycle_scopes(discovery_cycle_id, ingestion_run_id)
    ON DELETE RESTRICT NOT VALID;
ALTER TABLE public.linkedin_discovery_cycle_sources
    DROP CONSTRAINT IF EXISTS linkedin_cycle_sources_page_source_fk;
ALTER TABLE public.linkedin_discovery_cycle_sources
    ADD CONSTRAINT linkedin_cycle_sources_page_source_fk
    FOREIGN KEY (first_ingestion_run_id, first_page_number, provider, source_job_id)
    REFERENCES public.linkedin_ingestion_page_sources(
        ingestion_run_id, page_number, provider, source_job_id
    ) ON DELETE RESTRICT NOT VALID;
ALTER TABLE public.linkedin_discovery_requirements
    DROP CONSTRAINT IF EXISTS linkedin_requirements_scope_run_fk;
ALTER TABLE public.linkedin_discovery_requirements
    ADD CONSTRAINT linkedin_requirements_scope_run_fk
    FOREIGN KEY (discovery_cycle_id, ingestion_run_id)
    REFERENCES public.linkedin_discovery_cycle_scopes(discovery_cycle_id, ingestion_run_id)
    ON DELETE RESTRICT NOT VALID;
ALTER TABLE public.linkedin_coverage_debt
    DROP CONSTRAINT IF EXISTS linkedin_debt_resolving_cycle_run_fk;
ALTER TABLE public.linkedin_coverage_debt
    ADD CONSTRAINT linkedin_debt_resolving_cycle_run_fk
    FOREIGN KEY (resolved_by_discovery_cycle_id, resolved_by_ingestion_run_id)
    REFERENCES public.ingestion_runs(discovery_cycle_id, id)
    ON DELETE RESTRICT NOT VALID;
ALTER TABLE public.scrape_run_state
    DROP CONSTRAINT IF EXISTS scrape_run_state_discovery_pair_fk;
ALTER TABLE public.scrape_run_state
    ADD CONSTRAINT scrape_run_state_discovery_pair_fk
    FOREIGN KEY (last_successful_discovery_cycle_id, last_successful_discovery_sequence)
    REFERENCES public.linkedin_discovery_cycles(id, discovery_sequence)
    ON DELETE RESTRICT NOT VALID;
ALTER TABLE public.freehire_publication_state
    DROP CONSTRAINT IF EXISTS freehire_publication_state_discovery_pair_fk;
ALTER TABLE public.freehire_publication_state
    ADD CONSTRAINT freehire_publication_state_discovery_pair_fk
    FOREIGN KEY (source_discovery_cycle_id, source_discovery_sequence)
    REFERENCES public.linkedin_discovery_cycles(id, discovery_sequence)
    ON DELETE RESTRICT NOT VALID;
ALTER TABLE public.freehire_publication_generations
    DROP CONSTRAINT IF EXISTS freehire_publication_generations_discovery_pair_fk;
ALTER TABLE public.freehire_publication_generations
    ADD CONSTRAINT freehire_publication_generations_discovery_pair_fk
    FOREIGN KEY (source_discovery_cycle_id, source_discovery_sequence)
    REFERENCES public.linkedin_discovery_cycles(id, discovery_sequence)
    ON DELETE RESTRICT NOT VALID;
ALTER TABLE public.linkedin_discovery_cycle_sources
    VALIDATE CONSTRAINT linkedin_cycle_sources_scope_run_fk;
ALTER TABLE public.linkedin_discovery_cycle_sources
    VALIDATE CONSTRAINT linkedin_cycle_sources_page_source_fk;
ALTER TABLE public.linkedin_discovery_requirements
    VALIDATE CONSTRAINT linkedin_requirements_scope_run_fk;
ALTER TABLE public.linkedin_coverage_debt
    VALIDATE CONSTRAINT linkedin_debt_resolving_cycle_run_fk;
ALTER TABLE public.scrape_run_state
    VALIDATE CONSTRAINT scrape_run_state_discovery_pair_fk;
ALTER TABLE public.freehire_publication_state
    VALIDATE CONSTRAINT freehire_publication_state_discovery_pair_fk;
ALTER TABLE public.freehire_publication_generations
    VALIDATE CONSTRAINT freehire_publication_generations_discovery_pair_fk;

CREATE OR REPLACE FUNCTION public.acquire_linkedin_request_grant(
    p_producer text, p_request_kind text, p_request_key text
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    policy public.linkedin_source_request_policy%ROWTYPE;
    grant_id uuid;
    v_now timestamptz := pg_catalog.clock_timestamp();
    wait_ms integer;
BEGIN
    IF p_producer IS NULL OR pg_catalog.btrim(p_producer) = ''
       OR p_request_kind NOT IN ('search', 'detail', 'activity_check', 'backfill')
       OR p_request_key IS NULL OR pg_catalog.btrim(p_request_key) = '' THEN
        RAISE EXCEPTION 'invalid request grant parameters' USING ERRCODE = '22023';
    END IF;
    SELECT * INTO STRICT policy FROM public.linkedin_source_request_policy
    WHERE source = 'linkedin' FOR UPDATE;
    UPDATE public.linkedin_source_request_grants
    SET status = 'expired', finished_at = v_now
    WHERE source = 'linkedin' AND status = 'pending' AND expires_at <= v_now;
    IF policy.circuit_state = 'open' THEN
        RETURN pg_catalog.jsonb_build_object('outcome', 'circuit_open', 'reason', policy.circuit_reason);
    END IF;
    IF v_now < policy.next_allowed_at THEN
        wait_ms := pg_catalog.ceil(EXTRACT(EPOCH FROM (policy.next_allowed_at - v_now)) * 1000)::integer;
        RETURN pg_catalog.jsonb_build_object('outcome', 'wait', 'wait_ms', wait_ms);
    END IF;
    IF EXISTS (SELECT 1 FROM public.linkedin_source_request_grants WHERE source = 'linkedin' AND status = 'pending') THEN
        RETURN pg_catalog.jsonb_build_object('outcome', 'wait', 'wait_ms', 250);
    END IF;
    IF p_producer = 'adaptive-detail'
       AND p_request_key !~ '^task:[0-9]+:[0-9a-f-]{36}:[^:]+:[0-9]+$' THEN
        RAISE EXCEPTION 'adaptive detail request key is invalid'
            USING ERRCODE = '22023';
    END IF;
    IF p_producer = 'adaptive-detail'
       AND NOT EXISTS (
           SELECT 1
           FROM public.linkedin_discovery_tasks task
           WHERE task.id = pg_catalog.split_part(p_request_key, ':', 2)::bigint
             AND task.lease_token = pg_catalog.split_part(p_request_key, ':', 3)::uuid
             AND task.status = 'leased'
             AND task.lease_expires_at > v_now
       ) THEN
        RAISE EXCEPTION 'adaptive detail request requires an active task lease'
            USING ERRCODE = '55000';
    END IF;
    INSERT INTO public.linkedin_source_request_grants (
        source, producer, request_kind, request_key, requested_at, expires_at,
        circuit_generation, status, started_at
    ) VALUES (
        'linkedin', p_producer, p_request_kind, p_request_key, v_now,
        v_now + pg_catalog.make_interval(secs => policy.grant_ttl_ms / 1000.0),
        policy.circuit_generation, 'consumed', v_now
    ) RETURNING id INTO grant_id;
    UPDATE public.linkedin_source_request_policy
    SET next_allowed_at = v_now + pg_catalog.make_interval(secs => minimum_interval_ms / 1000.0),
        updated_at = v_now
    WHERE source = 'linkedin';
    IF p_producer = 'adaptive-detail'
       AND p_request_key ~ '^task:[0-9]+:[0-9a-f-]{36}:[^:]+:[0-9]+$' THEN
        INSERT INTO public.linkedin_discovery_task_attempts (
            task_id, lease_token, request_attempt, request_grant_id,
            response_class, parser_version, started_at
        ) VALUES (
            pg_catalog.split_part(p_request_key, ':', 2)::bigint,
            pg_catalog.split_part(p_request_key, ':', 3)::uuid,
            pg_catalog.split_part(p_request_key, ':', 5)::integer,
            grant_id, 'started', 'linkedin-detail-v1', v_now
        );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'outcome', 'grant', 'grant_id', grant_id, 'started_at', v_now,
        'consumed', true
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.consume_linkedin_request_grant(
    p_grant_id uuid, p_producer text
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    grant_row public.linkedin_source_request_grants%ROWTYPE;
    policy public.linkedin_source_request_policy%ROWTYPE;
    started timestamptz := pg_catalog.clock_timestamp();
BEGIN
    SELECT * INTO STRICT policy FROM public.linkedin_source_request_policy
    WHERE source = 'linkedin' FOR UPDATE;
    SELECT * INTO grant_row FROM public.linkedin_source_request_grants
    WHERE id = p_grant_id AND producer = p_producer FOR UPDATE;
    IF grant_row.id IS NULL OR grant_row.status <> 'pending' OR grant_row.expires_at <= started
       OR policy.circuit_state <> 'closed' OR grant_row.circuit_generation <> policy.circuit_generation THEN
        IF grant_row.id IS NOT NULL AND grant_row.status = 'pending' THEN
            UPDATE public.linkedin_source_request_grants SET status = 'invalidated', finished_at = started WHERE id = p_grant_id;
        END IF;
        RETURN pg_catalog.jsonb_build_object('consumed', false, 'reason', CASE WHEN policy.circuit_state = 'open' THEN 'circuit_open' ELSE 'invalid' END);
    END IF;
    UPDATE public.linkedin_source_request_grants SET status = 'consumed', started_at = started WHERE id = p_grant_id;
    UPDATE public.linkedin_source_request_policy
    SET next_allowed_at = started + pg_catalog.make_interval(secs => minimum_interval_ms / 1000.0), updated_at = started
    WHERE source = 'linkedin';
    IF p_producer = 'adaptive-detail'
       AND grant_row.request_key ~ '^task:[0-9]+:[0-9a-f-]{36}:[^:]+:[0-9]+$' THEN
        INSERT INTO public.linkedin_discovery_task_attempts (
            task_id, lease_token, request_attempt, request_grant_id,
            response_class, parser_version, started_at
        ) VALUES (
            pg_catalog.split_part(grant_row.request_key, ':', 2)::bigint,
            pg_catalog.split_part(grant_row.request_key, ':', 3)::uuid,
            pg_catalog.split_part(grant_row.request_key, ':', 5)::integer,
            grant_row.id, 'started', 'linkedin-detail-v1', started
        ) ON CONFLICT (request_grant_id) DO NOTHING;
    END IF;
    RETURN pg_catalog.jsonb_build_object('consumed', true, 'started_at', started);
END;
$$;

CREATE OR REPLACE FUNCTION public.finish_linkedin_request_grant(
    p_grant_id uuid, p_producer text, p_response_class text, p_http_status integer
) RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    finished_grant public.linkedin_source_request_grants%ROWTYPE;
BEGIN
    UPDATE public.linkedin_source_request_grants
    SET status = 'finished', finished_at = pg_catalog.clock_timestamp(),
        response_class = p_response_class, http_status = p_http_status
    WHERE id = p_grant_id AND producer = p_producer AND status = 'consumed'
    RETURNING * INTO finished_grant;
    IF finished_grant.id IS NULL THEN
        RETURN false;
    END IF;
    IF p_producer = 'adaptive-detail'
       AND finished_grant.request_key ~ '^task:[0-9]+:[0-9a-f-]{36}:[^:]+:[0-9]+$' THEN
        INSERT INTO public.linkedin_discovery_task_attempts (
            task_id, lease_token, request_attempt, request_grant_id,
            response_class, http_status, parser_version, started_at, finished_at
        ) VALUES (
            pg_catalog.split_part(finished_grant.request_key, ':', 2)::bigint,
            pg_catalog.split_part(finished_grant.request_key, ':', 3)::uuid,
            pg_catalog.split_part(finished_grant.request_key, ':', 5)::integer,
            finished_grant.id, p_response_class, p_http_status,
            'linkedin-detail-v1', finished_grant.started_at, finished_grant.finished_at
        ) ON CONFLICT (request_grant_id) DO UPDATE SET
            response_class = EXCLUDED.response_class,
            http_status = EXCLUDED.http_status,
            parser_version = EXCLUDED.parser_version,
            finished_at = EXCLUDED.finished_at;
    END IF;
    RETURN true;
END;
$$;

CREATE OR REPLACE FUNCTION public.open_linkedin_source_circuit(
    p_grant_id uuid, p_producer text, p_reason text, p_http_status integer
) RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    now_at timestamptz := pg_catalog.clock_timestamp();
    policy public.linkedin_source_request_policy%ROWTYPE;
    grant_row public.linkedin_source_request_grants%ROWTYPE;
BEGIN
    SELECT * INTO STRICT policy FROM public.linkedin_source_request_policy
    WHERE source = 'linkedin' FOR UPDATE;
    SELECT * INTO grant_row FROM public.linkedin_source_request_grants
    WHERE id = p_grant_id AND producer = p_producer FOR UPDATE;
    IF grant_row.id IS NULL OR grant_row.status <> 'consumed'
       OR grant_row.circuit_generation <> policy.circuit_generation THEN
        RETURN false;
    END IF;
    UPDATE public.linkedin_source_request_policy
    SET circuit_state = 'open', circuit_generation = circuit_generation + 1,
        circuit_reason = p_reason, opened_at = now_at, open_until = now_at + interval '1 hour', updated_at = now_at
    WHERE source = 'linkedin';
    UPDATE public.linkedin_source_request_grants
    SET status = 'invalidated', finished_at = now_at,
        response_class = 'invalidated', http_status = NULL
    WHERE source = 'linkedin' AND status IN ('pending', 'consumed')
      AND id <> p_grant_id;
    UPDATE public.linkedin_source_request_grants
    SET status = 'invalidated', finished_at = now_at,
        response_class = 'challenge', http_status = p_http_status
    WHERE id = p_grant_id;
    UPDATE public.linkedin_discovery_task_attempts attempt
    SET response_class = 'invalidated', http_status = NULL, finished_at = now_at
    FROM public.linkedin_source_request_grants request_grant
    WHERE attempt.request_grant_id = request_grant.id
      AND request_grant.source = 'linkedin'
      AND request_grant.id <> p_grant_id
      AND attempt.finished_at IS NULL;
    IF p_producer = 'adaptive-detail'
       AND grant_row.request_key ~ '^task:[0-9]+:[0-9a-f-]{36}:[^:]+:[0-9]+$' THEN
        INSERT INTO public.linkedin_discovery_task_attempts (
            task_id, lease_token, request_attempt, request_grant_id,
            response_class, http_status, parser_version, started_at, finished_at
        ) VALUES (
            pg_catalog.split_part(grant_row.request_key, ':', 2)::bigint,
            pg_catalog.split_part(grant_row.request_key, ':', 3)::uuid,
            pg_catalog.split_part(grant_row.request_key, ':', 5)::integer,
            grant_row.id, 'challenge', p_http_status, 'linkedin-detail-v1',
            grant_row.started_at, now_at
        ) ON CONFLICT (request_grant_id) DO UPDATE SET
            response_class = EXCLUDED.response_class,
            http_status = EXCLUDED.http_status,
            parser_version = EXCLUDED.parser_version,
            finished_at = EXCLUDED.finished_at;
    END IF;
    RETURN true;
END;
$$;

CREATE OR REPLACE FUNCTION public.reset_linkedin_source_circuit(
    p_actor text, p_reason text
) RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE reset_at timestamptz := pg_catalog.clock_timestamp();
BEGIN
    IF p_actor IS NULL OR pg_catalog.btrim(p_actor) = ''
       OR p_reason IS NULL OR pg_catalog.btrim(p_reason) = '' THEN
        RAISE EXCEPTION 'circuit reset requires actor and reason' USING ERRCODE = '22023';
    END IF;
    UPDATE public.linkedin_source_request_policy
    SET circuit_state = 'closed', circuit_reason = NULL, opened_at = NULL,
        open_until = NULL, next_allowed_at = reset_at, updated_at = reset_at,
        last_reset_at = reset_at, last_reset_actor = p_actor,
        last_reset_reason = p_reason
    WHERE source = 'linkedin' AND circuit_state = 'open';
    RETURN FOUND;
END;
$$;

CREATE OR REPLACE FUNCTION public.create_linkedin_discovery_cycle(
    p_execution_id uuid, p_config_revision bigint, p_config_content_hash text,
    p_user_agent text, p_scopes jsonb
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    cycle_id bigint;
    sequence_value bigint;
    existing_cycle public.linkedin_discovery_cycles%ROWTYPE;
    scope jsonb;
    run_id uuid;
    scope_rows jsonb := '[]'::jsonb;
BEGIN
    IF p_execution_id IS NULL OR p_config_content_hash !~ '^[0-9a-f]{64}$'
       OR p_user_agent IS NULL OR pg_catalog.btrim(p_user_agent) = ''
       OR pg_catalog.jsonb_typeof(p_scopes) <> 'array'
       OR pg_catalog.jsonb_array_length(p_scopes) = 0 THEN
        RAISE EXCEPTION 'p_scopes must be a non-empty array' USING ERRCODE = '22023';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('linkedin-discovery-sequence-v1', 0));
    SELECT * INTO existing_cycle FROM public.linkedin_discovery_cycles
    WHERE execution_id = p_execution_id FOR UPDATE;
    IF existing_cycle.id IS NOT NULL THEN
        IF existing_cycle.config_content_hash <> p_config_content_hash
           OR existing_cycle.pinned_user_agent <> p_user_agent
           OR existing_cycle.required_scope_count <> pg_catalog.jsonb_array_length(p_scopes)
           OR EXISTS (
                SELECT 1
                FROM pg_catalog.jsonb_array_elements(p_scopes) input_scope
                LEFT JOIN public.linkedin_discovery_cycle_scopes manifest
                  ON manifest.discovery_cycle_id = existing_cycle.id
                 AND manifest.scope_key = input_scope->>'scope_key'
                LEFT JOIN public.linkedin_scope_coverage_state state
                  ON state.scope_key = input_scope->>'scope_key'
                WHERE manifest.scope_key IS NULL
                   OR state.scope_key IS NULL
                   OR state.scope_definition_hash <> input_scope->>'scope_definition_hash'
           ) THEN
            RAISE EXCEPTION 'execution id conflicts with an existing discovery cycle' USING ERRCODE = '23000';
        END IF;
        SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
            'scope_key', manifest.scope_key,
            'ingestion_run_id', manifest.ingestion_run_id,
            'next_page', COALESCE(page.next_page, 1),
            'status', manifest.status,
            'query_scope', manifest.query_scope,
            'request_anchor_at', manifest.request_anchor_at,
            'source_window_earliest_at', manifest.source_window_earliest_at,
            'source_window_latest_at', manifest.source_window_latest_at,
            'truncated_window_earliest_at', manifest.truncated_window_earliest_at,
            'truncated_window_latest_at', manifest.truncated_window_latest_at,
            'expired_window_earliest_at', manifest.expired_window_earliest_at,
            'expired_window_latest_at', manifest.expired_window_latest_at,
            'minimum_pages', manifest.minimum_pages,
            'target_pages', manifest.target_pages,
            'committed_page_count', COALESCE(page.committed_page_count, 0),
            'latest_page_result', page.latest_page_result
        ) ORDER BY manifest.scope_key), '[]'::jsonb)
        INTO scope_rows
        FROM public.linkedin_discovery_cycle_scopes manifest
        LEFT JOIN (
            SELECT ingestion_run_id, MAX(page_number) + 1 AS next_page,
                   COUNT(*) AS committed_page_count,
                   (ARRAY_AGG(result ORDER BY page_number DESC))[1] AS latest_page_result
            FROM public.linkedin_ingestion_pages
            GROUP BY ingestion_run_id
        ) page ON page.ingestion_run_id = manifest.ingestion_run_id
        WHERE manifest.discovery_cycle_id = existing_cycle.id;
        RETURN pg_catalog.jsonb_build_object(
            'cycle_id', existing_cycle.id,
            'discovery_sequence', existing_cycle.discovery_sequence,
            'search_status', existing_cycle.search_status,
            'scopes', scope_rows,
            'replayed', true
        );
    END IF;
    SELECT COALESCE(MAX(discovery_sequence), 0) + 1 INTO sequence_value FROM public.linkedin_discovery_cycles;
    INSERT INTO public.linkedin_discovery_cycles (
        execution_id, discovery_sequence, config_revision, config_content_hash,
        required_scope_count, pinned_user_agent
    ) VALUES (
        p_execution_id, sequence_value, p_config_revision, p_config_content_hash,
        pg_catalog.jsonb_array_length(p_scopes), p_user_agent
    ) RETURNING id INTO cycle_id;
    FOR scope IN SELECT value FROM pg_catalog.jsonb_array_elements(p_scopes) LOOP
        run_id := extensions.gen_random_uuid();
        IF scope->>'scope_key' IS NULL OR scope->>'scope_definition_hash' !~ '^[0-9a-f]{64}$'
           OR scope->>'scope_key' <> 'linkedin:v1:' || (scope->>'scope_definition_hash') THEN
            RAISE EXCEPTION 'invalid scope identity' USING ERRCODE = '22023';
        END IF;
        IF EXISTS (
            SELECT 1 FROM public.linkedin_scope_coverage_state state
            WHERE state.scope_key = scope->>'scope_key'
              AND (state.scope_definition_hash <> scope->>'scope_definition_hash'
                   OR state.scope_definition <> scope->'scope_definition')
        ) THEN
            RAISE EXCEPTION 'scope identity collision' USING ERRCODE = '23000';
        END IF;
        INSERT INTO public.linkedin_scope_coverage_state (
            scope_key, scope_definition_hash, scope_definition, config_revision,
            config_content_hash, archetype, query_id, geography_id, recommended_pages
        ) VALUES (
            scope->>'scope_key', scope->>'scope_definition_hash', scope->'scope_definition',
            p_config_revision, p_config_content_hash, scope->>'archetype', scope->>'query_id',
            scope->>'geography_id', (scope->>'minimum_pages')::integer
        ) ON CONFLICT (scope_key) DO UPDATE SET
            config_revision = EXCLUDED.config_revision,
            config_content_hash = EXCLUDED.config_content_hash,
            updated_at = pg_catalog.clock_timestamp();
        INSERT INTO public.ingestion_runs (
            id, provider, search_query, archetype, filter_profile, query_scope, discovery_cycle_id
        ) VALUES (
            run_id, 'linkedin', scope->>'query', scope->>'archetype', (scope->>'archetype') || '_v1',
            scope->>'query_scope', cycle_id
        );
        INSERT INTO public.linkedin_discovery_cycle_scopes (
            discovery_cycle_id, scope_key, ingestion_run_id, query_scope, request_anchor_at,
            source_window_earliest_at, source_window_latest_at,
            truncated_window_earliest_at, truncated_window_latest_at,
            expired_window_earliest_at, expired_window_latest_at,
            minimum_pages, target_pages
        ) VALUES (
            cycle_id, scope->>'scope_key', run_id, scope->>'query_scope',
            (scope->>'request_anchor_at')::timestamptz,
            (scope->>'source_window_earliest_at')::timestamptz,
            (scope->>'source_window_latest_at')::timestamptz,
            NULLIF(scope->>'truncated_window_earliest_at', '')::timestamptz,
            NULLIF(scope->>'truncated_window_latest_at', '')::timestamptz,
            NULLIF(scope->>'expired_window_earliest_at', '')::timestamptz,
            NULLIF(scope->>'expired_window_latest_at', '')::timestamptz,
            (scope->>'minimum_pages')::integer, (scope->>'target_pages')::integer
        );
        scope_rows := scope_rows || pg_catalog.jsonb_build_array(pg_catalog.jsonb_build_object(
            'scope_key', scope->>'scope_key', 'ingestion_run_id', run_id, 'next_page', 1,
            'status', 'running', 'query_scope', scope->>'query_scope',
            'request_anchor_at', scope->>'request_anchor_at',
            'source_window_earliest_at', scope->>'source_window_earliest_at',
            'source_window_latest_at', scope->>'source_window_latest_at',
            'truncated_window_earliest_at', scope->>'truncated_window_earliest_at',
            'truncated_window_latest_at', scope->>'truncated_window_latest_at',
            'expired_window_earliest_at', scope->>'expired_window_earliest_at',
            'expired_window_latest_at', scope->>'expired_window_latest_at',
            'minimum_pages', (scope->>'minimum_pages')::integer,
            'target_pages', (scope->>'target_pages')::integer,
            'committed_page_count', 0, 'latest_page_result', NULL
        ));
    END LOOP;
    RETURN pg_catalog.jsonb_build_object(
        'cycle_id', cycle_id, 'discovery_sequence', sequence_value,
        'search_status', 'running', 'scopes', scope_rows, 'replayed', false
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.get_resumable_linkedin_discovery_cycle(
    p_partial boolean, p_scope_keys text[] DEFAULT NULL
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    cycle_row public.linkedin_discovery_cycles%ROWTYPE;
    scope_rows jsonb;
BEGIN
    SELECT cycle.* INTO cycle_row
    FROM public.linkedin_discovery_cycles cycle
    WHERE (
          cycle.search_status = 'running'
          OR (
              NOT p_partial
              AND cycle.search_status = 'sealed'
              AND cycle.canonical_status = 'pending'
          )
      )
      AND EXISTS (
          SELECT 1
          FROM public.linkedin_discovery_cycle_scopes scope
          WHERE scope.discovery_cycle_id = cycle.id
            AND COALESCE((scope.query_scope::jsonb->>'partial')::boolean, false) = p_partial
      )
      AND (
          NOT p_partial
          OR ARRAY(
              SELECT scope.scope_key
              FROM public.linkedin_discovery_cycle_scopes scope
              WHERE scope.discovery_cycle_id = cycle.id
              ORDER BY scope.scope_key
          ) = ARRAY(
              SELECT requested.scope_key
              FROM pg_catalog.unnest(p_scope_keys) AS requested(scope_key)
              ORDER BY requested.scope_key
          )
      )
    ORDER BY
        CASE WHEN cycle.search_status = 'running' THEN 0 ELSE 1 END,
        CASE WHEN cycle.search_status = 'running' THEN cycle.discovery_sequence END,
        CASE WHEN cycle.search_status = 'sealed' THEN cycle.discovery_sequence END DESC
    LIMIT 1;
    IF cycle_row.id IS NULL THEN
        RETURN NULL;
    END IF;
    SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
        'scope_key', manifest.scope_key,
        'scope_definition', state.scope_definition,
        'ingestion_run_id', manifest.ingestion_run_id,
        'next_page', COALESCE(page.next_page, 1),
        'status', manifest.status,
        'query_scope', manifest.query_scope,
        'request_anchor_at', manifest.request_anchor_at,
        'source_window_earliest_at', manifest.source_window_earliest_at,
        'source_window_latest_at', manifest.source_window_latest_at,
        'truncated_window_earliest_at', manifest.truncated_window_earliest_at,
        'truncated_window_latest_at', manifest.truncated_window_latest_at,
        'expired_window_earliest_at', manifest.expired_window_earliest_at,
        'expired_window_latest_at', manifest.expired_window_latest_at,
        'minimum_pages', manifest.minimum_pages,
        'target_pages', manifest.target_pages,
        'committed_page_count', COALESCE(page.committed_page_count, 0),
        'latest_page_result', page.latest_page_result
    ) ORDER BY manifest.scope_key), '[]'::jsonb)
    INTO scope_rows
    FROM public.linkedin_discovery_cycle_scopes manifest
    JOIN public.linkedin_scope_coverage_state state
      ON state.scope_key = manifest.scope_key
    LEFT JOIN (
        SELECT ingestion_run_id, MAX(page_number) + 1 AS next_page,
               COUNT(*) AS committed_page_count,
               (ARRAY_AGG(result ORDER BY page_number DESC))[1] AS latest_page_result
        FROM public.linkedin_ingestion_pages
        GROUP BY ingestion_run_id
    ) page ON page.ingestion_run_id = manifest.ingestion_run_id
    WHERE manifest.discovery_cycle_id = cycle_row.id;
    RETURN pg_catalog.jsonb_build_object(
        'cycle_id', cycle_row.id,
        'discovery_sequence', cycle_row.discovery_sequence,
        'search_status', cycle_row.search_status,
        'pinned_user_agent', cycle_row.pinned_user_agent,
        'scopes', scope_rows,
        'resumed', true
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.commit_linkedin_discovery_page(p_page jsonb)
RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    run_id uuid := (p_page->>'ingestion_run_id')::uuid;
    cycle_id bigint := (p_page->>'cycle_id')::bigint;
    page_no integer := (p_page->>'page_number')::integer;
    page_offset integer := (p_page->>'page_start')::integer;
    card jsonb;
    source_id text;
    task_id bigint;
    mapped_job text;
    new_scope_count integer := 0;
    new_cycle_count integer := 0;
    known_count integer := 0;
    task_count integer := 0;
    task_kind_value text;
    requirement_key_value text;
    prior_terminal_task_id bigint;
    existing_page public.linkedin_ingestion_pages%ROWTYPE;
    scope_row public.linkedin_discovery_cycle_scopes%ROWTYPE;
    cycle_row public.linkedin_discovery_cycles%ROWTYPE;
    expected_page integer;
    page_provenance jsonb;
    page_query_scope jsonb;
    page_membership_provenance jsonb;
    page_lane text;
    page_observed_at timestamptz;
BEGIN
    IF pg_catalog.jsonb_typeof(p_page) <> 'object'
       OR pg_catalog.jsonb_typeof(p_page->'cards') <> 'array'
       OR page_no IS NULL OR page_no < 1
       OR page_offset <> (page_no - 1) * 10
       OR p_page->>'kind' NOT IN ('cards', 'no_results')
       OR (p_page->>'kind' = 'no_results' AND pg_catalog.jsonb_array_length(p_page->'cards') <> 0)
       OR (p_page->>'kind' = 'cards' AND pg_catalog.jsonb_array_length(p_page->'cards') = 0)
       OR (p_page->>'elements')::integer < pg_catalog.jsonb_array_length(p_page->'cards') THEN
        RAISE EXCEPTION 'invalid discovery page payload' USING ERRCODE = '22023';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements(p_page->'cards') input_card
        GROUP BY pg_catalog.btrim(input_card->>'job_id')
        HAVING pg_catalog.btrim(input_card->>'job_id') = '' OR COUNT(*) > 1
    ) THEN
        RAISE EXCEPTION 'discovery page contains blank or duplicate source ids' USING ERRCODE = '22023';
    END IF;
    page_provenance := COALESCE(p_page->'provenance', '{}'::jsonb);
    IF pg_catalog.jsonb_typeof(page_provenance) <> 'object' THEN
        RAISE EXCEPTION 'discovery page provenance must be an object' USING ERRCODE = '22023';
    END IF;
    page_lane := pg_catalog.btrim(COALESCE(
        page_provenance->>'lane', page_provenance->>'archetype', ''
    ));
    IF page_lane = ''
       OR (page_provenance->>'lane' IS NOT NULL
           AND page_provenance->>'archetype' IS NOT NULL
           AND page_provenance->>'lane' IS DISTINCT FROM page_provenance->>'archetype')
       OR (page_provenance->>'search_query_type' IS NOT NULL
           AND page_provenance->>'search_query_type' NOT IN ('precision', 'recall'))
       OR (page_provenance->>'search_query_language' IS NOT NULL
           AND page_provenance->>'search_query_language' !~ '^[a-z]{2}(-[A-Z]{2})?$') THEN
        RAISE EXCEPTION 'discovery page has invalid lane provenance' USING ERRCODE = '22023';
    END IF;
    BEGIN
        page_query_scope := CASE
            WHEN page_provenance->'query_scope' IS NULL THEN '{}'::jsonb
            WHEN pg_catalog.jsonb_typeof(page_provenance->'query_scope') = 'object'
                THEN page_provenance->'query_scope'
            WHEN pg_catalog.jsonb_typeof(page_provenance->'query_scope') = 'string'
                THEN (page_provenance->>'query_scope')::jsonb
            ELSE NULL
        END;
    EXCEPTION WHEN invalid_text_representation THEN
        RAISE EXCEPTION 'discovery page query scope is invalid JSON' USING ERRCODE = '22023';
    END;
    IF pg_catalog.jsonb_typeof(page_query_scope) <> 'object' THEN
        RAISE EXCEPTION 'discovery page query scope must be an object' USING ERRCODE = '22023';
    END IF;
    page_observed_at := (p_page->>'requested_at')::timestamptz;
    page_membership_provenance := pg_catalog.jsonb_strip_nulls(
        page_query_scope || pg_catalog.jsonb_build_object(
            'lane', page_lane,
            'archetype', page_lane,
            'query_id', COALESCE(page_provenance->>'search_query_id', page_query_scope->>'query_id'),
            'query', COALESCE(page_provenance->>'search_query', page_query_scope->>'query', page_query_scope->>'search_query'),
            'query_type', COALESCE(page_provenance->>'search_query_type', page_query_scope->>'query_type', page_query_scope->>'query_kind'),
             'language', COALESCE(page_provenance->>'search_query_language', page_query_scope->>'language'),
             'location_scope', COALESCE(page_provenance->>'search_location_scope', page_query_scope->>'location_scope'),
             'geography_id', COALESCE(page_provenance->>'geography_id', page_query_scope->>'geography_id'),
             'observed_at', pg_catalog.to_jsonb(page_observed_at)
         )
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('linkedin-canonical-publication-v1', 0)
    );
    SELECT * INTO STRICT cycle_row FROM public.linkedin_discovery_cycles
    WHERE id = cycle_id FOR UPDATE;
    IF cycle_row.search_status <> 'running' THEN
        RAISE EXCEPTION 'discovery cycle is not running' USING ERRCODE = '55000';
    END IF;
    SELECT * INTO STRICT scope_row
    FROM public.linkedin_discovery_cycle_scopes
    WHERE discovery_cycle_id = cycle_id AND ingestion_run_id = run_id
      AND scope_key = p_page->>'scope_key' FOR UPDATE;
    IF scope_row.status <> 'running' THEN
        RAISE EXCEPTION 'discovery scope is not running' USING ERRCODE = '55000';
    END IF;
    IF (p_page->>'source_window_earliest_at')::timestamptz > scope_row.source_window_earliest_at
       OR (p_page->>'source_window_latest_at')::timestamptz < scope_row.source_window_latest_at THEN
        RAISE EXCEPTION 'page window does not contain manifest window' USING ERRCODE = '22023';
    END IF;
    SELECT * INTO existing_page FROM public.linkedin_ingestion_pages
    WHERE ingestion_run_id = run_id AND page_number = page_no FOR UPDATE;
    IF existing_page.ingestion_run_id IS NOT NULL THEN
        IF existing_page.page_start IS DISTINCT FROM page_offset
           OR existing_page.requested_at IS DISTINCT FROM (p_page->>'requested_at')::timestamptz
           OR existing_page.source_window_earliest_at IS DISTINCT FROM (p_page->>'source_window_earliest_at')::timestamptz
           OR existing_page.source_window_latest_at IS DISTINCT FROM (p_page->>'source_window_latest_at')::timestamptz
           OR existing_page.element_count IS DISTINCT FROM (p_page->>'elements')::integer
           OR existing_page.card_count IS DISTINCT FROM pg_catalog.jsonb_array_length(p_page->'cards')
           OR existing_page.result IS DISTINCT FROM p_page->>'kind'
           OR existing_page.request_attempts IS DISTINCT FROM (p_page->>'request_attempts')::integer
           OR existing_page.elapsed_ms IS DISTINCT FROM (p_page->>'elapsed_ms')::integer
           OR existing_page.classifier_version IS DISTINCT FROM p_page->>'classifier_version'
           OR existing_page.response_fingerprint IS DISTINCT FROM p_page->>'response_fingerprint'
           OR existing_page.membership_fingerprint IS DISTINCT FROM p_page->>'membership_fingerprint'
           OR EXISTS (
                (SELECT pg_catalog.btrim(input_card->>'job_id'),
                        (input_card->>'position_on_page')::integer,
                        (input_card->>'position_in_scope')::integer
                 FROM pg_catalog.jsonb_array_elements(p_page->'cards') input_card)
                EXCEPT
                (SELECT source_job_id, position_on_page, position_in_scope
                 FROM public.linkedin_ingestion_page_sources
                 WHERE ingestion_run_id = run_id AND page_number = page_no)
           ) OR EXISTS (
                (SELECT source_job_id, position_on_page, position_in_scope
                 FROM public.linkedin_ingestion_page_sources
                 WHERE ingestion_run_id = run_id AND page_number = page_no)
                EXCEPT
                (SELECT pg_catalog.btrim(input_card->>'job_id'),
                        (input_card->>'position_on_page')::integer,
                        (input_card->>'position_in_scope')::integer
                 FROM pg_catalog.jsonb_array_elements(p_page->'cards') input_card)
           ) THEN
            RAISE EXCEPTION 'conflicting replay for ingestion page' USING ERRCODE = '23000';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'cards', existing_page.card_count, 'new_source_ids', existing_page.new_source_ids,
            'new_workflow_source_ids', existing_page.new_workflow_source_ids,
            'known_source_ids', existing_page.known_source_ids, 'tasks_created', 0, 'replayed', true
        );
    END IF;
    SELECT COALESCE(MAX(page_number), 0) + 1 INTO expected_page
    FROM public.linkedin_ingestion_pages WHERE ingestion_run_id = run_id;
    IF page_no <> expected_page THEN
        RAISE EXCEPTION 'discovery pages must commit as a contiguous prefix' USING ERRCODE = '22023';
    END IF;
    INSERT INTO public.linkedin_ingestion_pages (
        ingestion_run_id, page_number, page_start, requested_at,
        source_window_earliest_at, source_window_latest_at, element_count, card_count,
        result, request_attempts, elapsed_ms, classifier_version,
        response_fingerprint, membership_fingerprint
    ) VALUES (
        run_id, page_no, page_offset, (p_page->>'requested_at')::timestamptz,
        (p_page->>'source_window_earliest_at')::timestamptz,
        (p_page->>'source_window_latest_at')::timestamptz,
        (p_page->>'elements')::integer, pg_catalog.jsonb_array_length(p_page->'cards'),
        p_page->>'kind', (p_page->>'request_attempts')::integer,
        (p_page->>'elapsed_ms')::integer, p_page->>'classifier_version',
        p_page->>'response_fingerprint', p_page->>'membership_fingerprint'
    );
    FOR card IN SELECT value FROM pg_catalog.jsonb_array_elements(p_page->'cards') LOOP
        source_id := pg_catalog.btrim(card->>'job_id');
        IF source_id = '' THEN CONTINUE; END IF;
        IF NOT EXISTS (
            SELECT 1 FROM public.linkedin_ingestion_page_sources
            WHERE ingestion_run_id = run_id AND provider = 'linkedin' AND source_job_id = source_id
        ) THEN new_scope_count := new_scope_count + 1; END IF;
        INSERT INTO public.linkedin_ingestion_page_sources (
            ingestion_run_id, page_number, source_job_id, position_on_page, position_in_scope
        ) VALUES (
            run_id, page_no, source_id, (card->>'position_on_page')::integer,
            (card->>'position_in_scope')::integer
        );
        INSERT INTO public.linkedin_discovery_cycle_sources (
            discovery_cycle_id, source_job_id, first_ingestion_run_id, first_page_number, first_position_on_page
        ) VALUES (cycle_id, source_id, run_id, page_no, (card->>'position_on_page')::integer)
        ON CONFLICT DO NOTHING;
        IF FOUND THEN new_cycle_count := new_cycle_count + 1; END IF;
        SELECT canonical_job_id INTO mapped_job FROM public.listing_states
        WHERE provider = 'linkedin' AND source_job_id = source_id;
        IF mapped_job IS NOT NULL THEN known_count := known_count + 1; END IF;
        INSERT INTO public.listing_observations (
            provider, source_job_id, canonical_job_id, ingestion_run_id, posted_at,
            posted_relative_text, location, card_label, query_scope,
            page_number, page_start, position_on_page, position_in_scope
        ) VALUES (
            'linkedin', source_id, mapped_job, run_id, NULLIF(card->>'posted_at', '')::date,
            card->>'posted_relative_text', card->>'location', card->>'card_label', scope_row.query_scope,
            page_no, page_offset, (card->>'position_on_page')::integer, (card->>'position_in_scope')::integer
        ) ON CONFLICT DO NOTHING;
        INSERT INTO public.listing_states (
            provider, source_job_id, canonical_job_id, first_seen_at, last_seen_at, latest_trusted_posted_date
        ) VALUES (
            'linkedin', source_id, mapped_job, pg_catalog.clock_timestamp(), pg_catalog.clock_timestamp(),
            NULLIF(card->>'posted_at', '')::date
        ) ON CONFLICT (provider, source_job_id) DO UPDATE SET
            last_seen_at = GREATEST(public.listing_states.last_seen_at, EXCLUDED.last_seen_at),
            latest_trusted_posted_date = COALESCE(
                GREATEST(public.listing_states.latest_trusted_posted_date, EXCLUDED.latest_trusted_posted_date),
                public.listing_states.latest_trusted_posted_date,
                EXCLUDED.latest_trusted_posted_date
            ),
            canonical_job_id = COALESCE(public.listing_states.canonical_job_id, EXCLUDED.canonical_job_id),
            updated_at = pg_catalog.clock_timestamp();
        IF mapped_job IS NOT NULL THEN
            INSERT INTO public.job_archetype_memberships AS membership_row (
                job_id, archetype, matched_queries, first_matched_at, last_matched_at, insights
            ) VALUES (
                mapped_job, page_lane, pg_catalog.jsonb_build_array(page_membership_provenance),
                page_observed_at, page_observed_at,
                pg_catalog.jsonb_build_object(
                    'matched_queries', pg_catalog.jsonb_build_array(page_membership_provenance),
                    'matched_query_provenance', pg_catalog.jsonb_build_array(page_membership_provenance),
                    'query_scopes', pg_catalog.jsonb_build_array(page_membership_provenance),
                    'last_matched_at', pg_catalog.to_jsonb(page_observed_at)
                )
            ) ON CONFLICT (job_id, archetype) DO UPDATE SET
                matched_queries = (
                    SELECT COALESCE(pg_catalog.jsonb_agg(item.value ORDER BY item.value::text), '[]'::jsonb)
                    FROM (
                        SELECT DISTINCT value
                        FROM pg_catalog.jsonb_array_elements(
                            membership_row.matched_queries || EXCLUDED.matched_queries
                        )
                    ) AS item
                ),
                first_matched_at = LEAST(membership_row.first_matched_at, EXCLUDED.first_matched_at),
                last_matched_at = GREATEST(membership_row.last_matched_at, EXCLUDED.last_matched_at),
                insights = membership_row.insights || pg_catalog.jsonb_build_object(
                    'matched_queries', (
                        SELECT COALESCE(pg_catalog.jsonb_agg(item.value ORDER BY item.value::text), '[]'::jsonb)
                        FROM (
                            SELECT DISTINCT value
                            FROM pg_catalog.jsonb_array_elements(
                                membership_row.matched_queries || EXCLUDED.matched_queries
                            )
                        ) AS item
                    ),
                    'matched_query_provenance', (
                        SELECT COALESCE(pg_catalog.jsonb_agg(item.value ORDER BY item.value::text), '[]'::jsonb)
                        FROM (
                            SELECT DISTINCT value
                            FROM pg_catalog.jsonb_array_elements(
                                membership_row.matched_queries || EXCLUDED.matched_queries
                            )
                        ) AS item
                    ),
                    'query_scopes', (
                        SELECT COALESCE(pg_catalog.jsonb_agg(item.value ORDER BY item.value::text), '[]'::jsonb)
                        FROM (
                            SELECT DISTINCT value
                            FROM pg_catalog.jsonb_array_elements(
                                membership_row.matched_queries || EXCLUDED.matched_queries
                            )
                        ) AS item
                    ),
                    'last_matched_at', pg_catalog.to_jsonb(
                        GREATEST(membership_row.last_matched_at, EXCLUDED.last_matched_at)
                    )
                ),
                updated_at = pg_catalog.clock_timestamp();
        END IF;
        SELECT id INTO prior_terminal_task_id
        FROM public.linkedin_discovery_tasks
        WHERE provider = 'linkedin' AND source_job_id = source_id
          AND task_kind = 'initial_detail' AND requirement_key = 'first'
          AND status = 'terminal_unavailable';
        IF prior_terminal_task_id IS NOT NULL AND mapped_job IS NULL THEN
            task_kind_value := 'availability_revalidation';
            requirement_key_value := prior_terminal_task_id::text || ':' || cycle_id::text;
        ELSE
            task_kind_value := 'initial_detail';
            requirement_key_value := 'first';
        END IF;
         INSERT INTO public.linkedin_discovery_tasks (
             source_job_id, task_kind, requirement_key, first_ingestion_run_id, first_query_scope,
             first_observed_at, latest_observed_at, posted_at, search_card, provenance,
             membership_provenances,
             status, completed_at, canonical_job_id
        ) VALUES (
            source_id, task_kind_value, requirement_key_value, run_id,
            p_page->>'scope_key', pg_catalog.clock_timestamp(),
             pg_catalog.clock_timestamp(), NULLIF(card->>'posted_at', '')::date, card,
             COALESCE(p_page->'provenance', '{}'::jsonb),
             pg_catalog.jsonb_build_array(page_membership_provenance),
             CASE WHEN mapped_job IS NULL THEN 'pending' ELSE 'complete' END,
            CASE WHEN mapped_job IS NULL THEN NULL ELSE pg_catalog.clock_timestamp() END, mapped_job
        ) ON CONFLICT (provider, source_job_id, task_kind, requirement_key) DO UPDATE SET
            latest_observed_at = GREATEST(public.linkedin_discovery_tasks.latest_observed_at, EXCLUDED.latest_observed_at),
            search_card = CASE WHEN EXCLUDED.latest_observed_at >= public.linkedin_discovery_tasks.latest_observed_at THEN EXCLUDED.search_card ELSE public.linkedin_discovery_tasks.search_card END,
             posted_at = CASE WHEN EXCLUDED.latest_observed_at >= public.linkedin_discovery_tasks.latest_observed_at THEN EXCLUDED.posted_at ELSE public.linkedin_discovery_tasks.posted_at END,
             provenance = public.linkedin_discovery_tasks.provenance || EXCLUDED.provenance,
             membership_provenance_revision = public.linkedin_discovery_tasks.membership_provenance_revision
                 + CASE
                     WHEN EXCLUDED.membership_provenances <@ public.linkedin_discovery_tasks.membership_provenances
                     THEN 0 ELSE 1
                   END,
             membership_provenances = (
                 SELECT COALESCE(pg_catalog.jsonb_agg(item.value ORDER BY item.value::text), '[]'::jsonb)
                 FROM (
                     SELECT DISTINCT value
                     FROM pg_catalog.jsonb_array_elements(
                         public.linkedin_discovery_tasks.membership_provenances
                         || EXCLUDED.membership_provenances
                     )
                 ) AS item
             ),
             canonical_job_id = COALESCE(public.linkedin_discovery_tasks.canonical_job_id, EXCLUDED.canonical_job_id),
            status = CASE WHEN public.linkedin_discovery_tasks.status IN ('complete', 'terminal_unavailable', 'leased') THEN public.linkedin_discovery_tasks.status WHEN EXCLUDED.canonical_job_id IS NOT NULL THEN 'complete' ELSE public.linkedin_discovery_tasks.status END,
            completed_at = CASE WHEN public.linkedin_discovery_tasks.completed_at IS NOT NULL THEN public.linkedin_discovery_tasks.completed_at WHEN EXCLUDED.canonical_job_id IS NOT NULL THEN pg_catalog.clock_timestamp() ELSE NULL END
        RETURNING id INTO task_id;
         INSERT INTO public.linkedin_discovery_requirements (
             discovery_cycle_id, ingestion_run_id, provider, source_job_id,
             task_kind, requirement_key, task_id, membership_provenance
         ) VALUES (
             cycle_id, run_id, 'linkedin', source_id,
             task_kind_value, requirement_key_value, task_id, page_membership_provenance
         )
        ON CONFLICT DO NOTHING;
        task_count := task_count + 1;
    END LOOP;
    UPDATE public.linkedin_ingestion_pages SET
        new_source_ids = new_scope_count, new_workflow_source_ids = new_cycle_count,
        known_source_ids = known_count
    WHERE ingestion_run_id = run_id AND page_number = page_no;
    RETURN pg_catalog.jsonb_build_object(
        'cards', pg_catalog.jsonb_array_length(p_page->'cards'), 'new_source_ids', new_scope_count,
        'new_workflow_source_ids', new_cycle_count, 'known_source_ids', known_count,
        'tasks_created', task_count, 'replayed', false
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.finish_linkedin_discovery_scope(
    p_ingestion_run_id uuid, p_coverage_status text, p_coverage_reason text
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    scope_row public.linkedin_discovery_cycle_scopes%ROWTYPE;
    cycle_row public.linkedin_discovery_cycles%ROWTYPE;
    page_count integer;
    total_cards integer;
    last_result text;
    tail_workflow_ids integer;
    run_status text;
    existing_coverage_status text;
    existing_coverage_reason text;
BEGIN
    IF p_coverage_status NOT IN ('exhausted', 'right_censored') THEN
        RAISE EXCEPTION 'invalid successful coverage status' USING ERRCODE = '22023';
    END IF;
    SELECT cycle.* INTO STRICT cycle_row
    FROM public.linkedin_discovery_cycles cycle
    JOIN public.linkedin_discovery_cycle_scopes scope
      ON scope.discovery_cycle_id = cycle.id
    WHERE scope.ingestion_run_id = p_ingestion_run_id FOR UPDATE OF cycle;
    SELECT * INTO STRICT scope_row FROM public.linkedin_discovery_cycle_scopes
    WHERE ingestion_run_id = p_ingestion_run_id FOR UPDATE;
    SELECT status, coverage_status, coverage_reason
    INTO STRICT run_status, existing_coverage_status, existing_coverage_reason
    FROM public.ingestion_runs WHERE id = p_ingestion_run_id FOR UPDATE;
    IF scope_row.status = 'complete' AND run_status = 'complete' THEN
        IF existing_coverage_status IS DISTINCT FROM p_coverage_status
           OR existing_coverage_reason IS DISTINCT FROM p_coverage_reason THEN
            RAISE EXCEPTION 'conflicting replay for completed discovery scope' USING ERRCODE = '23000';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'ingestion_run_id', p_ingestion_run_id,
            'coverage_status', existing_coverage_status,
            'replayed', true
        );
    END IF;
    IF cycle_row.search_status <> 'running' OR scope_row.status <> 'running' THEN
        RAISE EXCEPTION 'discovery scope is not running' USING ERRCODE = '55000';
    END IF;
    SELECT COUNT(*), COALESCE(SUM(page.card_count), 0),
           (ARRAY_AGG(page.result ORDER BY page.page_number DESC))[1],
           (ARRAY_AGG(page.new_workflow_source_ids ORDER BY page.page_number DESC))[1]
    INTO page_count, total_cards, last_result, tail_workflow_ids
    FROM public.linkedin_ingestion_pages page
    WHERE page.ingestion_run_id = p_ingestion_run_id;
    IF page_count = 0
       OR page_count <> (SELECT MAX(page.page_number) FROM public.linkedin_ingestion_pages page WHERE page.ingestion_run_id = p_ingestion_run_id)
       OR EXISTS (
            SELECT 1 FROM public.linkedin_ingestion_pages page
            WHERE page.ingestion_run_id = p_ingestion_run_id
              AND (page.source_window_earliest_at > scope_row.source_window_earliest_at
                   OR page.source_window_latest_at < scope_row.source_window_latest_at)
       )
       OR EXISTS (
            SELECT 1
            FROM public.linkedin_ingestion_pages page
            LEFT JOIN (
                SELECT ingestion_run_id, page_number, COUNT(*) AS members
                FROM public.linkedin_ingestion_page_sources
                GROUP BY ingestion_run_id, page_number
            ) membership USING (ingestion_run_id, page_number)
            WHERE page.ingestion_run_id = p_ingestion_run_id
              AND page.card_count <> COALESCE(membership.members, 0)
       )
       OR EXISTS (
            SELECT 1
            FROM public.linkedin_ingestion_page_sources source
            LEFT JOIN public.linkedin_discovery_requirements requirement
              ON requirement.discovery_cycle_id = scope_row.discovery_cycle_id
             AND requirement.ingestion_run_id = source.ingestion_run_id
             AND requirement.provider = source.provider
             AND requirement.source_job_id = source.source_job_id
             AND requirement.required
            WHERE source.ingestion_run_id = p_ingestion_run_id
              AND requirement.task_id IS NULL
       ) THEN
        RAISE EXCEPTION 'discovery scope has incomplete durable page evidence' USING ERRCODE = '55000';
    END IF;
    IF (p_coverage_status = 'exhausted' AND last_result <> 'no_results')
       OR (p_coverage_status = 'right_censored' AND page_count < scope_row.minimum_pages) THEN
        RAISE EXCEPTION 'coverage status is not supported by page evidence' USING ERRCODE = '22023';
    END IF;
    UPDATE public.ingestion_runs SET status = 'complete', finished_at = pg_catalog.clock_timestamp(),
        coverage_status = p_coverage_status, coverage_reason = p_coverage_reason,
        pages_attempted = page_count, pages_completed = page_count, cards_seen = total_cards
    WHERE id = p_ingestion_run_id AND status = 'running';
    IF NOT FOUND THEN
        RAISE EXCEPTION 'ingestion run is not running' USING ERRCODE = '55000';
    END IF;
    UPDATE public.linkedin_discovery_cycle_scopes SET status = 'complete', enqueue_committed_at = pg_catalog.clock_timestamp()
    WHERE ingestion_run_id = p_ingestion_run_id;
    IF scope_row.truncated_window_earliest_at IS NOT NULL THEN
        INSERT INTO public.linkedin_coverage_debt (
            scope_key, origin_discovery_cycle_id, origin_ingestion_run_id, debt_kind,
            source_window_earliest_at, source_window_latest_at, page_cap
        ) VALUES (
            scope_row.scope_key, scope_row.discovery_cycle_id, p_ingestion_run_id,
            'lookback_truncated', scope_row.truncated_window_earliest_at,
            scope_row.truncated_window_latest_at, scope_row.target_pages
        ) ON CONFLICT DO NOTHING;
    END IF;
    IF scope_row.expired_window_earliest_at IS NOT NULL THEN
        INSERT INTO public.linkedin_coverage_debt (
            scope_key, origin_discovery_cycle_id, origin_ingestion_run_id, debt_kind,
            source_window_earliest_at, source_window_latest_at, page_cap,
            status, resolved_at, resolution
        ) VALUES (
            scope_row.scope_key, scope_row.discovery_cycle_id, p_ingestion_run_id,
            'lookback_truncated', scope_row.expired_window_earliest_at,
            scope_row.expired_window_latest_at, scope_row.target_pages,
            'expired_unresolved', pg_catalog.clock_timestamp(),
            'outside configured outage recovery cap'
        ) ON CONFLICT DO NOTHING;
    END IF;
    INSERT INTO public.linkedin_coverage_debt_attempts (
        debt_id, recovery_ingestion_run_id, recovery_discovery_cycle_id,
        requested_window_earliest_at, requested_window_latest_at,
        requested_page_cap, outcome
    )
    SELECT debt.id, p_ingestion_run_id, scope_row.discovery_cycle_id,
           scope_row.source_window_earliest_at, scope_row.source_window_latest_at,
           scope_row.target_pages,
           CASE
               WHEN p_coverage_status = 'exhausted'
                    AND debt.source_window_earliest_at >= scope_row.source_window_earliest_at
                    AND debt.source_window_latest_at <= scope_row.source_window_latest_at
               THEN 'resolved'
               WHEN debt.source_window_latest_at < scope_row.source_window_earliest_at
                    OR debt.source_window_earliest_at > scope_row.source_window_latest_at
               THEN 'not_contained'
               ELSE 'right_censored'
           END
    FROM public.linkedin_coverage_debt debt
    JOIN public.linkedin_discovery_cycles origin
      ON origin.id = debt.origin_discovery_cycle_id
    WHERE debt.scope_key = scope_row.scope_key AND debt.status = 'pending'
      AND origin.discovery_sequence < cycle_row.discovery_sequence
    ON CONFLICT DO NOTHING;
    IF p_coverage_status = 'right_censored' THEN
        INSERT INTO public.linkedin_coverage_debt (
            scope_key, origin_discovery_cycle_id, origin_ingestion_run_id, debt_kind,
            source_window_earliest_at, source_window_latest_at, page_cap
        ) VALUES (
            scope_row.scope_key, scope_row.discovery_cycle_id, p_ingestion_run_id, 'search_right_censored',
            scope_row.source_window_earliest_at, scope_row.source_window_latest_at, GREATEST(page_count, 1)
        ) ON CONFLICT DO NOTHING;
        UPDATE public.linkedin_scope_coverage_state SET coverage_debt = true,
            coverage_debt_since = COALESCE(coverage_debt_since, pg_catalog.clock_timestamp()),
            last_saturated_at = pg_catalog.clock_timestamp(),
            last_deep_sweep_at = CASE
                WHEN page_count > scope_row.minimum_pages
                     AND scope_row.target_pages > scope_row.minimum_pages
                THEN pg_catalog.clock_timestamp()
                ELSE public.linkedin_scope_coverage_state.last_deep_sweep_at
            END,
            consecutive_saturated_runs = consecutive_saturated_runs + 1,
            recommended_pages = LEAST(100, GREATEST(recommended_pages, page_count + CASE WHEN tail_workflow_ids > 0 THEN 2 ELSE 0 END)),
            latest_tail_workflow_new_ids = COALESCE(tail_workflow_ids, 0),
            updated_at = pg_catalog.clock_timestamp()
        WHERE scope_key = scope_row.scope_key;
    ELSE
        UPDATE public.linkedin_coverage_debt
        SET status = 'resolved', resolved_at = pg_catalog.clock_timestamp(),
            resolution = 'covered by exhausted ingestion run ' || p_ingestion_run_id::text,
            resolved_by_discovery_cycle_id = scope_row.discovery_cycle_id,
            resolved_by_ingestion_run_id = p_ingestion_run_id
        WHERE scope_key = scope_row.scope_key AND status = 'pending'
          AND source_window_earliest_at >= scope_row.source_window_earliest_at
          AND source_window_latest_at <= scope_row.source_window_latest_at
          AND EXISTS (
              SELECT 1 FROM public.linkedin_discovery_cycles origin
              WHERE origin.id = public.linkedin_coverage_debt.origin_discovery_cycle_id
                AND origin.discovery_sequence < cycle_row.discovery_sequence
          );
        UPDATE public.linkedin_scope_coverage_state SET last_exhausted_at = pg_catalog.clock_timestamp(),
            consecutive_saturated_runs = 0,
            latest_tail_workflow_new_ids = COALESCE(tail_workflow_ids, 0),
            coverage_debt = EXISTS (
                SELECT 1 FROM public.linkedin_coverage_debt debt
                WHERE debt.scope_key = scope_row.scope_key
                  AND debt.status IN ('pending', 'expired_unresolved')
            ),
            coverage_debt_since = (
                SELECT MIN(debt.created_at) FROM public.linkedin_coverage_debt debt
                WHERE debt.scope_key = scope_row.scope_key
                  AND debt.status IN ('pending', 'expired_unresolved')
            ),
            updated_at = pg_catalog.clock_timestamp()
        WHERE scope_key = scope_row.scope_key;
    END IF;
    RETURN pg_catalog.jsonb_build_object('ingestion_run_id', p_ingestion_run_id, 'coverage_status', p_coverage_status);
END;
$$;

CREATE OR REPLACE FUNCTION public.fail_linkedin_discovery_cycle(p_cycle_id bigint, p_reason text)
RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    cycle_row public.linkedin_discovery_cycles%ROWTYPE;
    scope_row public.linkedin_discovery_cycle_scopes%ROWTYPE;
    page_count integer;
BEGIN
    SELECT * INTO STRICT cycle_row FROM public.linkedin_discovery_cycles
    WHERE id = p_cycle_id FOR UPDATE;
    IF cycle_row.search_status = 'sealed' THEN
        RETURN false;
    ELSIF cycle_row.search_status = 'failed' THEN
        RETURN true;
    END IF;
    UPDATE public.linkedin_discovery_cycles SET search_status = 'failed', search_completed_at = pg_catalog.clock_timestamp(), failure_reason = p_reason
    WHERE id = p_cycle_id AND search_status = 'running';
    FOR scope_row IN
        SELECT * FROM public.linkedin_discovery_cycle_scopes
        WHERE discovery_cycle_id = p_cycle_id AND status = 'running'
        ORDER BY scope_key FOR UPDATE
    LOOP
        SELECT COUNT(*) INTO page_count FROM public.linkedin_ingestion_pages WHERE ingestion_run_id = scope_row.ingestion_run_id;
        UPDATE public.linkedin_discovery_cycle_scopes SET status = 'failed' WHERE ingestion_run_id = scope_row.ingestion_run_id;
        UPDATE public.ingestion_runs SET status = 'failed', coverage_status = 'failed', failure_code = 'cycle_failed',
            coverage_reason = p_reason, finished_at = pg_catalog.clock_timestamp(), pages_attempted = page_count, pages_completed = page_count
        WHERE id = scope_row.ingestion_run_id;
        INSERT INTO public.linkedin_coverage_debt_attempts (
            debt_id, recovery_ingestion_run_id, recovery_discovery_cycle_id,
            requested_window_earliest_at, requested_window_latest_at,
            requested_page_cap, outcome
        )
        SELECT debt.id, scope_row.ingestion_run_id, p_cycle_id,
               scope_row.source_window_earliest_at, scope_row.source_window_latest_at,
               scope_row.target_pages, 'failed'
        FROM public.linkedin_coverage_debt debt
        JOIN public.linkedin_discovery_cycles origin
          ON origin.id = debt.origin_discovery_cycle_id
        WHERE debt.scope_key = scope_row.scope_key AND debt.status = 'pending'
          AND origin.discovery_sequence < cycle_row.discovery_sequence
        ON CONFLICT DO NOTHING;
        INSERT INTO public.linkedin_coverage_debt (
            scope_key, origin_discovery_cycle_id, origin_ingestion_run_id, debt_kind,
            source_window_earliest_at, source_window_latest_at, page_cap
        ) VALUES (
            scope_row.scope_key, p_cycle_id, scope_row.ingestion_run_id,
            CASE WHEN page_count = 0 THEN 'scope_unattempted_after_cycle_failure' ELSE 'search_failed' END,
            scope_row.source_window_earliest_at, scope_row.source_window_latest_at, GREATEST(page_count, 1)
        ) ON CONFLICT DO NOTHING;
        UPDATE public.linkedin_scope_coverage_state
        SET coverage_debt = true,
            coverage_debt_since = COALESCE(coverage_debt_since, pg_catalog.clock_timestamp()),
            updated_at = pg_catalog.clock_timestamp()
        WHERE scope_key = scope_row.scope_key;
    END LOOP;
    RETURN true;
END;
$$;

CREATE OR REPLACE FUNCTION public.expire_linkedin_coverage_debt(
    p_scope_key text, p_recovery_floor timestamptz
) RETURNS integer
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    debt_row public.linkedin_coverage_debt%ROWTYPE;
    expired_count integer := 0;
BEGIN
    IF p_scope_key IS NULL OR pg_catalog.btrim(p_scope_key) = ''
       OR p_recovery_floor IS NULL THEN
        RAISE EXCEPTION 'invalid debt expiry parameters' USING ERRCODE = '22023';
    END IF;
    FOR debt_row IN
        SELECT * FROM public.linkedin_coverage_debt debt
        WHERE debt.scope_key = p_scope_key AND debt.status = 'pending'
          AND debt.source_window_earliest_at < p_recovery_floor
        ORDER BY debt.id FOR UPDATE
    LOOP
        IF debt_row.source_window_latest_at > p_recovery_floor THEN
            INSERT INTO public.linkedin_coverage_debt (
                scope_key, origin_discovery_cycle_id, origin_ingestion_run_id,
                debt_kind, source_window_earliest_at, source_window_latest_at,
                page_cap
            ) VALUES (
                debt_row.scope_key, debt_row.origin_discovery_cycle_id,
                debt_row.origin_ingestion_run_id, debt_row.debt_kind,
                p_recovery_floor, debt_row.source_window_latest_at,
                debt_row.page_cap
            ) ON CONFLICT DO NOTHING;
        END IF;
        UPDATE public.linkedin_coverage_debt
        SET source_window_latest_at = LEAST(source_window_latest_at, p_recovery_floor),
            status = 'expired_unresolved', resolved_at = pg_catalog.clock_timestamp(),
            resolution = 'outside configured outage recovery cap'
        WHERE id = debt_row.id;
        expired_count := expired_count + 1;
    END LOOP;
    UPDATE public.linkedin_scope_coverage_state state
    SET coverage_debt = EXISTS (
            SELECT 1 FROM public.linkedin_coverage_debt debt
            WHERE debt.scope_key = p_scope_key
              AND debt.status IN ('pending', 'expired_unresolved')
        ),
        coverage_debt_since = (
            SELECT MIN(debt.created_at) FROM public.linkedin_coverage_debt debt
            WHERE debt.scope_key = p_scope_key
              AND debt.status IN ('pending', 'expired_unresolved')
        ),
        updated_at = pg_catalog.clock_timestamp()
    WHERE state.scope_key = p_scope_key;
    RETURN expired_count;
END;
$$;

CREATE OR REPLACE FUNCTION public.prepare_linkedin_discovery_scope_state(
    p_scope_keys text[], p_recovery_floor timestamptz
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    scope_key_value text;
    states jsonb;
    debt_rows jsonb;
BEGIN
    IF p_scope_keys IS NULL OR pg_catalog.array_length(p_scope_keys, 1) IS NULL
       OR p_recovery_floor IS NULL THEN
        RAISE EXCEPTION 'scope keys and recovery floor are required' USING ERRCODE = '22023';
    END IF;
    FOREACH scope_key_value IN ARRAY p_scope_keys LOOP
        PERFORM public.expire_linkedin_coverage_debt(
            scope_key_value, p_recovery_floor
        );
    END LOOP;
    SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
        'scope_key', state.scope_key,
        'last_operational_success_at', state.last_operational_success_at,
        'recommended_pages', state.recommended_pages,
        'coverage_debt', state.coverage_debt,
        'last_deep_sweep_at', state.last_deep_sweep_at
    ) ORDER BY state.scope_key), '[]'::jsonb)
    INTO states
    FROM public.linkedin_scope_coverage_state state
    WHERE state.scope_key = ANY(p_scope_keys);
    SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
        'scope_key', selected.scope_key,
        'source_window_earliest_at', selected.source_window_earliest_at,
        'source_window_latest_at', selected.source_window_latest_at,
        'created_at', selected.created_at
    ) ORDER BY selected.scope_key), '[]'::jsonb)
    INTO debt_rows
    FROM (
        SELECT DISTINCT ON (debt.scope_key)
               debt.scope_key, debt.source_window_earliest_at,
               debt.source_window_latest_at, debt.created_at
        FROM public.linkedin_coverage_debt debt
        WHERE debt.scope_key = ANY(p_scope_keys) AND debt.status = 'pending'
        ORDER BY debt.scope_key, debt.source_window_earliest_at, debt.id
    ) selected;
    RETURN pg_catalog.jsonb_build_object('states', states, 'debt', debt_rows);
END;
$$;

CREATE OR REPLACE FUNCTION public.accept_linkedin_coverage_debt(
    p_debt_id bigint, p_reviewer text, p_reason text
) RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
BEGIN
    IF p_reviewer IS NULL OR pg_catalog.btrim(p_reviewer) = ''
       OR p_reason IS NULL OR pg_catalog.btrim(p_reason) = '' THEN
        RAISE EXCEPTION 'debt acceptance needs reviewer and reason' USING ERRCODE = '22023';
    END IF;
    UPDATE public.linkedin_coverage_debt
    SET status = 'accepted_boundary', reviewer = p_reviewer, resolution = p_reason,
        resolved_at = pg_catalog.clock_timestamp()
    WHERE id = p_debt_id AND status IN ('pending', 'expired_unresolved');
    IF NOT FOUND THEN
        IF EXISTS (
            SELECT 1 FROM public.linkedin_coverage_debt debt
            WHERE debt.id = p_debt_id AND debt.status = 'accepted_boundary'
              AND debt.reviewer = p_reviewer AND debt.resolution = p_reason
        ) THEN
            RETURN true;
        END IF;
        RAISE EXCEPTION 'only unresolved debt can be accepted' USING ERRCODE = '55000';
    END IF;
    UPDATE public.linkedin_scope_coverage_state state
    SET coverage_debt = EXISTS (
            SELECT 1 FROM public.linkedin_coverage_debt debt
            WHERE debt.scope_key = state.scope_key
              AND debt.status IN ('pending', 'expired_unresolved')
        ),
        coverage_debt_since = (
            SELECT MIN(debt.created_at) FROM public.linkedin_coverage_debt debt
            WHERE debt.scope_key = state.scope_key
              AND debt.status IN ('pending', 'expired_unresolved')
        ),
        updated_at = pg_catalog.clock_timestamp()
    WHERE EXISTS (
        SELECT 1 FROM public.linkedin_coverage_debt debt
        WHERE debt.id = p_debt_id AND debt.scope_key = state.scope_key
    );
    RETURN true;
END;
$$;

CREATE OR REPLACE FUNCTION public.advance_linkedin_discovery_watermark()
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    candidate public.linkedin_discovery_cycles%ROWTYPE;
    advanced boolean := false;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('linkedin-discovery-sequence-v1', 0));
    SELECT cycle.* INTO candidate
    FROM public.linkedin_discovery_cycles cycle
    WHERE cycle.search_status = 'sealed'
      AND cycle.operational_watermark_eligible
      AND NOT EXISTS (
          SELECT 1 FROM public.linkedin_discovery_cycles predecessor
          WHERE predecessor.discovery_sequence <= cycle.discovery_sequence
            AND predecessor.search_status <> 'sealed'
            AND NOT EXISTS (
                SELECT 1 FROM public.linkedin_discovery_cycle_resolutions resolution
                JOIN public.linkedin_discovery_cycles recovery
                  ON recovery.id = resolution.resolving_discovery_cycle_id
                WHERE resolution.failed_discovery_cycle_id = predecessor.id
                  AND recovery.search_status = 'sealed'
                  AND recovery.discovery_sequence <= cycle.discovery_sequence
            )
      )
    ORDER BY cycle.discovery_sequence DESC
    LIMIT 1;
    IF candidate.id IS NOT NULL THEN
        INSERT INTO public.scrape_run_state (
            id, last_successful_scrape_at, last_successful_discovery_cycle_id,
            last_successful_discovery_sequence
        ) VALUES (
            1, candidate.search_completed_at, candidate.id, candidate.discovery_sequence
        ) ON CONFLICT (id) DO UPDATE SET
            last_successful_scrape_at = GREATEST(
                COALESCE(public.scrape_run_state.last_successful_scrape_at, '-infinity'::timestamptz),
                EXCLUDED.last_successful_scrape_at
            ),
            last_successful_discovery_cycle_id = EXCLUDED.last_successful_discovery_cycle_id,
            last_successful_discovery_sequence = EXCLUDED.last_successful_discovery_sequence
        WHERE public.scrape_run_state.last_successful_discovery_sequence IS NULL
           OR public.scrape_run_state.last_successful_discovery_sequence < EXCLUDED.last_successful_discovery_sequence;
        advanced := FOUND;
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'watermark_advanced', advanced,
        'cycle_id', candidate.id,
        'discovery_sequence', candidate.discovery_sequence
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.resolve_failed_linkedin_discovery_cycle(
    p_failed_cycle_id bigint,
    p_resolving_cycle_id bigint,
    p_resolution_type text,
    p_reviewer text,
    p_reason text
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    failed_cycle public.linkedin_discovery_cycles%ROWTYPE;
    resolving_cycle public.linkedin_discovery_cycles%ROWTYPE;
    v_resolved_at timestamptz := pg_catalog.clock_timestamp();
    watermark_advanced boolean := false;
BEGIN
    IF p_resolution_type NOT IN ('recovered', 'reviewed_acceptance')
       OR p_reason IS NULL OR pg_catalog.btrim(p_reason) = ''
       OR (p_resolution_type = 'reviewed_acceptance'
           AND (p_reviewer IS NULL OR pg_catalog.btrim(p_reviewer) = '')) THEN
        RAISE EXCEPTION 'invalid failed-cycle resolution parameters' USING ERRCODE = '22023';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('linkedin-discovery-sequence-v1', 0));
    PERFORM id FROM public.linkedin_discovery_cycles
    WHERE id IN (p_failed_cycle_id, p_resolving_cycle_id)
    ORDER BY id FOR UPDATE;
    SELECT * INTO STRICT failed_cycle FROM public.linkedin_discovery_cycles
    WHERE id = p_failed_cycle_id;
    SELECT * INTO STRICT resolving_cycle FROM public.linkedin_discovery_cycles
    WHERE id = p_resolving_cycle_id;
    IF failed_cycle.search_status <> 'failed'
       OR resolving_cycle.search_status <> 'sealed'
       OR resolving_cycle.discovery_sequence <= failed_cycle.discovery_sequence THEN
        RAISE EXCEPTION 'failed-cycle resolution has invalid cycle states' USING ERRCODE = '55000';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.linkedin_discovery_requirements requirement
        JOIN public.linkedin_discovery_tasks task ON task.id = requirement.task_id
        LEFT JOIN public.listing_states state
          ON state.provider = task.provider AND state.source_job_id = task.source_job_id
        LEFT JOIN public.linkedin_discovery_requirement_acceptances acceptance
          ON acceptance.discovery_cycle_id = requirement.discovery_cycle_id
         AND acceptance.ingestion_run_id = requirement.ingestion_run_id
         AND acceptance.provider = requirement.provider
         AND acceptance.source_job_id = requirement.source_job_id
         AND acceptance.task_kind = requirement.task_kind
         AND acceptance.requirement_key = requirement.requirement_key
        WHERE requirement.discovery_cycle_id = p_failed_cycle_id AND requirement.required
          AND acceptance.discovery_cycle_id IS NULL
          AND NOT (
            task.status = 'terminal_unavailable'
            OR (task.status = 'complete' AND task.canonical_job_id IS NOT NULL
                AND state.canonical_job_id = task.canonical_job_id)
          )
    ) THEN
        RAISE EXCEPTION 'failed cycle still has unresolved requirements' USING ERRCODE = '55000';
    END IF;
    IF p_resolution_type = 'recovered' AND EXISTS (
        SELECT 1
        FROM public.linkedin_coverage_debt debt
        LEFT JOIN public.linkedin_discovery_cycles recovery
          ON recovery.id = debt.resolved_by_discovery_cycle_id
        WHERE debt.origin_discovery_cycle_id = p_failed_cycle_id
          AND (
              debt.status <> 'resolved'
              OR recovery.search_status <> 'sealed'
              OR recovery.discovery_sequence <= failed_cycle.discovery_sequence
              OR recovery.discovery_sequence > resolving_cycle.discovery_sequence
          )
    ) THEN
        RAISE EXCEPTION 'recovery cycle does not cover every failed interval' USING ERRCODE = '55000';
    END IF;
    UPDATE public.linkedin_coverage_debt
    SET status = CASE WHEN p_resolution_type = 'recovered' THEN 'resolved' ELSE 'accepted_boundary' END,
        resolved_at = v_resolved_at,
        resolution = p_reason,
        reviewer = CASE WHEN p_resolution_type = 'reviewed_acceptance' THEN p_reviewer ELSE reviewer END
    WHERE origin_discovery_cycle_id = p_failed_cycle_id
      AND status IN ('pending', 'expired_unresolved');
    INSERT INTO public.linkedin_discovery_cycle_resolutions (
        failed_discovery_cycle_id, resolving_discovery_cycle_id,
        resolution_type, reviewer, reason
    ) VALUES (
        p_failed_cycle_id, p_resolving_cycle_id, p_resolution_type,
        p_reviewer, p_reason
    ) ON CONFLICT (failed_discovery_cycle_id) DO NOTHING;
    IF NOT FOUND THEN
        IF NOT EXISTS (
            SELECT 1 FROM public.linkedin_discovery_cycle_resolutions
            WHERE failed_discovery_cycle_id = p_failed_cycle_id
              AND resolving_discovery_cycle_id = p_resolving_cycle_id
              AND resolution_type = p_resolution_type
              AND reviewer IS NOT DISTINCT FROM p_reviewer
              AND reason = p_reason
        ) THEN
            RAISE EXCEPTION 'conflicting failed-cycle resolution' USING ERRCODE = '23000';
        END IF;
    END IF;
    UPDATE public.linkedin_scope_coverage_state state
    SET coverage_debt = EXISTS (
            SELECT 1 FROM public.linkedin_coverage_debt debt
            WHERE debt.scope_key = state.scope_key
              AND debt.status IN ('pending', 'expired_unresolved')
        ),
        coverage_debt_since = (
            SELECT MIN(debt.created_at) FROM public.linkedin_coverage_debt debt
            WHERE debt.scope_key = state.scope_key
              AND debt.status IN ('pending', 'expired_unresolved')
        ),
        updated_at = v_resolved_at
    WHERE EXISTS (
        SELECT 1 FROM public.linkedin_discovery_cycle_scopes scope
        WHERE scope.discovery_cycle_id = p_failed_cycle_id
          AND scope.scope_key = state.scope_key
    );
    watermark_advanced := COALESCE(
        (public.advance_linkedin_discovery_watermark()->>'watermark_advanced')::boolean,
        false
    );
    RETURN pg_catalog.jsonb_build_object(
        'failed_cycle_id', p_failed_cycle_id,
        'resolving_cycle_id', p_resolving_cycle_id,
        'resolution_type', p_resolution_type,
        'watermark_advanced', watermark_advanced
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.resolve_eligible_failed_linkedin_discovery_cycles(
    p_resolving_cycle_id bigint
) RETURNS integer
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    resolving_cycle public.linkedin_discovery_cycles%ROWTYPE;
    failed_cycle_id bigint;
    resolved_count integer := 0;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('linkedin-discovery-sequence-v1', 0));
    SELECT * INTO STRICT resolving_cycle FROM public.linkedin_discovery_cycles
    WHERE id = p_resolving_cycle_id FOR SHARE;
    IF resolving_cycle.search_status <> 'sealed' THEN
        RAISE EXCEPTION 'resolving cycle must be sealed' USING ERRCODE = '55000';
    END IF;
    FOR failed_cycle_id IN
        SELECT failed.id
        FROM public.linkedin_discovery_cycles failed
        WHERE failed.search_status = 'failed'
          AND failed.discovery_sequence < resolving_cycle.discovery_sequence
          AND NOT EXISTS (
              SELECT 1 FROM public.linkedin_discovery_cycle_resolutions resolution
              WHERE resolution.failed_discovery_cycle_id = failed.id
          )
          AND NOT EXISTS (
              SELECT 1 FROM public.linkedin_coverage_debt debt
              LEFT JOIN public.linkedin_discovery_cycles recovery
                ON recovery.id = debt.resolved_by_discovery_cycle_id
              WHERE debt.origin_discovery_cycle_id = failed.id
                AND (
                    debt.status <> 'resolved'
                    OR recovery.search_status <> 'sealed'
                    OR recovery.discovery_sequence <= failed.discovery_sequence
                    OR recovery.discovery_sequence > resolving_cycle.discovery_sequence
                )
          )
          AND NOT EXISTS (
              SELECT 1
              FROM public.linkedin_discovery_requirements requirement
              JOIN public.linkedin_discovery_tasks task ON task.id = requirement.task_id
              LEFT JOIN public.listing_states state
                ON state.provider = task.provider AND state.source_job_id = task.source_job_id
              LEFT JOIN public.linkedin_discovery_requirement_acceptances acceptance
                ON acceptance.discovery_cycle_id = requirement.discovery_cycle_id
               AND acceptance.ingestion_run_id = requirement.ingestion_run_id
               AND acceptance.provider = requirement.provider
               AND acceptance.source_job_id = requirement.source_job_id
               AND acceptance.task_kind = requirement.task_kind
               AND acceptance.requirement_key = requirement.requirement_key
              WHERE requirement.discovery_cycle_id = failed.id AND requirement.required
                AND acceptance.discovery_cycle_id IS NULL
                AND NOT (
                  task.status = 'terminal_unavailable'
                  OR (task.status = 'complete' AND task.canonical_job_id IS NOT NULL
                      AND state.canonical_job_id = task.canonical_job_id)
                )
          )
        ORDER BY failed.discovery_sequence
    LOOP
        PERFORM public.resolve_failed_linkedin_discovery_cycle(
            failed_cycle_id, p_resolving_cycle_id, 'recovered', NULL,
            'all failed scope intervals and requirements recovered'
        );
        resolved_count := resolved_count + 1;
    END LOOP;
    RETURN resolved_count;
END;
$$;

CREATE OR REPLACE FUNCTION public.accept_linkedin_discovery_requirement(
    p_discovery_cycle_id bigint,
    p_ingestion_run_id uuid,
    p_provider text,
    p_source_job_id text,
    p_task_kind text,
    p_requirement_key text,
    p_reviewer text,
    p_reason text
) RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    task_row public.linkedin_discovery_tasks%ROWTYPE;
BEGIN
    IF p_reviewer IS NULL OR pg_catalog.btrim(p_reviewer) = ''
       OR p_reason IS NULL OR pg_catalog.btrim(p_reason) = '' THEN
        RAISE EXCEPTION 'requirement acceptance needs reviewer and reason' USING ERRCODE = '22023';
    END IF;
    SELECT task.* INTO STRICT task_row
    FROM public.linkedin_discovery_requirements requirement
    JOIN public.linkedin_discovery_tasks task ON task.id = requirement.task_id
    WHERE requirement.discovery_cycle_id = p_discovery_cycle_id
      AND requirement.ingestion_run_id = p_ingestion_run_id
      AND requirement.provider = p_provider
      AND requirement.source_job_id = p_source_job_id
      AND requirement.task_kind = p_task_kind
      AND requirement.requirement_key = p_requirement_key
      AND requirement.required
    FOR UPDATE OF task;
    IF task_row.status <> 'failed_terminal' THEN
        RAISE EXCEPTION 'only a failed-terminal requirement can be accepted' USING ERRCODE = '55000';
    END IF;
    INSERT INTO public.linkedin_discovery_requirement_acceptances (
        discovery_cycle_id, ingestion_run_id, provider, source_job_id,
        task_kind, requirement_key, reviewer, reason
    ) VALUES (
        p_discovery_cycle_id, p_ingestion_run_id, p_provider, p_source_job_id,
        p_task_kind, p_requirement_key, p_reviewer, p_reason
    ) ON CONFLICT DO NOTHING;
    IF NOT FOUND AND NOT EXISTS (
        SELECT 1 FROM public.linkedin_discovery_requirement_acceptances acceptance
        WHERE acceptance.discovery_cycle_id = p_discovery_cycle_id
          AND acceptance.ingestion_run_id = p_ingestion_run_id
          AND acceptance.provider = p_provider
          AND acceptance.source_job_id = p_source_job_id
          AND acceptance.task_kind = p_task_kind
          AND acceptance.requirement_key = p_requirement_key
          AND acceptance.reviewer = p_reviewer
          AND acceptance.reason = p_reason
    ) THEN
        RAISE EXCEPTION 'conflicting requirement acceptance' USING ERRCODE = '23000';
    END IF;
    RETURN true;
END;
$$;

CREATE OR REPLACE FUNCTION public.seal_linkedin_discovery_cycle(p_cycle_id bigint, p_advance_watermark boolean)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    cycle_row public.linkedin_discovery_cycles%ROWTYPE;
    completed integer;
    debt integer;
    unresolved integer;
    sealed_at timestamptz := pg_catalog.clock_timestamp();
    watermark_advanced boolean := false;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('linkedin-discovery-sequence-v1', 0));
    SELECT * INTO STRICT cycle_row FROM public.linkedin_discovery_cycles WHERE id = p_cycle_id FOR UPDATE;
    IF cycle_row.search_status = 'sealed' THEN
        IF cycle_row.operational_watermark_eligible THEN
            watermark_advanced := COALESCE(
                (public.advance_linkedin_discovery_watermark()->>'watermark_advanced')::boolean,
                false
            );
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'cycle_id', p_cycle_id,
            'discovery_sequence', cycle_row.discovery_sequence,
            'sealed_at', cycle_row.search_completed_at,
            'watermark_advanced', watermark_advanced
        );
    END IF;
    SELECT COUNT(*) INTO completed
    FROM public.linkedin_discovery_cycle_scopes scope
    WHERE scope.discovery_cycle_id = p_cycle_id AND scope.required;
    IF cycle_row.search_status <> 'running'
       OR completed <> cycle_row.required_scope_count
       OR EXISTS (
            SELECT 1
            FROM public.linkedin_discovery_cycle_scopes scope
            LEFT JOIN public.ingestion_runs run ON run.id = scope.ingestion_run_id
            WHERE scope.discovery_cycle_id = p_cycle_id AND scope.required
              AND (scope.status <> 'complete' OR scope.enqueue_committed_at IS NULL
                   OR run.status <> 'complete'
                    OR run.coverage_status <> 'exhausted')
       )
       OR EXISTS (
            SELECT 1
            FROM public.linkedin_discovery_cycle_scopes scope
            JOIN public.linkedin_ingestion_pages page ON page.ingestion_run_id = scope.ingestion_run_id
            WHERE scope.discovery_cycle_id = p_cycle_id
              AND (page.source_window_earliest_at > scope.source_window_earliest_at
                   OR page.source_window_latest_at < scope.source_window_latest_at)
       )
       OR EXISTS (
            SELECT 1
            FROM public.linkedin_discovery_cycle_sources source
            LEFT JOIN public.linkedin_discovery_requirements requirement
              ON requirement.discovery_cycle_id = source.discovery_cycle_id
             AND requirement.provider = source.provider
             AND requirement.source_job_id = source.source_job_id
             AND requirement.required
            WHERE source.discovery_cycle_id = p_cycle_id AND requirement.task_id IS NULL
       ) THEN
        RAISE EXCEPTION 'discovery cycle is not sealable' USING ERRCODE = '55000';
    END IF;
    SELECT COUNT(*) INTO debt FROM public.linkedin_coverage_debt
    WHERE origin_discovery_cycle_id = p_cycle_id
      AND status IN ('pending', 'expired_unresolved');
    SELECT COUNT(*) INTO unresolved
    FROM public.linkedin_discovery_requirements requirement
    JOIN public.linkedin_discovery_tasks task ON task.id = requirement.task_id
    LEFT JOIN public.listing_states state
      ON state.provider = task.provider AND state.source_job_id = task.source_job_id
    LEFT JOIN public.linkedin_discovery_requirement_acceptances acceptance
      ON acceptance.discovery_cycle_id = requirement.discovery_cycle_id
     AND acceptance.ingestion_run_id = requirement.ingestion_run_id
     AND acceptance.provider = requirement.provider
     AND acceptance.source_job_id = requirement.source_job_id
     AND acceptance.task_kind = requirement.task_kind
     AND acceptance.requirement_key = requirement.requirement_key
    WHERE requirement.discovery_cycle_id = p_cycle_id AND requirement.required
      AND acceptance.discovery_cycle_id IS NULL
      AND NOT (
        task.status = 'terminal_unavailable'
        OR (task.status = 'complete' AND task.canonical_job_id IS NOT NULL
            AND state.canonical_job_id = task.canonical_job_id)
      );
    UPDATE public.linkedin_discovery_cycles SET search_status = 'sealed', search_completed_at = sealed_at,
        completed_scope_count = completed, coverage_debt_count = debt,
        canonical_status = CASE WHEN unresolved = 0 THEN 'applied' ELSE 'pending' END,
        operational_watermark_eligible = p_advance_watermark
    WHERE id = p_cycle_id;
    UPDATE public.linkedin_scope_coverage_state state
    SET last_operational_success_at = sealed_at,
        last_operational_discovery_sequence = cycle_row.discovery_sequence,
        updated_at = sealed_at
    FROM public.linkedin_discovery_cycle_scopes scope
    WHERE scope.discovery_cycle_id = p_cycle_id AND state.scope_key = scope.scope_key
      AND (state.last_operational_discovery_sequence IS NULL
           OR state.last_operational_discovery_sequence < cycle_row.discovery_sequence);
    IF p_advance_watermark THEN
        watermark_advanced := COALESCE(
            (public.advance_linkedin_discovery_watermark()->>'watermark_advanced')::boolean,
            false
        );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'cycle_id', p_cycle_id, 'discovery_sequence', cycle_row.discovery_sequence,
        'sealed_at', sealed_at, 'coverage_debt_count', debt,
        'canonical_status', CASE WHEN unresolved = 0 THEN 'applied' ELSE 'pending' END,
        'watermark_advanced', watermark_advanced
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.claim_linkedin_discovery_tasks(p_worker_id text, p_limit integer, p_order_mode text)
RETURNS SETOF public.linkedin_discovery_tasks
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
BEGIN
    IF p_worker_id IS NULL OR pg_catalog.btrim(p_worker_id) = ''
       OR p_limit IS NULL OR p_limit < 1 OR p_limit > 100
       OR p_order_mode IS NULL OR p_order_mode NOT IN ('oldest', 'newest') THEN
        RAISE EXCEPTION 'invalid task claim parameters' USING ERRCODE = '22023';
    END IF;
    UPDATE public.linkedin_discovery_tasks
    SET status = 'failed_terminal', completed_at = pg_catalog.clock_timestamp(),
        last_error_code = COALESCE(last_error_code, 'lease_expired_attempts_exhausted'),
        leased_by = NULL, leased_at = NULL, lease_expires_at = NULL, lease_token = NULL
    WHERE status = 'leased' AND lease_expires_at <= pg_catalog.clock_timestamp()
      AND attempt_count >= max_attempts;
    UPDATE public.linkedin_discovery_tasks SET status = 'failed_retryable', leased_by = NULL, leased_at = NULL,
        lease_expires_at = NULL, lease_token = NULL, available_at = pg_catalog.clock_timestamp()
    WHERE status = 'leased' AND lease_expires_at <= pg_catalog.clock_timestamp()
      AND attempt_count < max_attempts;
    IF p_order_mode = 'newest' THEN
        RETURN QUERY WITH picked AS (
            SELECT id FROM public.linkedin_discovery_tasks
            WHERE status IN ('pending', 'failed_retryable')
              AND task_kind = 'initial_detail' AND requirement_key = 'first'
              AND attempt_count < max_attempts
              AND available_at <= pg_catalog.clock_timestamp()
            ORDER BY priority DESC, first_observed_at DESC, id FOR UPDATE SKIP LOCKED LIMIT p_limit
        ) UPDATE public.linkedin_discovery_tasks task SET status = 'leased', leased_by = p_worker_id,
            leased_at = pg_catalog.clock_timestamp(), lease_expires_at = pg_catalog.clock_timestamp() + interval '10 minutes',
            lease_token = extensions.gen_random_uuid(), attempt_count = task.attempt_count + 1
        FROM picked WHERE task.id = picked.id RETURNING task.*;
    ELSE
        RETURN QUERY WITH picked AS (
            SELECT task.id FROM public.linkedin_discovery_tasks task
            WHERE task.status IN ('pending', 'failed_retryable')
              AND task.attempt_count < task.max_attempts
              AND task.available_at <= pg_catalog.clock_timestamp()
            ORDER BY COALESCE((
                SELECT MIN(cycle.discovery_sequence)
                FROM public.linkedin_discovery_requirements requirement
                JOIN public.linkedin_discovery_cycles cycle
                  ON cycle.id = requirement.discovery_cycle_id
                WHERE requirement.task_id = task.id
                  AND requirement.required
            ), 9223372036854775807),
            task.priority DESC, task.first_observed_at, task.id
            FOR UPDATE OF task SKIP LOCKED LIMIT p_limit
        ) UPDATE public.linkedin_discovery_tasks task SET status = 'leased', leased_by = p_worker_id,
            leased_at = pg_catalog.clock_timestamp(), lease_expires_at = pg_catalog.clock_timestamp() + interval '10 minutes',
            lease_token = extensions.gen_random_uuid(), attempt_count = task.attempt_count + 1
        FROM picked WHERE task.id = picked.id RETURNING task.*;
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION public.heartbeat_linkedin_discovery_task(
    p_task_id bigint, p_worker_id text, p_lease_token uuid
) RETURNS timestamptz
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE extended_until timestamptz := pg_catalog.clock_timestamp() + interval '10 minutes';
BEGIN
    UPDATE public.linkedin_discovery_tasks
    SET lease_expires_at = extended_until
    WHERE id = p_task_id AND status = 'leased' AND leased_by = p_worker_id
      AND lease_token = p_lease_token AND lease_expires_at > pg_catalog.clock_timestamp();
    IF NOT FOUND THEN
        RAISE EXCEPTION 'task lease lost' USING ERRCODE = '40001';
    END IF;
    RETURN extended_until;
END;
$$;

CREATE OR REPLACE FUNCTION public.apply_linkedin_discovery_task_canonical(
    p_task_id bigint,
    p_worker_id text,
    p_lease_token uuid,
    p_application jsonb
) RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $$
DECLARE
    task_row public.linkedin_discovery_tasks%ROWTYPE;
    application_hash text;
    candidate_set_revision text;
    current_candidate_set_revision text;
    expected_membership_provenance_revision bigint;
    source_payload jsonb;
    canonical jsonb;
    canonical_payload jsonb;
    expected jsonb;
    content_version jsonb;
    memberships jsonb;
    membership jsonb;
    relist jsonb;
    canonical_action text;
    target_job_id text;
    task_source_job_id text;
    task_ingestion_run_id uuid;
    expected_last_seen_at timestamptz;
    expected_listing_instances jsonb;
    application_observed_at timestamptz;
    memberships_observed_at timestamptz;
    content_observed_at timestamptz;
    membership_first_matched_at timestamptz;
    membership_last_matched_at timestamptz;
    provenance jsonb;
    write_columns text;
    affected integer;
    mapped_canonical_job_id text;
    mapped_content_job_id text;
    prior_state_last_seen_at timestamptz;
    prior_state_latest_posted_date date;
    prior_state_content_hash text;
    application_completed_at timestamptz;
    applied_canonical_revision bigint;
    relist_applied boolean;
    job_write_fields constant text[] := ARRAY[
        'job_id', 'company', 'job_title', 'level', 'location', 'description',
        'status', 'is_active', 'application_date', 'resume_score', 'notes',
        'scraped_at', 'last_checked', 'job_state', 'resume_score_stage',
        'is_interested', 'customized_resume_id', 'provider', 'posted_at',
        'search_query', 'archetype', 'filter_profile', 'canonical_key',
        'original_job_id', 'latest_job_id', 'first_seen_at', 'last_seen_at',
        'last_seen_posted_at', 'seen_count', 'posting_wave_count', 'repost_count',
        'listing_instances', 'description_fingerprint', 'same_id_relist_count',
        'posted_relative_text', 'applicant_count', 'applicant_count_text',
        'applicant_count_type', 'salary_text', 'salary_min', 'salary_max',
        'salary_currency', 'recruiter_name', 'recruiter_profile_url',
        'recruiter_identifier', 'detail_metadata_checked_at', 'freehire_category',
        'freehire_seniority', 'is_remote', 'freehire_remote_evidence',
        'freehire_compat_status', 'freehire_compat_input_hash',
        'freehire_compat_import_hash', 'freehire_compat_model',
        'freehire_compat_prompt_version', 'freehire_compat_schema_version',
        'freehire_compat_confidence', 'freehire_compat_classified_at',
        'freehire_compat_error', 'freehire_compat_attempts',
        'freehire_compat_claimed_at', 'freehire_compat_claimed_by',
        'freehire_compat_next_retry_at', 'freehire_compat_provenance'
    ];
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('linkedin-canonical-publication-v1', 0)
    );

    IF p_application IS NULL
       OR pg_catalog.jsonb_typeof(p_application) <> 'object'
       OR p_application->>'version' NOT IN (
           'linkedin-canonical-task-apply-v3',
           'linkedin-canonical-task-apply-v4'
       ) THEN
        RAISE EXCEPTION 'invalid canonical task application' USING ERRCODE = '22023';
    END IF;
    candidate_set_revision := p_application->>'provider_candidate_set_revision';
    IF candidate_set_revision IS NULL
       OR candidate_set_revision !~ '^[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'invalid provider candidate-set revision' USING ERRCODE = '22023';
    END IF;
    application_hash := pg_catalog.encode(
        extensions.digest(p_application::text, 'sha256'), 'hex'
    );

    SELECT * INTO task_row
    FROM public.linkedin_discovery_tasks
    WHERE id = p_task_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'adaptive discovery task does not exist' USING ERRCODE = '22023';
    END IF;
    IF task_row.status = 'complete' THEN
        IF task_row.canonical_applied_lease_token = p_lease_token
           AND task_row.canonical_application_hash = application_hash THEN
            RETURN pg_catalog.jsonb_build_object(
                'outcome', 'replayed', 'task_id', task_row.id,
                'canonical_job_id', task_row.canonical_job_id,
                'action', p_application->'canonical'->>'action',
                'canonical_revision', (
                    SELECT job.canonical_revision
                    FROM public.jobs job
                    WHERE job.job_id = task_row.canonical_job_id
                ),
                'provider_candidate_set_revision',
                    public.get_canonical_provider_revision('linkedin'),
                'application_hash', application_hash,
                'completed_at', task_row.completed_at, 'replayed', true
            );
        END IF;
        RAISE EXCEPTION 'canonical task receipt conflicts with completed application'
            USING ERRCODE = '23505';
    END IF;
    IF task_row.status <> 'leased'
       OR task_row.leased_by IS DISTINCT FROM p_worker_id
       OR task_row.lease_token IS DISTINCT FROM p_lease_token
       OR task_row.lease_expires_at <= pg_catalog.clock_timestamp() THEN
        RAISE EXCEPTION 'adaptive discovery task lease lost' USING ERRCODE = '55000';
    END IF;

    IF pg_catalog.jsonb_typeof(p_application->'membership_provenance_revision') <> 'number' THEN
        RAISE EXCEPTION 'invalid membership provenance revision' USING ERRCODE = '22023';
    END IF;
    BEGIN
        expected_membership_provenance_revision :=
            (p_application->>'membership_provenance_revision')::bigint;
    EXCEPTION WHEN invalid_text_representation OR numeric_value_out_of_range THEN
        RAISE EXCEPTION 'invalid membership provenance revision' USING ERRCODE = '22023';
    END;
    IF expected_membership_provenance_revision < 0 THEN
        RAISE EXCEPTION 'invalid membership provenance revision' USING ERRCODE = '22023';
    END IF;
    IF expected_membership_provenance_revision IS DISTINCT FROM
       task_row.membership_provenance_revision THEN
        RETURN pg_catalog.jsonb_build_object(
            'outcome', 'stale_plan', 'task_id', task_row.id,
            'canonical_job_id', task_row.canonical_job_id,
            'action', p_application->'canonical'->>'action',
            'application_hash', application_hash, 'completed_at', NULL,
            'replayed', false,
            'task_membership_provenances', task_row.membership_provenances,
            'task_membership_provenance_revision', task_row.membership_provenance_revision
        );
    END IF;

    source_payload := p_application->'source';
    canonical := p_application->'canonical';
    canonical_payload := canonical->'payload';
    expected := COALESCE(canonical->'expected', '{}'::jsonb);
    content_version := NULLIF(p_application->'content_version', 'null'::jsonb);
    memberships := p_application->'memberships';
    relist := NULLIF(p_application->'relist', 'null'::jsonb);
    IF pg_catalog.jsonb_typeof(source_payload) <> 'object'
       OR pg_catalog.jsonb_typeof(canonical) <> 'object'
       OR pg_catalog.jsonb_typeof(canonical_payload) <> 'object'
       OR pg_catalog.jsonb_typeof(expected) <> 'object'
       OR pg_catalog.jsonb_typeof(memberships) <> 'array'
       OR pg_catalog.jsonb_array_length(memberships) = 0 THEN
        RAISE EXCEPTION 'canonical task application has invalid object fields' USING ERRCODE = '22023';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements(memberships) membership_item(value)
        GROUP BY membership_item.value
        HAVING COUNT(*) > 1
    ) THEN
        RAISE EXCEPTION 'canonical task application contains duplicate memberships'
            USING ERRCODE = '22023';
    END IF;
    FOR membership IN
        SELECT value
        FROM pg_catalog.jsonb_array_elements(memberships)
        ORDER BY value::text
    LOOP
        BEGIN
            membership_first_matched_at := NULLIF(membership->>'first_matched_at', '')::timestamptz;
            membership_last_matched_at := NULLIF(membership->>'last_matched_at', '')::timestamptz;
        EXCEPTION WHEN invalid_text_representation OR datetime_field_overflow THEN
            RAISE EXCEPTION 'canonical membership application has invalid timestamps'
                USING ERRCODE = '22023';
        END;
        IF pg_catalog.jsonb_typeof(membership) <> 'object'
           OR COALESCE(pg_catalog.btrim(membership->>'archetype'), '') = ''
           OR pg_catalog.jsonb_typeof(membership->'query_scope') <> 'object'
           OR COALESCE(pg_catalog.btrim(membership->>'query_id'), '') = ''
           OR COALESCE(pg_catalog.btrim(membership->>'query'), '') = ''
           OR membership->>'query_type' NOT IN ('precision', 'recall')
           OR COALESCE(membership->>'language', '') !~ '^[a-z]{2}(-[A-Z]{2})?$'
           OR membership_first_matched_at IS NULL
           OR membership_last_matched_at IS NULL
           OR membership_first_matched_at > membership_last_matched_at
           OR (
               membership->'query_scope'->>'lane' IS NOT NULL
               AND membership->'query_scope'->>'lane' IS DISTINCT FROM membership->>'archetype'
           )
           OR (
               membership->'query_scope'->>'archetype' IS NOT NULL
               AND membership->'query_scope'->>'archetype' IS DISTINCT FROM membership->>'archetype'
           )
           OR membership->>'filter_status' NOT IN ('pending', 'included', 'review', 'filtered')
           OR membership->'is_filtered' IS NULL
           OR pg_catalog.jsonb_typeof(membership->'is_filtered') <> 'boolean' THEN
            RAISE EXCEPTION 'canonical membership application is invalid' USING ERRCODE = '22023';
        END IF;
        memberships_observed_at := COALESCE(
            GREATEST(memberships_observed_at, membership_last_matched_at),
            memberships_observed_at,
            membership_last_matched_at
        );
    END LOOP;

    canonical_action := canonical->>'action';
    target_job_id := pg_catalog.btrim(COALESCE(canonical->>'canonical_job_id', ''));
    task_source_job_id := pg_catalog.btrim(COALESCE(source_payload->>'source_job_id', ''));
    BEGIN
        task_ingestion_run_id := (source_payload->>'ingestion_run_id')::uuid;
    EXCEPTION WHEN invalid_text_representation THEN
        RAISE EXCEPTION 'canonical task application has invalid ingestion run' USING ERRCODE = '22023';
    END;
    IF canonical_action NOT IN ('insert', 'update', 'accepted_relist')
       OR target_job_id = '' OR task_source_job_id = ''
       OR source_payload->>'provider' IS DISTINCT FROM 'linkedin'
       OR task_source_job_id IS DISTINCT FROM task_row.source_job_id
       OR task_ingestion_run_id IS DISTINCT FROM task_row.first_ingestion_run_id THEN
        RAISE EXCEPTION 'canonical task application source or action mismatch' USING ERRCODE = '22023';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_catalog.jsonb_object_keys(canonical_payload) AS supplied(field)
        WHERE NOT supplied.field = ANY (job_write_fields)
    ) THEN
        RAISE EXCEPTION 'canonical task application contains an unsupported job field'
            USING ERRCODE = '22023';
    END IF;
    IF canonical_action = 'insert' THEN
        IF canonical_payload->>'job_id' IS DISTINCT FROM target_job_id
           OR canonical_payload->>'provider' IS DISTINCT FROM 'linkedin' THEN
            RAISE EXCEPTION 'canonical insert identity mismatch' USING ERRCODE = '22023';
        END IF;
    ELSIF canonical_payload ? 'job_id' THEN
        RAISE EXCEPTION 'canonical update payload must not contain job_id' USING ERRCODE = '22023';
    END IF;
    LOCK TABLE public.jobs IN SHARE ROW EXCLUSIVE MODE;
    IF p_application->>'version' = 'linkedin-canonical-task-apply-v3' THEN
        SELECT pg_catalog.encode(extensions.digest(pg_catalog.convert_to(COALESCE(
            pg_catalog.string_agg(
                pg_catalog.octet_length(job.job_id)::text || ':' || job.job_id
                    || pg_catalog.octet_length(job.canonical_revision::text)::text || ':'
                    || job.canonical_revision::text,
                '' ORDER BY pg_catalog.convert_to(job.job_id, 'UTF8')
            ), ''
        ), 'UTF8'), 'sha256'), 'hex')
        INTO current_candidate_set_revision
        FROM public.jobs job
        WHERE job.provider = 'linkedin';
    ELSE
        SELECT pg_catalog.lpad(pg_catalog.to_hex(provider_revision.revision), 64, '0')
        INTO STRICT current_candidate_set_revision
        FROM public.canonical_provider_revisions provider_revision
        WHERE provider_revision.provider = 'linkedin'
        FOR UPDATE;
    END IF;
    IF candidate_set_revision IS DISTINCT FROM current_candidate_set_revision THEN
        RETURN pg_catalog.jsonb_build_object(
            'outcome', 'stale_plan', 'task_id', task_row.id,
            'canonical_job_id', target_job_id, 'action', canonical_action,
            'application_hash', application_hash, 'completed_at', NULL,
            'replayed', false
        );
    END IF;

    SELECT pg_catalog.string_agg(
        pg_catalog.format('%I', allowed.field), ', ' ORDER BY allowed.position
    ) INTO write_columns
    FROM pg_catalog.unnest(job_write_fields) WITH ORDINALITY AS allowed(field, position)
    WHERE canonical_payload ? allowed.field;
    IF write_columns IS NULL OR canonical_payload = '{}'::jsonb THEN
        RAISE EXCEPTION 'canonical task application has an empty job payload' USING ERRCODE = '22023';
    END IF;

    BEGIN
        expected_last_seen_at := NULLIF(expected->>'last_seen_at', '')::timestamptz;
    EXCEPTION WHEN invalid_text_representation OR datetime_field_overflow THEN
        RAISE EXCEPTION 'canonical expected timestamp is invalid' USING ERRCODE = '22023';
    END;
    expected_listing_instances := COALESCE(expected->'listing_instances', '[]'::jsonb);
    IF canonical_action = 'accepted_relist'
       AND pg_catalog.jsonb_typeof(expected_listing_instances) <> 'array' THEN
        RAISE EXCEPTION 'canonical expected listing instances must be an array' USING ERRCODE = '22023';
    END IF;

    IF canonical_action <> 'insert' THEN
        PERFORM 1 FROM public.jobs WHERE job_id = target_job_id FOR UPDATE;
        IF NOT FOUND THEN
            RETURN pg_catalog.jsonb_build_object(
                'outcome', 'stale_plan', 'task_id', task_row.id,
                'canonical_job_id', target_job_id, 'action', canonical_action,
                'application_hash', application_hash, 'completed_at', NULL,
                'replayed', false
            );
        END IF;
    END IF;
    SELECT state.canonical_job_id, state.last_seen_at,
           state.latest_trusted_posted_date, state.current_content_hash
    INTO mapped_canonical_job_id, prior_state_last_seen_at,
         prior_state_latest_posted_date, prior_state_content_hash
    FROM public.listing_states state
    WHERE state.provider = 'linkedin' AND state.source_job_id = task_source_job_id
    FOR UPDATE;
    IF FOUND AND mapped_canonical_job_id IS NOT NULL
       AND mapped_canonical_job_id IS DISTINCT FROM target_job_id THEN
        RETURN pg_catalog.jsonb_build_object(
            'outcome', 'stale_plan', 'task_id', task_row.id,
            'canonical_job_id', mapped_canonical_job_id, 'action', canonical_action,
            'application_hash', application_hash, 'completed_at', NULL,
            'replayed', false
        );
    END IF;
    IF content_version IS NOT NULL THEN
        IF pg_catalog.jsonb_typeof(content_version) <> 'object'
           OR COALESCE(content_version->>'content_hash', '') !~ '^[0-9a-f]{64}$'
           OR content_version->>'description' IS NULL THEN
            RAISE EXCEPTION 'canonical content version is invalid' USING ERRCODE = '22023';
        END IF;
        BEGIN
            content_observed_at := NULLIF(content_version->>'observed_at', '')::timestamptz;
        EXCEPTION WHEN invalid_text_representation OR datetime_field_overflow THEN
            RAISE EXCEPTION 'canonical content version has invalid observed_at' USING ERRCODE = '22023';
        END;
        IF content_observed_at IS NULL THEN
            RAISE EXCEPTION 'canonical content version has invalid observed_at' USING ERRCODE = '22023';
        END IF;
        SELECT version.canonical_job_id INTO mapped_content_job_id
        FROM public.listing_content_versions version
        WHERE version.provider = 'linkedin'
          AND version.source_job_id = task_source_job_id
          AND version.content_hash = content_version->>'content_hash'
        FOR UPDATE;
        IF FOUND AND mapped_content_job_id IS NOT NULL
           AND mapped_content_job_id IS DISTINCT FROM target_job_id THEN
            RAISE EXCEPTION 'listing content canonical mapping conflict' USING ERRCODE = '23000';
        END IF;
    END IF;

    IF canonical_action = 'accepted_relist' THEN
        IF pg_catalog.jsonb_typeof(content_version) <> 'object'
           OR pg_catalog.jsonb_typeof(relist) <> 'object' THEN
            RAISE EXCEPTION 'accepted relist application lacks evidence or content'
                USING ERRCODE = '22023';
        END IF;
        relist_applied := public.apply_linkedin_relist_projection(
            target_job_id,
            task_source_job_id,
            task_ingestion_run_id,
            NULLIF(relist->>'relisted_on', '')::date,
            NULLIF(relist->>'observed_at', '')::timestamptz,
            canonical_payload,
            expected_listing_instances,
            expected_last_seen_at,
            COALESCE(relist->'evidence', '{}'::jsonb),
            content_version->>'description',
            content_version->>'content_hash',
            content_version->>'description_fingerprint'
        );
        IF NOT relist_applied THEN
            RETURN pg_catalog.jsonb_build_object(
                'outcome', 'stale_plan', 'task_id', task_row.id,
                'canonical_job_id', target_job_id, 'action', canonical_action,
                'application_hash', application_hash, 'completed_at', NULL,
                'replayed', false
            );
        END IF;
        EXECUTE pg_catalog.format(
            'UPDATE public.jobs AS target SET (%1$s) = (SELECT %1$s FROM pg_catalog.jsonb_populate_record(NULL::public.jobs, $1) AS populated) WHERE target.job_id = $2',
            write_columns
        ) USING canonical_payload, target_job_id;
        GET DIAGNOSTICS affected = ROW_COUNT;
        IF affected <> 1 THEN
            RAISE EXCEPTION 'accepted relist target disappeared after successful CAS'
                USING ERRCODE = '55000';
        END IF;
    ELSIF canonical_action = 'insert' THEN
        EXECUTE pg_catalog.format(
            'INSERT INTO public.jobs (%1$s) SELECT %1$s FROM pg_catalog.jsonb_populate_record(NULL::public.jobs, $1) AS populated ON CONFLICT DO NOTHING',
            write_columns
        ) USING canonical_payload;
        GET DIAGNOSTICS affected = ROW_COUNT;
    ELSE
        EXECUTE pg_catalog.format(
            'UPDATE public.jobs AS target SET (%1$s) = (SELECT %1$s FROM pg_catalog.jsonb_populate_record(NULL::public.jobs, $1) AS populated) WHERE target.job_id = $2 AND target.last_seen_at IS NOT DISTINCT FROM $3',
            write_columns
        ) USING canonical_payload, target_job_id, expected_last_seen_at;
        GET DIAGNOSTICS affected = ROW_COUNT;
    END IF;
    IF canonical_action <> 'accepted_relist' AND affected <> 1 THEN
        RETURN pg_catalog.jsonb_build_object(
            'outcome', 'stale_plan', 'task_id', task_row.id,
            'canonical_job_id', target_job_id, 'action', canonical_action,
            'application_hash', application_hash, 'completed_at', NULL,
            'replayed', false
        );
    END IF;

    IF canonical_action <> 'accepted_relist' AND content_version IS NOT NULL THEN
        INSERT INTO public.listing_content_versions AS version (
            provider, source_job_id, content_hash, canonical_job_id, description,
            description_fingerprint, first_observed_at, last_observed_at,
            last_ingestion_run_id, observation_count
        ) VALUES (
            'linkedin', task_source_job_id, content_version->>'content_hash', target_job_id,
            content_version->>'description', content_version->>'description_fingerprint',
            content_observed_at, content_observed_at, task_ingestion_run_id, 1
        ) ON CONFLICT (provider, source_job_id, content_hash) DO UPDATE SET
            canonical_job_id = EXCLUDED.canonical_job_id,
            description = EXCLUDED.description,
            description_fingerprint = EXCLUDED.description_fingerprint,
            last_observed_at = GREATEST(version.last_observed_at, EXCLUDED.last_observed_at),
            observation_count = CASE
                WHEN version.last_ingestion_run_id IS DISTINCT FROM EXCLUDED.last_ingestion_run_id
                THEN version.observation_count + 1 ELSE version.observation_count END,
            last_ingestion_run_id = EXCLUDED.last_ingestion_run_id
        WHERE version.canonical_job_id IS NULL
           OR version.canonical_job_id = EXCLUDED.canonical_job_id;
        GET DIAGNOSTICS affected = ROW_COUNT;
        IF affected <> 1 THEN
            RAISE EXCEPTION 'listing content canonical mapping conflict' USING ERRCODE = '23000';
        END IF;
    END IF;

    application_observed_at := COALESCE(
        GREATEST(content_observed_at, memberships_observed_at, task_row.latest_observed_at),
        content_observed_at,
        memberships_observed_at,
        task_row.latest_observed_at
    );
    INSERT INTO public.listing_states AS state (
        provider, source_job_id, canonical_job_id, first_seen_at, last_seen_at,
        latest_trusted_posted_date, current_content_hash
    ) VALUES (
        'linkedin', task_source_job_id, target_job_id, task_row.first_observed_at,
        GREATEST(task_row.latest_observed_at, application_observed_at), task_row.posted_at,
        content_version->>'content_hash'
    ) ON CONFLICT (provider, source_job_id) DO UPDATE SET
        canonical_job_id = EXCLUDED.canonical_job_id,
        first_seen_at = LEAST(state.first_seen_at, EXCLUDED.first_seen_at),
        last_seen_at = COALESCE(
            GREATEST(state.last_seen_at, EXCLUDED.last_seen_at, prior_state_last_seen_at),
            state.last_seen_at,
            EXCLUDED.last_seen_at,
            prior_state_last_seen_at
        ),
        latest_trusted_posted_date = COALESCE(
            GREATEST(
                state.latest_trusted_posted_date,
                EXCLUDED.latest_trusted_posted_date,
                prior_state_latest_posted_date
            ),
            state.latest_trusted_posted_date,
            EXCLUDED.latest_trusted_posted_date,
            prior_state_latest_posted_date
        ),
        current_content_hash = CASE
            WHEN prior_state_content_hash IS NULL
              OR application_observed_at >= prior_state_last_seen_at
            THEN COALESCE(EXCLUDED.current_content_hash, state.current_content_hash)
            ELSE prior_state_content_hash
        END,
        updated_at = pg_catalog.clock_timestamp();

    FOR membership IN
        SELECT value
        FROM pg_catalog.jsonb_array_elements(memberships)
        ORDER BY value::text
    LOOP
        membership_first_matched_at := (membership->>'first_matched_at')::timestamptz;
        membership_last_matched_at := (membership->>'last_matched_at')::timestamptz;
        provenance := pg_catalog.jsonb_strip_nulls(
            membership->'query_scope'
            || pg_catalog.jsonb_build_object(
                'lane', membership->>'archetype',
                'archetype', membership->>'archetype',
                'query_id', membership->>'query_id',
                'query', membership->>'query',
                'query_type', membership->>'query_type',
                'language', membership->>'language',
                'location_scope', membership->>'location_scope',
                'geography_id', membership->>'geography_id',
                'observed_at', pg_catalog.to_jsonb(membership_last_matched_at)
            )
        );
        INSERT INTO public.job_archetype_memberships AS lane_membership (
            job_id, archetype, matched_queries, first_matched_at, last_matched_at,
            filter_status, is_filtered, filter_reason, insights
        ) VALUES (
            target_job_id, membership->>'archetype', pg_catalog.jsonb_build_array(provenance),
            membership_first_matched_at, membership_last_matched_at,
            membership->>'filter_status', (membership->>'is_filtered')::boolean,
            membership->>'filter_reason',
            pg_catalog.jsonb_build_object(
                'matched_queries', pg_catalog.jsonb_build_array(provenance),
                'matched_query_provenance', pg_catalog.jsonb_build_array(provenance),
                'query_scopes', pg_catalog.jsonb_build_array(provenance),
                'last_matched_at', pg_catalog.to_jsonb(membership_last_matched_at)
            )
        ) ON CONFLICT (job_id, archetype) DO UPDATE SET
            matched_queries = (
                SELECT COALESCE(pg_catalog.jsonb_agg(item.value ORDER BY item.value::text), '[]'::jsonb)
                FROM (
                    SELECT DISTINCT value
                    FROM pg_catalog.jsonb_array_elements(lane_membership.matched_queries || EXCLUDED.matched_queries)
                ) AS item
            ),
            first_matched_at = LEAST(lane_membership.first_matched_at, EXCLUDED.first_matched_at),
            last_matched_at = GREATEST(lane_membership.last_matched_at, EXCLUDED.last_matched_at),
            filter_status = EXCLUDED.filter_status,
            is_filtered = EXCLUDED.is_filtered,
            filter_reason = EXCLUDED.filter_reason,
            insights = lane_membership.insights || pg_catalog.jsonb_build_object(
                'matched_queries', (
                    SELECT COALESCE(pg_catalog.jsonb_agg(item.value ORDER BY item.value::text), '[]'::jsonb)
                    FROM (
                        SELECT DISTINCT value
                        FROM pg_catalog.jsonb_array_elements(lane_membership.matched_queries || EXCLUDED.matched_queries)
                    ) AS item
                ),
                'matched_query_provenance', (
                    SELECT COALESCE(pg_catalog.jsonb_agg(item.value ORDER BY item.value::text), '[]'::jsonb)
                    FROM (
                        SELECT DISTINCT value
                        FROM pg_catalog.jsonb_array_elements(lane_membership.matched_queries || EXCLUDED.matched_queries)
                    ) AS item
                ),
                'query_scopes', (
                    SELECT COALESCE(pg_catalog.jsonb_agg(item.value ORDER BY item.value::text), '[]'::jsonb)
                    FROM (
                        SELECT DISTINCT value
                        FROM pg_catalog.jsonb_array_elements(lane_membership.matched_queries || EXCLUDED.matched_queries)
                    ) AS item
                ),
                'last_matched_at', pg_catalog.to_jsonb(
                    GREATEST(lane_membership.last_matched_at, EXCLUDED.last_matched_at)
                )
            ),
            updated_at = pg_catalog.clock_timestamp();
    END LOOP;

    SELECT job.canonical_revision INTO STRICT applied_canonical_revision
    FROM public.jobs job
    WHERE job.job_id = target_job_id;
    current_candidate_set_revision := public.get_canonical_provider_revision('linkedin');

    application_completed_at := pg_catalog.clock_timestamp();
    UPDATE public.linkedin_discovery_tasks
    SET status = 'complete', canonical_job_id = target_job_id,
        canonical_applied_lease_token = p_lease_token,
        canonical_application_hash = application_hash,
        last_error_code = NULL, completed_at = application_completed_at,
        leased_by = NULL, leased_at = NULL, lease_expires_at = NULL, lease_token = NULL
    WHERE id = p_task_id AND status = 'leased' AND leased_by = p_worker_id
      AND lease_token = p_lease_token AND lease_expires_at > pg_catalog.clock_timestamp();
    IF NOT FOUND THEN
        RAISE EXCEPTION 'adaptive discovery task lease expired during canonical publication'
            USING ERRCODE = '55000';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'outcome', 'applied', 'task_id', p_task_id,
        'canonical_job_id', target_job_id, 'action', canonical_action,
        'canonical_revision', applied_canonical_revision,
        'provider_candidate_set_revision', current_candidate_set_revision,
        'application_hash', application_hash, 'completed_at', application_completed_at,
        'replayed', false
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.transition_linkedin_discovery_task(
    p_task_id bigint, p_worker_id text, p_lease_token uuid, p_status text,
    p_canonical_job_id text, p_error_code text
) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog AS $$
DECLARE
    changed public.linkedin_discovery_tasks%ROWTYPE;
    next_status text := p_status;
    terminal_evidence integer;
    task_request_prefix text;
BEGIN
    IF p_status NOT IN ('terminal_unavailable', 'failed_retryable', 'failed_terminal') THEN
        RAISE EXCEPTION 'invalid task transition' USING ERRCODE = '22023';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('linkedin-canonical-publication-v1', 0));
    IF p_status = 'terminal_unavailable' THEN
        task_request_prefix := 'task:' || p_task_id::text || ':' || p_lease_token::text || ':';
        IF p_error_code = 'source_unavailable_404_confirmed' THEN
            SELECT COUNT(*) INTO terminal_evidence
            FROM public.linkedin_source_request_grants request_grant
            WHERE request_grant.producer = 'adaptive-detail'
              AND request_grant.request_kind = 'detail'
              AND pg_catalog.starts_with(request_grant.request_key, task_request_prefix)
              AND request_grant.status = 'finished'
              AND request_grant.http_status = 404
              AND request_grant.response_class IN ('not_found_unconfirmed', 'terminal_unavailable');
            IF terminal_evidence < 2 OR NOT EXISTS (
                SELECT 1 FROM public.linkedin_source_request_grants request_grant
                WHERE request_grant.producer = 'adaptive-detail'
                  AND request_grant.request_kind = 'detail'
                  AND pg_catalog.starts_with(request_grant.request_key, task_request_prefix)
                  AND request_grant.status = 'finished'
                  AND request_grant.http_status = 404
                  AND request_grant.response_class = 'not_found_unconfirmed'
            ) OR NOT EXISTS (
                SELECT 1 FROM public.linkedin_source_request_grants request_grant
                WHERE request_grant.producer = 'adaptive-detail'
                  AND request_grant.request_kind = 'detail'
                  AND pg_catalog.starts_with(request_grant.request_key, task_request_prefix)
                  AND request_grant.status = 'finished'
                  AND request_grant.http_status = 404
                  AND request_grant.response_class = 'terminal_unavailable'
            ) THEN
                RAISE EXCEPTION 'confirmed 404 transition lacks durable request evidence' USING ERRCODE = '55000';
            END IF;
        ELSIF p_error_code = 'source_unavailable_410' THEN
            IF NOT EXISTS (
                SELECT 1 FROM public.linkedin_source_request_grants request_grant
                WHERE request_grant.producer = 'adaptive-detail'
                  AND request_grant.request_kind = 'detail'
                  AND pg_catalog.starts_with(request_grant.request_key, task_request_prefix)
                  AND request_grant.status = 'finished'
                  AND request_grant.http_status = 410
                  AND request_grant.response_class = 'terminal_unavailable'
            ) THEN
                RAISE EXCEPTION '410 transition lacks durable request evidence' USING ERRCODE = '55000';
            END IF;
        ELSE
            RAISE EXCEPTION 'terminal-unavailable transition lacks confirmed evidence' USING ERRCODE = '22023';
        END IF;
    END IF;
    IF p_status = 'failed_retryable' AND EXISTS (
        SELECT 1 FROM public.linkedin_discovery_tasks
        WHERE id = p_task_id AND attempt_count >= max_attempts
    ) THEN
        next_status := 'failed_terminal';
    END IF;
    UPDATE public.linkedin_discovery_tasks SET
        status = next_status,
        last_error_code = p_error_code,
        completed_at = CASE WHEN next_status IN ('terminal_unavailable', 'failed_terminal') THEN pg_catalog.clock_timestamp() ELSE NULL END,
        available_at = CASE WHEN next_status = 'failed_retryable' THEN pg_catalog.clock_timestamp() + pg_catalog.make_interval(secs => LEAST(3600, 30 * pg_catalog.power(2, LEAST(attempt_count, 6)))) ELSE available_at END,
        leased_by = NULL, leased_at = NULL, lease_expires_at = NULL, lease_token = NULL
    WHERE id = p_task_id AND status = 'leased' AND leased_by = p_worker_id
      AND lease_token = p_lease_token AND lease_expires_at > pg_catalog.clock_timestamp()
    RETURNING * INTO changed;
    IF changed.id IS NULL THEN RAISE EXCEPTION 'task lease lost or transition rejected' USING ERRCODE = '40001'; END IF;
    RETURN pg_catalog.jsonb_build_object('task_id', changed.id, 'status', changed.status, 'canonical_job_id', changed.canonical_job_id);
END;
$$;

CREATE OR REPLACE FUNCTION public.finalize_freehire_publication_v2(p_cycle_id bigint)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog SET statement_timeout = '5min' AS $$
DECLARE
    cycle_row public.linkedin_discovery_cycles%ROWTYPE;
    current_publication public.freehire_publication_state%ROWTYPE;
    publication record;
    blocking_count bigint;
    predecessor_count bigint;
    source_watermark timestamptz;
    source_sequence bigint;
    source_cycle_id bigint;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('linkedin-canonical-publication-v1', 0));
    SELECT * INTO STRICT cycle_row FROM public.linkedin_discovery_cycles WHERE id = p_cycle_id FOR SHARE;
    IF cycle_row.search_status <> 'sealed' THEN
        SELECT COUNT(*) INTO blocking_count
        FROM public.linkedin_discovery_cycle_scopes scope
        JOIN public.ingestion_runs run ON run.id = scope.ingestion_run_id
        WHERE scope.discovery_cycle_id = p_cycle_id
          AND run.coverage_status <> 'exhausted';
        RETURN pg_catalog.jsonb_build_object(
            'outcome', 'deferred', 'reason', 'coverage work remains',
            'requested_cycle_id', p_cycle_id, 'eligible_cycle_id', NULL,
            'blocking_count', blocking_count
        );
    END IF;
    SELECT * INTO STRICT current_publication
    FROM public.freehire_publication_state WHERE id = 1 FOR UPDATE;
    IF current_publication.source_discovery_sequence > cycle_row.discovery_sequence THEN
        RAISE EXCEPTION 'requested discovery cycle is stale' USING ERRCODE = '55000';
    ELSIF current_publication.source_discovery_sequence = cycle_row.discovery_sequence THEN
        RETURN pg_catalog.jsonb_build_object(
            'outcome', 'unchanged', 'reason', 'discovery cycle already published',
            'requested_cycle_id', p_cycle_id, 'eligible_cycle_id', p_cycle_id,
            'generation', current_publication.generation,
            'published_at', current_publication.published_at,
            'source_scrape_watermark', current_publication.source_scrape_watermark,
            'source_discovery_sequence', current_publication.source_discovery_sequence,
            'row_count', current_publication.row_count,
            'schema_version', current_publication.schema_version
        );
    END IF;
    SELECT COUNT(*) INTO predecessor_count
    FROM public.linkedin_discovery_cycles predecessor
    WHERE predecessor.discovery_sequence <= cycle_row.discovery_sequence
      AND predecessor.search_status <> 'sealed'
      AND NOT EXISTS (
          SELECT 1 FROM public.linkedin_discovery_cycle_resolutions resolution
          JOIN public.linkedin_discovery_cycles recovery
            ON recovery.id = resolution.resolving_discovery_cycle_id
          WHERE resolution.failed_discovery_cycle_id = predecessor.id
            AND recovery.search_status = 'sealed'
            AND recovery.discovery_sequence <= cycle_row.discovery_sequence
      );
    IF predecessor_count > 0 THEN
        RETURN pg_catalog.jsonb_build_object(
            'outcome', 'deferred', 'reason', 'predecessor discovery cycle is incomplete',
            'requested_cycle_id', p_cycle_id, 'eligible_cycle_id', NULL,
            'blocking_count', predecessor_count
        );
    END IF;
    SELECT COUNT(*) INTO blocking_count
    FROM public.linkedin_coverage_debt debt
    JOIN public.linkedin_discovery_cycles origin
      ON origin.id = debt.origin_discovery_cycle_id
    WHERE origin.discovery_sequence <= cycle_row.discovery_sequence
      AND debt.status IN ('pending', 'expired_unresolved');
    IF blocking_count > 0 THEN
        RETURN pg_catalog.jsonb_build_object(
            'outcome', 'deferred', 'reason', 'unresolved coverage debt',
            'requested_cycle_id', p_cycle_id, 'eligible_cycle_id', NULL,
            'blocking_count', blocking_count
        );
    END IF;
    SELECT COUNT(*) INTO blocking_count
    FROM public.linkedin_discovery_requirements requirement
    JOIN public.linkedin_discovery_cycles requirement_cycle
      ON requirement_cycle.id = requirement.discovery_cycle_id
    JOIN public.linkedin_discovery_tasks task ON task.id = requirement.task_id
    LEFT JOIN public.listing_states state ON state.provider = task.provider AND state.source_job_id = task.source_job_id
    LEFT JOIN public.linkedin_discovery_requirement_acceptances acceptance
      ON acceptance.discovery_cycle_id = requirement.discovery_cycle_id
     AND acceptance.ingestion_run_id = requirement.ingestion_run_id
     AND acceptance.provider = requirement.provider
     AND acceptance.source_job_id = requirement.source_job_id
     AND acceptance.task_kind = requirement.task_kind
     AND acceptance.requirement_key = requirement.requirement_key
    WHERE requirement_cycle.discovery_sequence <= cycle_row.discovery_sequence
      AND requirement.required
      AND acceptance.discovery_cycle_id IS NULL
      AND NOT (
        task.status = 'terminal_unavailable'
        OR (task.status = 'complete' AND task.canonical_job_id IS NOT NULL AND state.canonical_job_id = task.canonical_job_id)
      );
    IF blocking_count > 0 THEN
        RETURN pg_catalog.jsonb_build_object(
            'outcome', 'deferred', 'reason', 'unresolved discovery tasks',
            'requested_cycle_id', p_cycle_id, 'eligible_cycle_id', NULL,
            'blocking_count', blocking_count
        );
    END IF;
    SELECT last_successful_scrape_at, last_successful_discovery_sequence,
           last_successful_discovery_cycle_id
    INTO source_watermark, source_sequence, source_cycle_id
    FROM public.scrape_run_state WHERE id = 1 FOR SHARE;
    IF source_watermark IS NULL OR source_sequence IS NULL
       OR source_sequence < cycle_row.discovery_sequence THEN
        RETURN pg_catalog.jsonb_build_object(
            'outcome', 'deferred', 'reason', 'operational discovery watermark is not eligible',
            'requested_cycle_id', p_cycle_id, 'eligible_cycle_id', NULL,
            'blocking_count', 1
        );
    ELSIF source_sequence > cycle_row.discovery_sequence THEN
        RAISE EXCEPTION 'requested discovery cycle is older than the operational watermark'
            USING ERRCODE = '55000';
    ELSIF source_cycle_id IS DISTINCT FROM p_cycle_id THEN
        RAISE EXCEPTION 'operational watermark cycle does not match requested cycle'
            USING ERRCODE = '55000';
    END IF;
    UPDATE public.linkedin_discovery_cycles applied
    SET canonical_status = 'applied'
    WHERE applied.discovery_sequence <= cycle_row.discovery_sequence
      AND applied.search_status = 'sealed';
    SELECT * INTO publication FROM public.finalize_freehire_publication(source_watermark);
    UPDATE public.freehire_publication_state SET source_discovery_cycle_id = p_cycle_id,
        source_discovery_sequence = cycle_row.discovery_sequence WHERE id = 1;
    UPDATE public.freehire_publication_generations SET source_discovery_cycle_id = p_cycle_id,
        source_discovery_sequence = cycle_row.discovery_sequence WHERE generation = publication.generation;
    RETURN pg_catalog.jsonb_build_object(
        'outcome', 'published', 'reason', NULL, 'requested_cycle_id', p_cycle_id,
        'eligible_cycle_id', p_cycle_id,
        'generation', publication.generation, 'published_at', publication.published_at,
        'source_scrape_watermark', publication.source_scrape_watermark,
        'source_discovery_sequence', cycle_row.discovery_sequence,
        'row_count', publication.row_count, 'schema_version', publication.schema_version
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.get_linkedin_discovery_status()
RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog AS $$
WITH latest AS (
    SELECT cycle.*
    FROM public.linkedin_discovery_cycles cycle
    ORDER BY cycle.discovery_sequence DESC
    LIMIT 1
), scope_summary AS (
    SELECT COUNT(*) AS scopes,
           COUNT(*) FILTER (WHERE run.coverage_status = 'exhausted') AS exhausted,
           COUNT(*) FILTER (WHERE scope.status = 'running') AS running,
           COALESCE(SUM(page.pages), 0) AS pages,
           COALESCE(SUM(page.cards), 0) AS cards
    FROM latest
    LEFT JOIN public.linkedin_discovery_cycle_scopes scope
      ON scope.discovery_cycle_id = latest.id
    LEFT JOIN public.ingestion_runs run ON run.id = scope.ingestion_run_id
    LEFT JOIN (
        SELECT ingestion_run_id, COUNT(*) AS pages, SUM(card_count) AS cards
        FROM public.linkedin_ingestion_pages
        GROUP BY ingestion_run_id
    ) page ON page.ingestion_run_id = scope.ingestion_run_id
), debt_summary AS (
    SELECT COUNT(*) FILTER (WHERE debt.status = 'pending') AS pending,
           COUNT(*) FILTER (WHERE debt.status = 'expired_unresolved') AS expired,
           MIN(debt.created_at) FILTER (
               WHERE debt.status IN ('pending', 'expired_unresolved')
           ) AS oldest
    FROM public.linkedin_coverage_debt debt
), task_summary AS (
    SELECT COUNT(*) FILTER (WHERE task.status = 'pending') AS pending,
           COUNT(*) FILTER (WHERE task.status = 'leased') AS leased,
           COUNT(*) FILTER (WHERE task.status = 'failed_retryable') AS retryable,
           COUNT(*) FILTER (WHERE task.status = 'failed_terminal') AS terminal,
           COUNT(*) FILTER (WHERE task.status = 'complete') AS complete
    FROM public.linkedin_discovery_tasks task
), lane_summary AS (
    SELECT COALESCE(pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
        'archetype', grouped.archetype,
        'scopes', grouped.scopes,
        'exhausted', grouped.exhausted,
        'running', grouped.running,
        'pages', grouped.pages,
        'cards', grouped.cards
    ) ORDER BY grouped.archetype), '[]'::jsonb) AS lanes
    FROM (
        SELECT state.archetype, COUNT(*) AS scopes,
               COUNT(*) FILTER (WHERE run.coverage_status = 'exhausted') AS exhausted,
               COUNT(*) FILTER (WHERE scope.status = 'running') AS running,
               COALESCE(SUM(page.pages), 0) AS pages,
               COALESCE(SUM(page.cards), 0) AS cards
        FROM latest
        JOIN public.linkedin_discovery_cycle_scopes scope
          ON scope.discovery_cycle_id = latest.id
        JOIN public.linkedin_scope_coverage_state state
          ON state.scope_key = scope.scope_key
        JOIN public.ingestion_runs run ON run.id = scope.ingestion_run_id
        LEFT JOIN (
            SELECT ingestion_run_id, COUNT(*) AS pages, SUM(card_count) AS cards
            FROM public.linkedin_ingestion_pages
            GROUP BY ingestion_run_id
        ) page ON page.ingestion_run_id = scope.ingestion_run_id
        GROUP BY state.archetype
    ) grouped
)
SELECT pg_catalog.jsonb_build_object(
    'latest_cycle', CASE WHEN latest.id IS NULL THEN NULL ELSE pg_catalog.jsonb_build_object(
        'id', latest.id,
        'sequence', latest.discovery_sequence,
        'started_at', latest.started_at,
        'completed_at', latest.search_completed_at,
        'search_status', latest.search_status,
        'canonical_status', latest.canonical_status,
        'required_scopes', latest.required_scope_count,
        'completed_scopes', scope_summary.exhausted,
        'running_scopes', scope_summary.running,
        'pages', scope_summary.pages,
        'cards', scope_summary.cards
    ) END,
    'coverage_debt', pg_catalog.jsonb_build_object(
        'pending', debt_summary.pending,
        'expired', debt_summary.expired,
        'oldest_at', debt_summary.oldest
    ),
    'tasks', to_jsonb(task_summary),
    'publication', COALESCE((
        SELECT to_jsonb(publication) FROM public.freehire_publication_state publication
        WHERE publication.id = 1
    ), '{}'::jsonb),
    'lanes', lane_summary.lanes
)
FROM scope_summary
CROSS JOIN debt_summary
CROSS JOIN task_summary
CROSS JOIN lane_summary
LEFT JOIN latest ON true;
$$;

ALTER TABLE public.linkedin_source_request_policy ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.canonical_provider_revisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_source_request_grants ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_scope_coverage_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_discovery_cycles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_discovery_cycle_scopes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_ingestion_pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_ingestion_page_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_discovery_cycle_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_discovery_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_discovery_requirements ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_coverage_debt ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_discovery_cycle_resolutions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_coverage_debt_attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_discovery_requirement_acceptances ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.linkedin_discovery_task_attempts ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON TABLE public.canonical_provider_revisions,
    public.linkedin_source_request_policy, public.linkedin_source_request_grants,
    public.linkedin_scope_coverage_state, public.linkedin_discovery_cycles,
    public.linkedin_discovery_cycle_scopes, public.linkedin_ingestion_pages,
    public.linkedin_ingestion_page_sources, public.linkedin_discovery_cycle_sources,
    public.linkedin_discovery_tasks, public.linkedin_discovery_requirements,
    public.linkedin_coverage_debt, public.linkedin_discovery_cycle_resolutions,
    public.linkedin_coverage_debt_attempts,
    public.linkedin_discovery_requirement_acceptances,
    public.linkedin_discovery_task_attempts
FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON SEQUENCE public.linkedin_discovery_cycles_id_seq,
    public.linkedin_discovery_tasks_id_seq,
    public.linkedin_coverage_debt_id_seq,
    public.linkedin_discovery_task_attempts_id_seq
FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT ON TABLE public.canonical_provider_revisions,
    public.linkedin_source_request_policy, public.linkedin_source_request_grants,
    public.linkedin_scope_coverage_state, public.linkedin_discovery_cycles,
    public.linkedin_discovery_cycle_scopes, public.linkedin_ingestion_pages,
    public.linkedin_ingestion_page_sources, public.linkedin_discovery_cycle_sources,
    public.linkedin_discovery_tasks, public.linkedin_discovery_requirements,
    public.linkedin_coverage_debt, public.linkedin_discovery_cycle_resolutions,
    public.linkedin_coverage_debt_attempts,
    public.linkedin_discovery_requirement_acceptances,
    public.linkedin_discovery_task_attempts
TO service_role;

REVOKE ALL ON FUNCTION public.bump_canonical_provider_revision(),
    public.get_canonical_provider_revision(text),
    public.acquire_linkedin_request_grant(text, text, text),
    public.consume_linkedin_request_grant(uuid, text),
    public.finish_linkedin_request_grant(uuid, text, text, integer),
    public.open_linkedin_source_circuit(uuid, text, text, integer),
    public.reset_linkedin_source_circuit(text, text),
    public.create_linkedin_discovery_cycle(uuid, bigint, text, text, jsonb),
    public.get_resumable_linkedin_discovery_cycle(boolean, text[]),
    public.commit_linkedin_discovery_page(jsonb),
    public.finish_linkedin_discovery_scope(uuid, text, text),
    public.fail_linkedin_discovery_cycle(bigint, text),
    public.expire_linkedin_coverage_debt(text, timestamptz),
    public.prepare_linkedin_discovery_scope_state(text[], timestamptz),
    public.accept_linkedin_coverage_debt(bigint, text, text),
    public.advance_linkedin_discovery_watermark(),
    public.resolve_failed_linkedin_discovery_cycle(bigint, bigint, text, text, text),
    public.resolve_eligible_failed_linkedin_discovery_cycles(bigint),
    public.accept_linkedin_discovery_requirement(bigint, uuid, text, text, text, text, text, text),
    public.seal_linkedin_discovery_cycle(bigint, boolean),
    public.claim_linkedin_discovery_tasks(text, integer, text),
    public.heartbeat_linkedin_discovery_task(bigint, text, uuid),
    public.transition_linkedin_discovery_task(bigint, text, uuid, text, text, text),
    public.finalize_freehire_publication_v2(bigint),
    public.get_linkedin_discovery_status()
FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.get_canonical_provider_revision(text),
    public.acquire_linkedin_request_grant(text, text, text),
    public.consume_linkedin_request_grant(uuid, text),
    public.finish_linkedin_request_grant(uuid, text, text, integer),
    public.open_linkedin_source_circuit(uuid, text, text, integer),
    public.reset_linkedin_source_circuit(text, text),
    public.create_linkedin_discovery_cycle(uuid, bigint, text, text, jsonb),
    public.get_resumable_linkedin_discovery_cycle(boolean, text[]),
    public.commit_linkedin_discovery_page(jsonb),
    public.finish_linkedin_discovery_scope(uuid, text, text),
    public.fail_linkedin_discovery_cycle(bigint, text),
    public.expire_linkedin_coverage_debt(text, timestamptz),
    public.prepare_linkedin_discovery_scope_state(text[], timestamptz),
    public.accept_linkedin_coverage_debt(bigint, text, text),
    public.advance_linkedin_discovery_watermark(),
    public.resolve_failed_linkedin_discovery_cycle(bigint, bigint, text, text, text),
    public.resolve_eligible_failed_linkedin_discovery_cycles(bigint),
    public.accept_linkedin_discovery_requirement(bigint, uuid, text, text, text, text, text, text),
    public.seal_linkedin_discovery_cycle(bigint, boolean),
    public.claim_linkedin_discovery_tasks(text, integer, text),
    public.heartbeat_linkedin_discovery_task(bigint, text, uuid),
    public.transition_linkedin_discovery_task(bigint, text, uuid, text, text, text),
    public.finalize_freehire_publication_v2(bigint),
    public.get_linkedin_discovery_status()
TO service_role;
REVOKE ALL ON FUNCTION public.apply_linkedin_discovery_task_canonical(bigint, text, uuid, jsonb)
FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.apply_linkedin_discovery_task_canonical(bigint, text, uuid, jsonb)
TO service_role;

REVOKE EXECUTE ON FUNCTION public.record_scrape_success(timestamptz),
    public.finalize_freehire_publication(timestamptz)
FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL PRIVILEGES ON TABLE public.scrape_run_state FROM service_role;
GRANT SELECT ON TABLE public.scrape_run_state TO service_role;

COMMIT;
