-- Forward migration for resumable, terminal-evidence LinkedIn discovery.
BEGIN;

SET LOCAL ROLE postgres;
SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '15min';

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
    WHERE cycle.search_status = 'running'
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
    ORDER BY cycle.discovery_sequence
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
        RETURN pg_catalog.jsonb_build_object(
            'outcome', 'deferred', 'reason', 'coverage work remains',
            'requested_cycle_id', p_cycle_id, 'eligible_cycle_id', NULL,
            'blocking_count', cycle_row.required_scope_count - cycle_row.completed_scope_count
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
           COALESCE(SUM(run.pages_completed), 0) AS pages,
           COALESCE(SUM(run.cards_seen), 0) AS cards
    FROM latest
    LEFT JOIN public.linkedin_discovery_cycle_scopes scope
      ON scope.discovery_cycle_id = latest.id
    LEFT JOIN public.ingestion_runs run ON run.id = scope.ingestion_run_id
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
               COALESCE(SUM(run.pages_completed), 0) AS pages,
               COALESCE(SUM(run.cards_seen), 0) AS cards
        FROM latest
        JOIN public.linkedin_discovery_cycle_scopes scope
          ON scope.discovery_cycle_id = latest.id
        JOIN public.linkedin_scope_coverage_state state
          ON state.scope_key = scope.scope_key
        JOIN public.ingestion_runs run ON run.id = scope.ingestion_run_id
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

ALTER TABLE public.canonical_provider_revisions ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.canonical_provider_revisions FROM PUBLIC, anon, authenticated;
GRANT SELECT ON TABLE public.canonical_provider_revisions TO service_role;
REVOKE ALL ON FUNCTION public.bump_canonical_provider_revision(),
    public.get_canonical_provider_revision(text),
    public.get_resumable_linkedin_discovery_cycle(boolean, text[]),
    public.prepare_linkedin_discovery_scope_state(text[], timestamptz),
    public.get_linkedin_discovery_status()
FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.get_canonical_provider_revision(text),
    public.get_resumable_linkedin_discovery_cycle(boolean, text[]),
    public.prepare_linkedin_discovery_scope_state(text[], timestamptz),
    public.get_linkedin_discovery_status()
TO service_role;

DO $lane_filters$
BEGIN
    IF pg_catalog.to_regclass('public.career_lane_definitions') IS NOT NULL THEN
        UPDATE public.career_lane_definitions lane
        SET title_exclude = ARRAY(
            SELECT DISTINCT value
            FROM pg_catalog.unnest(lane.title_exclude || CASE lane.archetype
                WHEN 'technology_delivery' THEN ARRAY['\bprocess project manager\b']
                WHEN 'systems_platform_ops' THEN ARRAY['\bbackend (?:software )?engineer\b']
                WHEN 'network_infrastructure' THEN ARRAY['\bnetwork software engineer\b']
                WHEN 'datacenter_operations' THEN ARRAY['\bsoftware engineer\b']
                WHEN 'ai_workflow_automation' THEN ARRAY['\bsoftware (?:developer|engineer)\b.*\bbrokerage\b']
                WHEN 'building_controls' THEN ARRAY[
                    '\bmaintenance electrician\b',
                    '\bdata scientist\b',
                    '\brobotics engineer\b'
                ]
                ELSE ARRAY[]::text[]
            END) AS value
            ORDER BY value
        )
        WHERE lane.archetype IN (
            'technology_delivery', 'systems_platform_ops',
            'network_infrastructure', 'datacenter_operations',
            'ai_workflow_automation', 'building_controls'
        );
    END IF;
END;
$lane_filters$;

DO $membership_filters$
BEGIN
    IF pg_catalog.to_regclass('public.job_archetype_memberships') IS NOT NULL THEN
        UPDATE public.job_archetype_memberships membership
        SET filter_status = 'filtered',
            is_filtered = true,
            filter_reason = 'title:reviewed_cross_lane_false_positive',
            updated_at = pg_catalog.clock_timestamp()
        FROM public.jobs job
        WHERE job.job_id = membership.job_id
          AND CASE membership.archetype
              WHEN 'technology_delivery' THEN job.job_title ~* '\mprocess project manager\M'
              WHEN 'systems_platform_ops' THEN job.job_title ~* '\mbackend (software )?engineer\M'
              WHEN 'network_infrastructure' THEN job.job_title ~* '\mnetwork software engineer\M'
              WHEN 'datacenter_operations' THEN job.job_title ~* '\msoftware engineer\M'
              WHEN 'ai_workflow_automation' THEN job.job_title ~* '\msoftware (developer|engineer)\M.*\mbrokerage\M'
              WHEN 'building_controls' THEN job.job_title ~* '\m(maintenance electrician|data scientist|robotics engineer)\M'
              ELSE false
          END;
    END IF;
END;
$membership_filters$;

COMMIT;
