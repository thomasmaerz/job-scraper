-- Correct live progress and publication blocker counts.
BEGIN;

SET LOCAL ROLE postgres;
SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '15min';

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

COMMIT;
