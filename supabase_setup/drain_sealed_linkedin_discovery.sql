-- Drain canonical work for the latest sealed cycle before starting another search.
BEGIN;

SET LOCAL ROLE postgres;
SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '15min';

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

COMMIT;
