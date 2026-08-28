-- Establish an exact aggregate baseline, then enable atomic incremental writes.
-- Safe to run repeatedly on an existing Supabase database.

BEGIN;

SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '15min';
SET LOCAL idle_in_transaction_session_timeout = '16min';

-- This lock is the protocol shared by baseline rebuilds and incremental writes.
-- Taking it before the baseline makes the lock transaction-wide for migration.
SELECT pg_advisory_xact_lock(hashtextextended('keyword-insights-aggregate-global', 0));
LOCK TABLE public.job_keyword_insights, public.keyword_insights
IN ACCESS EXCLUSIVE MODE;

-- Normalize grants left by older schema exports before installing the only
-- supported mutation path (the SECURITY DEFINER RPC below).
REVOKE ALL PRIVILEGES ON TABLE public.keyword_insights, public.job_keyword_insights
FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT ON TABLE public.keyword_insights, public.job_keyword_insights
TO anon, authenticated, service_role;

CREATE OR REPLACE FUNCTION public.rebuild_keyword_insights_atomic()
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    rebuilt_count integer;
BEGIN
    PERFORM pg_advisory_xact_lock(
        hashtextextended('keyword-insights-aggregate-global', 0)
    );

    DELETE FROM public.keyword_insights;

    INSERT INTO public.keyword_insights (
        archetype, provider, keyword, category, count, last_updated
    )
    SELECT
        archetype,
        COALESCE(provider, 'unknown'),
        keyword,
        category,
        count(*)::integer,
        now()
    FROM public.job_keyword_insights
    GROUP BY archetype, COALESCE(provider, 'unknown'), keyword, category;

    GET DIAGNOSTICS rebuilt_count = ROW_COUNT;
    RETURN rebuilt_count;
END;
$$;

ALTER FUNCTION public.rebuild_keyword_insights_atomic() OWNER TO postgres;
REVOKE ALL ON FUNCTION public.rebuild_keyword_insights_atomic()
FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.rebuild_keyword_insights_atomic() TO service_role;

-- Remove stale rows as well as rebuilding every archetype/provider/category key.
-- The delta RPC is created only after this exact baseline succeeds.
SELECT public.rebuild_keyword_insights_atomic();

CREATE OR REPLACE FUNCTION public.replace_job_keyword_facts_and_refresh_aggregates(
    p_job_ids text[],
    p_archetype text,
    p_facts jsonb
) RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_delta jsonb;
    v_deltas jsonb;
    v_fact_count integer := 0;
    v_job_id text;
BEGIN
    PERFORM pg_advisory_xact_lock(
        hashtextextended('keyword-insights-aggregate-global', 0)
    );

    IF p_job_ids IS NULL OR cardinality(p_job_ids) = 0 THEN
        IF p_facts IS NULL OR p_facts = '[]'::jsonb THEN
            RETURN 0;
        END IF;
        RAISE EXCEPTION 'p_job_ids must not be empty';
    END IF;
    IF cardinality(p_job_ids) > 1000 THEN
        RAISE EXCEPTION 'p_job_ids exceeds the 1000-job RPC limit';
    END IF;
    IF EXISTS (SELECT 1 FROM unnest(p_job_ids) AS id WHERE id IS NULL OR btrim(id) = '') THEN
        RAISE EXCEPTION 'p_job_ids must contain only non-empty IDs';
    END IF;
    IF p_archetype IS NULL OR btrim(p_archetype) = '' THEN
        RAISE EXCEPTION 'p_archetype must not be empty';
    END IF;
    IF p_facts IS NULL OR jsonb_typeof(p_facts) <> 'array' THEN
        RAISE EXCEPTION 'p_facts must be a JSON array';
    END IF;
    IF jsonb_array_length(p_facts) > 50000 THEN
        RAISE EXCEPTION 'p_facts exceeds the 50000-fact RPC limit';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM jsonb_array_elements(p_facts) AS e(fact)
        WHERE jsonb_typeof(fact) <> 'object'
           OR jsonb_typeof(fact->'job_id') IS DISTINCT FROM 'string'
           OR jsonb_typeof(fact->'archetype') IS DISTINCT FROM 'string'
           OR jsonb_typeof(fact->'keyword') IS DISTINCT FROM 'string'
           OR jsonb_typeof(fact->'category') IS DISTINCT FROM 'string'
           OR (fact ? 'provider' AND fact->'provider' <> 'null'::jsonb
               AND jsonb_typeof(fact->'provider') IS DISTINCT FROM 'string')
           OR NOT ((fact->>'job_id') = ANY (p_job_ids))
           OR fact->>'archetype' IS DISTINCT FROM p_archetype
           OR btrim(fact->>'keyword') = ''
           OR fact->>'category' NOT IN ('skill', 'technology', 'certification', 'attribute')
    ) THEN
        RAISE EXCEPTION 'p_facts contains an invalid or out-of-scope fact';
    END IF;

    -- Serialize replacements of the same job/archetype before reading old facts.
    FOR v_job_id IN
        SELECT DISTINCT id FROM unnest(p_job_ids) AS id ORDER BY id
    LOOP
        PERFORM pg_advisory_xact_lock(
            hashtextextended('job-keyword-facts:' || p_archetype || ':' || v_job_id, 0)
        );
    END LOOP;

    WITH new_facts AS MATERIALIZED (
        SELECT DISTINCT ON (fact->>'job_id', fact->>'archetype', fact->>'keyword', fact->>'category')
            fact->>'job_id' AS job_id,
            fact->>'archetype' AS archetype,
            fact->>'keyword' AS keyword,
            fact->>'category' AS category,
            NULLIF(fact->>'provider', '') AS provider
        FROM jsonb_array_elements(p_facts) WITH ORDINALITY AS e(fact, ordinal)
        ORDER BY fact->>'job_id', fact->>'archetype', fact->>'keyword', fact->>'category', ordinal
    ), contributions AS (
        SELECT archetype, COALESCE(provider, 'unknown') AS provider, keyword, category,
               -count(*)::integer AS delta
        FROM public.job_keyword_insights
        WHERE job_id = ANY (p_job_ids) AND archetype = p_archetype
        GROUP BY archetype, COALESCE(provider, 'unknown'), keyword, category
        UNION ALL
        SELECT archetype, COALESCE(provider, 'unknown'), keyword, category,
               count(*)::integer AS delta
        FROM new_facts
        GROUP BY archetype, COALESCE(provider, 'unknown'), keyword, category
    )
    SELECT COALESCE(
        jsonb_agg(
            jsonb_build_object(
                'archetype', archetype,
                'provider', provider,
                'keyword', keyword,
                'category', category,
                'delta', delta
            ) ORDER BY archetype, provider, keyword, category
        ),
        '[]'::jsonb
    )
    INTO v_deltas
    FROM (
        SELECT archetype, provider, keyword, category, sum(delta)::integer AS delta
        FROM contributions
        GROUP BY archetype, provider, keyword, category
        HAVING sum(delta) <> 0
    ) AS net;

    DELETE FROM public.job_keyword_insights
    WHERE job_id = ANY (p_job_ids) AND archetype = p_archetype;

    INSERT INTO public.job_keyword_insights (
        job_id, archetype, keyword, category, provider
    )
    SELECT job_id, archetype, keyword, category, provider
    FROM (
        SELECT DISTINCT ON (fact->>'job_id', fact->>'archetype', fact->>'keyword', fact->>'category')
            fact->>'job_id' AS job_id,
            fact->>'archetype' AS archetype,
            fact->>'keyword' AS keyword,
            fact->>'category' AS category,
            NULLIF(fact->>'provider', '') AS provider
        FROM jsonb_array_elements(p_facts) WITH ORDINALITY AS e(fact, ordinal)
        ORDER BY fact->>'job_id', fact->>'archetype', fact->>'keyword', fact->>'category', ordinal
    ) AS new_facts;
    GET DIAGNOSTICS v_fact_count = ROW_COUNT;

    FOR v_delta IN SELECT value FROM jsonb_array_elements(v_deltas)
    LOOP
        IF (v_delta->>'delta')::integer > 0 THEN
            INSERT INTO public.keyword_insights (
                archetype, provider, keyword, category, count
            ) VALUES (
                v_delta->>'archetype', v_delta->>'provider',
                v_delta->>'keyword', v_delta->>'category',
                (v_delta->>'delta')::integer
            )
            ON CONFLICT (archetype, provider, keyword, category) DO UPDATE
            SET count = public.keyword_insights.count + EXCLUDED.count;
        ELSE
            UPDATE public.keyword_insights
            SET count = GREATEST(0, count + (v_delta->>'delta')::integer)
            WHERE archetype = v_delta->>'archetype'
              AND provider = v_delta->>'provider'
              AND keyword = v_delta->>'keyword'
              AND category = v_delta->>'category';

            DELETE FROM public.keyword_insights
            WHERE archetype = v_delta->>'archetype'
              AND provider = v_delta->>'provider'
              AND keyword = v_delta->>'keyword'
              AND category = v_delta->>'category'
              AND count = 0;
        END IF;
    END LOOP;

    RETURN v_fact_count;
END;
$$;

ALTER FUNCTION public.replace_job_keyword_facts_and_refresh_aggregates(text[], text, jsonb) OWNER TO postgres;
REVOKE ALL ON FUNCTION public.replace_job_keyword_facts_and_refresh_aggregates(text[], text, jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.replace_job_keyword_facts_and_refresh_aggregates(text[], text, jsonb) TO service_role;

COMMIT;
