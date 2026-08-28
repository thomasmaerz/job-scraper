BEGIN;

SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '5min';
SET LOCAL idle_in_transaction_session_timeout = '6min';
SELECT pg_catalog.set_config('search_path', '', true);

CREATE TABLE IF NOT EXISTS public.freehire_publication_state (
    id integer PRIMARY KEY CHECK (id = 1),
    generation bigint NOT NULL DEFAULT 0 CHECK (generation >= 0),
    published_at timestamptz,
    source_scrape_watermark timestamptz,
    row_count bigint NOT NULL DEFAULT 0 CHECK (row_count >= 0),
    schema_version text NOT NULL DEFAULT 'freehire-publication-v1'
);

CREATE TABLE IF NOT EXISTS public.freehire_publication_generations (
    generation bigint PRIMARY KEY CHECK (generation > 0),
    published_at timestamptz NOT NULL,
    source_scrape_watermark timestamptz NOT NULL UNIQUE,
    row_count bigint NOT NULL CHECK (row_count > 0),
    schema_version text NOT NULL
);

CREATE TABLE IF NOT EXISTS public.freehire_publication_snapshots (
    generation bigint NOT NULL REFERENCES public.freehire_publication_generations(generation) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
    canonical_job_id text NOT NULL,
    import_hash text NOT NULL,
    payload jsonb NOT NULL,
    PRIMARY KEY (generation, canonical_job_id),
    CHECK (jsonb_typeof(payload) = 'object')
);

COMMENT ON TABLE public.freehire_publication_state IS
    'Singleton pointer to the latest complete immutable Freehire publication generation.';
COMMENT ON TABLE public.freehire_publication_generations IS
    'Completion records for exactly the latest three retained Freehire publication generations.';
COMMENT ON TABLE public.freehire_publication_snapshots IS
    'Append-only immutable payload rows for retained complete Freehire publication generations; generation pruning cascades only after a completion record exists.';

INSERT INTO public.freehire_publication_state (id)
VALUES (1)
ON CONFLICT (id) DO NOTHING;

-- Upgrade a deployment of the earlier draft contract by retaining its latest
-- generation only when its state and copied row count prove it complete.
INSERT INTO public.freehire_publication_generations (
    generation, published_at, source_scrape_watermark, row_count, schema_version
)
SELECT state.generation, state.published_at, state.source_scrape_watermark,
       state.row_count, state.schema_version
FROM public.freehire_publication_state AS state
WHERE state.id = 1
  AND state.generation > 0
  AND state.published_at IS NOT NULL
  AND state.source_scrape_watermark IS NOT NULL
  AND state.row_count > 0
  AND state.schema_version = 'freehire-publication-v1'
  AND state.row_count = (
      SELECT pg_catalog.count(*)
      FROM public.freehire_publication_snapshots AS snapshot
      WHERE snapshot.generation = state.generation
  )
ON CONFLICT (generation) DO NOTHING;

DELETE FROM public.freehire_publication_snapshots AS snapshot
WHERE NOT EXISTS (
    SELECT 1
    FROM public.freehire_publication_generations AS retained
    WHERE retained.generation = snapshot.generation
);

DO $migration$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint
        WHERE conrelid = 'public.freehire_publication_snapshots'::pg_catalog.regclass
          AND conname = 'freehire_publication_snapshots_generation_fkey'
    ) THEN
        ALTER TABLE public.freehire_publication_snapshots
            ADD CONSTRAINT freehire_publication_snapshots_generation_fkey
            FOREIGN KEY (generation)
            REFERENCES public.freehire_publication_generations(generation)
            ON DELETE CASCADE
            DEFERRABLE INITIALLY DEFERRED;
    END IF;
END;
$migration$;

DELETE FROM public.freehire_publication_generations AS expired
WHERE expired.generation NOT IN (
    SELECT retained.generation
    FROM public.freehire_publication_generations AS retained
    ORDER BY retained.generation DESC
    LIMIT 3
);

CREATE OR REPLACE FUNCTION public.finalize_freehire_publication(
    p_source_scrape_watermark timestamptz
)
RETURNS TABLE (
    generation bigint,
    published_at timestamptz,
    source_scrape_watermark timestamptz,
    row_count bigint,
    schema_version text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
DECLARE
    current_state public.freehire_publication_state%ROWTYPE;
    completed_generation public.freehire_publication_generations%ROWTYPE;
    authoritative_watermark timestamptz;
    next_generation bigint;
    source_count bigint;
    copied_count bigint;
    snapshot_count bigint;
BEGIN
    PERFORM pg_catalog.set_config('lock_timeout', '10s', true);
    PERFORM pg_catalog.set_config('statement_timeout', '5min', true);
    PERFORM pg_catalog.set_config('idle_in_transaction_session_timeout', '6min', true);

    IF p_source_scrape_watermark IS NULL THEN
        RAISE EXCEPTION 'p_source_scrape_watermark must not be null' USING ERRCODE = '22004';
    END IF;

    -- Serialize publishers first, then freeze jobs against INSERT/UPDATE/DELETE.
    -- Under READ COMMITTED this explicit SHARE lock makes all source validation
    -- and the INSERT ... SELECT observe one unchanged freehire_jobs relation
    -- through commit, while ordinary readers remain unblocked.
    SELECT *
    INTO STRICT current_state
    FROM public.freehire_publication_state AS state
    WHERE state.id = 1
    FOR UPDATE;

    LOCK TABLE public.jobs IN SHARE MODE;

    SELECT scrape_state.last_successful_scrape_at
    INTO authoritative_watermark
    FROM public.scrape_run_state AS scrape_state
    WHERE scrape_state.id = 1
    FOR SHARE;

    IF authoritative_watermark IS NULL THEN
        RAISE EXCEPTION 'authoritative scrape watermark is absent' USING ERRCODE = '55000';
    END IF;

    IF current_state.schema_version <> 'freehire-publication-v1' THEN
        RAISE EXCEPTION 'stored publication schema version is invalid (%)',
            current_state.schema_version USING ERRCODE = '55000';
    END IF;

    IF current_state.generation = 0 THEN
        IF current_state.published_at IS NOT NULL
           OR current_state.source_scrape_watermark IS NOT NULL
           OR current_state.row_count <> 0 THEN
            RAISE EXCEPTION 'initial publication state is inconsistent' USING ERRCODE = '55000';
        END IF;
    ELSIF current_state.published_at IS NULL
          OR current_state.source_scrape_watermark IS NULL
          OR current_state.row_count <= 0 THEN
        RAISE EXCEPTION 'stored publication state is incomplete' USING ERRCODE = '55000';
    END IF;

    IF current_state.source_scrape_watermark IS NOT NULL
       AND p_source_scrape_watermark < current_state.source_scrape_watermark THEN
        RAISE EXCEPTION 'source scrape watermark cannot move backwards (% < %)',
            p_source_scrape_watermark, current_state.source_scrape_watermark
            USING ERRCODE = '22000';
    END IF;

    IF p_source_scrape_watermark > authoritative_watermark THEN
        RAISE EXCEPTION 'source scrape watermark is in the future (% > %)',
            p_source_scrape_watermark, authoritative_watermark
            USING ERRCODE = '22000';
    ELSIF p_source_scrape_watermark < authoritative_watermark THEN
        RAISE EXCEPTION 'source scrape watermark is stale (% < %)',
            p_source_scrape_watermark, authoritative_watermark
            USING ERRCODE = '22000';
    END IF;

    SELECT pg_catalog.count(*)
    INTO source_count
    FROM public.freehire_jobs;

    IF source_count = 0 THEN
        RAISE EXCEPTION 'refusing to publish an empty generation' USING ERRCODE = '22000';
    END IF;

    IF current_state.source_scrape_watermark = p_source_scrape_watermark THEN
        SELECT *
        INTO completed_generation
        FROM public.freehire_publication_generations AS retained
        WHERE retained.generation = current_state.generation
        FOR SHARE;

        IF NOT FOUND THEN
            RAISE EXCEPTION 'stored publication generation % is unavailable',
                current_state.generation USING ERRCODE = '55000';
        END IF;

        SELECT pg_catalog.count(*)
        INTO snapshot_count
        FROM public.freehire_publication_snapshots AS snapshot
        WHERE snapshot.generation = current_state.generation;

        IF completed_generation.published_at <> current_state.published_at
           OR completed_generation.source_scrape_watermark <> current_state.source_scrape_watermark
           OR completed_generation.row_count <> current_state.row_count
           OR completed_generation.schema_version <> current_state.schema_version
           OR completed_generation.schema_version <> 'freehire-publication-v1'
           OR snapshot_count <> current_state.row_count
           OR source_count <> current_state.row_count THEN
            RAISE EXCEPTION 'same-watermark publication generation % failed integrity validation',
                current_state.generation USING ERRCODE = '55000';
        END IF;

        RETURN QUERY SELECT
            current_state.generation,
            current_state.published_at,
            current_state.source_scrape_watermark,
            current_state.row_count,
            current_state.schema_version;
        RETURN;
    END IF;

    next_generation := current_state.generation + 1;

    INSERT INTO public.freehire_publication_snapshots (
        generation,
        canonical_job_id,
        import_hash,
        payload
    )
    SELECT
        next_generation,
        source.job_id,
        source.freehire_compat_import_hash,
        pg_catalog.to_jsonb(source)
    FROM public.freehire_jobs AS source
    ORDER BY source.job_id;

    GET DIAGNOSTICS copied_count = ROW_COUNT;

    IF copied_count = 0 OR copied_count <> source_count THEN
        RAISE EXCEPTION 'publication copy count is invalid (copied %, validated %)',
            copied_count, source_count USING ERRCODE = '55000';
    END IF;

    INSERT INTO public.freehire_publication_generations (
        generation, published_at, source_scrape_watermark, row_count, schema_version
    )
    VALUES (
        next_generation, pg_catalog.clock_timestamp(), p_source_scrape_watermark,
        copied_count, 'freehire-publication-v1'
    )
    RETURNING * INTO completed_generation;

    UPDATE public.freehire_publication_state AS state
    SET generation = completed_generation.generation,
        published_at = completed_generation.published_at,
        source_scrape_watermark = completed_generation.source_scrape_watermark,
        row_count = completed_generation.row_count,
        schema_version = completed_generation.schema_version
    WHERE state.id = 1;

    -- A generation is complete only after its completion row is inserted.
    -- Deleting older completion rows cascades their immutable snapshot rows.
    DELETE FROM public.freehire_publication_generations AS expired
    WHERE expired.generation NOT IN (
        SELECT retained.generation
        FROM public.freehire_publication_generations AS retained
        ORDER BY retained.generation DESC
        LIMIT 3
    );

    RETURN QUERY SELECT
        completed_generation.generation,
        completed_generation.published_at,
        completed_generation.source_scrape_watermark,
        completed_generation.row_count,
        completed_generation.schema_version;
END;
$function$;

CREATE OR REPLACE FUNCTION public.get_freehire_publication_state()
RETURNS TABLE (
    generation bigint,
    published_at timestamptz,
    source_scrape_watermark timestamptz,
    row_count bigint,
    schema_version text
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
DECLARE
    current_state public.freehire_publication_state%ROWTYPE;
    completed_generation public.freehire_publication_generations%ROWTYPE;
    snapshot_count bigint;
BEGIN
    SELECT *
    INTO STRICT current_state
    FROM public.freehire_publication_state AS state
    WHERE state.id = 1;

    IF current_state.generation = 0 THEN
        IF current_state.published_at IS NOT NULL
           OR current_state.source_scrape_watermark IS NOT NULL
           OR current_state.row_count <> 0
           OR current_state.schema_version <> 'freehire-publication-v1' THEN
            RAISE EXCEPTION 'initial publication state is inconsistent' USING ERRCODE = '55000';
        END IF;
    ELSE
        SELECT *
        INTO completed_generation
        FROM public.freehire_publication_generations AS retained
        WHERE retained.generation = current_state.generation
        FOR SHARE;

        IF NOT FOUND THEN
            RAISE EXCEPTION 'current publication generation % is unavailable',
                current_state.generation USING ERRCODE = '55000';
        END IF;

        SELECT pg_catalog.count(*)
        INTO snapshot_count
        FROM public.freehire_publication_snapshots AS snapshot
        WHERE snapshot.generation = current_state.generation;

        IF completed_generation.published_at <> current_state.published_at
           OR completed_generation.source_scrape_watermark <> current_state.source_scrape_watermark
           OR completed_generation.row_count <> current_state.row_count
           OR completed_generation.schema_version <> current_state.schema_version
           OR completed_generation.schema_version <> 'freehire-publication-v1'
           OR completed_generation.row_count <= 0
           OR snapshot_count <> completed_generation.row_count THEN
            RAISE EXCEPTION 'current publication generation % failed integrity validation',
                current_state.generation USING ERRCODE = '55000';
        END IF;
    END IF;

    RETURN QUERY SELECT
        current_state.generation,
        current_state.published_at,
        current_state.source_scrape_watermark,
        current_state.row_count,
        current_state.schema_version;
END;
$function$;

CREATE OR REPLACE FUNCTION public.get_freehire_publication_page(
    p_generation bigint,
    p_after_canonical_job_id text DEFAULT NULL,
    p_page_size integer DEFAULT 1000
)
RETURNS TABLE (
    canonical_job_id text,
    import_hash text,
    payload jsonb
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $function$
DECLARE
    completed_generation public.freehire_publication_generations%ROWTYPE;
    snapshot_count bigint;
BEGIN
    IF p_generation IS NULL OR p_generation <= 0 THEN
        RAISE EXCEPTION 'p_generation must be positive' USING ERRCODE = '22023';
    END IF;
    IF p_page_size IS NULL OR p_page_size <= 0 THEN
        RAISE EXCEPTION 'p_page_size must be positive; values over 1000 are capped' USING ERRCODE = '22023';
    END IF;

    -- The row lock keeps this retained generation available for the complete
    -- page RPC transaction while finalization attempts bounded pruning.
    SELECT *
    INTO completed_generation
    FROM public.freehire_publication_generations AS retained
    WHERE retained.generation = p_generation
    FOR SHARE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'publication generation % is unavailable; restart from current state',
            p_generation USING ERRCODE = '22023';
    END IF;

    SELECT pg_catalog.count(*)
    INTO snapshot_count
    FROM public.freehire_publication_snapshots AS snapshot
    WHERE snapshot.generation = p_generation;

    IF completed_generation.schema_version <> 'freehire-publication-v1'
       OR completed_generation.row_count <= 0
       OR snapshot_count <> completed_generation.row_count THEN
        RAISE EXCEPTION 'publication generation % is incomplete',
            p_generation USING ERRCODE = '55000';
    END IF;

    RETURN QUERY
    SELECT snapshot.canonical_job_id, snapshot.import_hash, snapshot.payload
    FROM public.freehire_publication_snapshots AS snapshot
    WHERE snapshot.generation = p_generation
      AND (
          p_after_canonical_job_id IS NULL
          OR snapshot.canonical_job_id > p_after_canonical_job_id
      )
    ORDER BY snapshot.canonical_job_id
    LIMIT LEAST(p_page_size, 1000);
END;
$function$;

DO $role$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'freehire_publication_reader'
    ) THEN
        CREATE ROLE freehire_publication_reader
            NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
    END IF;
END;
$role$;
ALTER TABLE public.freehire_publication_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.freehire_publication_generations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.freehire_publication_snapshots ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_publication_state_access ON public.freehire_publication_state;
DROP POLICY IF EXISTS service_role_publication_generation_access ON public.freehire_publication_generations;
DROP POLICY IF EXISTS service_role_publication_snapshot_access ON public.freehire_publication_snapshots;

REVOKE ALL ON TABLE public.freehire_publication_state FROM PUBLIC, anon, authenticated, service_role, freehire_publication_reader;
REVOKE ALL ON TABLE public.freehire_publication_generations FROM PUBLIC, anon, authenticated, service_role, freehire_publication_reader;
REVOKE ALL ON TABLE public.freehire_publication_snapshots FROM PUBLIC, anon, authenticated, service_role, freehire_publication_reader;
REVOKE ALL ON SCHEMA public FROM freehire_publication_reader;
GRANT USAGE ON SCHEMA public TO freehire_publication_reader;

REVOKE ALL ON FUNCTION public.finalize_freehire_publication(timestamptz) FROM PUBLIC, anon, authenticated, service_role, freehire_publication_reader;
REVOKE ALL ON FUNCTION public.get_freehire_publication_state() FROM PUBLIC, anon, authenticated, service_role, freehire_publication_reader;
REVOKE ALL ON FUNCTION public.get_freehire_publication_page(bigint, text, integer) FROM PUBLIC, anon, authenticated, service_role, freehire_publication_reader;

GRANT EXECUTE ON FUNCTION public.finalize_freehire_publication(timestamptz) TO service_role;
GRANT EXECUTE ON FUNCTION public.get_freehire_publication_state() TO freehire_publication_reader;
GRANT EXECUTE ON FUNCTION public.get_freehire_publication_page(bigint, text, integer) TO freehire_publication_reader;
REVOKE EXECUTE ON FUNCTION public.finalize_freehire_publication(timestamptz) FROM freehire_publication_reader;

COMMIT;
