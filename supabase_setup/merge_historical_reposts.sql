BEGIN;

SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '15min';

ALTER TABLE public.jobs
    ADD COLUMN IF NOT EXISTS last_seen_at timestamptz,
    ADD COLUMN IF NOT EXISTS listing_instances jsonb NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS seen_count integer NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS posting_wave_count integer NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS repost_count integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS latest_job_id text,
    ADD COLUMN IF NOT EXISTS last_seen_posted_at timestamptz,
    ADD COLUMN IF NOT EXISTS description_fingerprint text,
    ADD COLUMN IF NOT EXISTS resume_score_stage text NOT NULL DEFAULT 'initial',
    ADD COLUMN IF NOT EXISTS search_query text,
    ADD COLUMN IF NOT EXISTS archetype text,
    ADD COLUMN IF NOT EXISTS filter_profile text,
    ADD COLUMN IF NOT EXISTS location_province_code text,
    ADD COLUMN IF NOT EXISTS location_scope text,
    ADD COLUMN IF NOT EXISTS location_metro text;

CREATE OR REPLACE FUNCTION public.calculate_listing_posting_waves(instances jsonb)
RETURNS TABLE(listing_instances jsonb, posting_wave_count integer, repost_count integer)
LANGUAGE sql
IMMUTABLE
SET search_path = pg_catalog, public
AS $$
WITH RECURSIVE source AS (
    SELECT
        ordinality::integer AS node_id,
        instance,
        NULLIF(
            btrim(regexp_replace(
                regexp_replace(
                    regexp_replace(lower(COALESCE(instance->>'location', '')), '[-/]', ' ', 'g'),
                    '[[:punct:]]', '', 'g'
                ),
                '\s+', ' ', 'g'
            )),
            ''
        ) AS normalized_location,
        CASE
            WHEN COALESCE(instance->>'posted_at', '') ~ '^\d{4}-\d{2}-\d{2}'
            THEN substring(instance->>'posted_at' FROM 1 FOR 10)
        END AS posted_date,
        NULLIF(instance->>'scrape_run_id', '') AS scrape_run_id,
        CASE
            WHEN COALESCE(instance->>'scraped_at', '') ~ '^\d{4}-\d{2}-\d{2}'
            THEN substring(instance->>'scraped_at' FROM 1 FOR 10)
        END AS scraped_date
    FROM jsonb_array_elements(COALESCE(instances, '[]'::jsonb)) WITH ORDINALITY values(instance, ordinality)
), edges AS (
    SELECT left_source.node_id AS left_id, right_source.node_id AS right_id
    FROM source left_source
    JOIN source right_source
      ON left_source.node_id <> right_source.node_id
     AND left_source.normalized_location IS NOT DISTINCT FROM right_source.normalized_location
     AND (
        left_source.normalized_location IS NULL
        OR (left_source.posted_date IS NOT NULL AND left_source.posted_date = right_source.posted_date)
        OR (left_source.scrape_run_id IS NOT NULL AND left_source.scrape_run_id = right_source.scrape_run_id)
        OR (
            left_source.posted_date IS NULL
            AND right_source.posted_date IS NULL
            AND left_source.scrape_run_id IS NULL
            AND right_source.scrape_run_id IS NULL
            AND (
                (left_source.scraped_date IS NOT NULL AND left_source.scraped_date = right_source.scraped_date)
                OR (left_source.scraped_date IS NULL AND right_source.scraped_date IS NULL)
            )
        )
     )
), reach(node_id, root_id) AS (
    SELECT node_id, node_id FROM source
    UNION
    SELECT edges.right_id, reach.root_id
    FROM reach
    JOIN edges ON edges.left_id = reach.node_id
), components AS (
    SELECT node_id, min(root_id) AS component_id
    FROM reach
    GROUP BY node_id
), component_values AS (
    SELECT
        components.component_id,
        min(CASE
            WHEN source.posted_date IS NOT NULL THEN source.posted_date
            WHEN source.instance->>'scraped_at' IS NOT NULL THEN source.instance->>'scraped_at'
            WHEN source.scrape_run_id IS NOT NULL THEN source.scrape_run_id
            ELSE '9999-12-31'
        END) AS component_sort,
        min(CASE
            WHEN source.normalized_location IS NULL THEN 'unknown_location'
            WHEN source.posted_date IS NOT NULL THEN 'posted:' || source.posted_date
            WHEN source.scrape_run_id IS NOT NULL THEN 'scrape_run:' || source.scrape_run_id
            WHEN source.scraped_date IS NOT NULL THEN 'scrape_date:' || source.scraped_date
            ELSE 'unknown'
        END) AS component_key
    FROM components
    JOIN source USING (node_id)
    GROUP BY components.component_id
), location_values AS (
    SELECT normalized_location, min(node_id) AS location_sort
    FROM source
    GROUP BY normalized_location
), ranked AS (
    SELECT
        source.*,
        components.component_id,
        dense_rank() OVER (
            PARTITION BY source.normalized_location
            ORDER BY component_values.component_sort, components.component_id
        )::integer AS wave_index,
        row_number() OVER (PARTITION BY components.component_id ORDER BY source.node_id)::integer AS member_index,
        dense_rank() OVER (
            ORDER BY location_values.location_sort
        )::integer AS location_index
    FROM source
    JOIN components USING (node_id)
    JOIN component_values USING (component_id)
    JOIN location_values ON location_values.normalized_location IS NOT DISTINCT FROM source.normalized_location
), annotated AS (
    SELECT
        ranked.node_id,
        jsonb_set(
            jsonb_set(
                jsonb_set(
                    CASE
                        WHEN ranked.normalized_location IS NULL THEN ranked.instance - 'normalized_location'
                        ELSE jsonb_set(
                            ranked.instance,
                            '{normalized_location}',
                            to_jsonb(ranked.normalized_location),
                            true
                        )
                    END,
                    '{posting_wave_key}',
                    to_jsonb(COALESCE(ranked.normalized_location, '') || '|' || component_values.component_key),
                    true
                ),
                '{posting_wave_index}',
                to_jsonb(ranked.wave_index),
                true
            ),
            '{variant_type}',
            to_jsonb(CASE
                WHEN ranked.member_index > 1 THEN 'simultaneous_variant'
                WHEN ranked.wave_index > 1 THEN 'repost'
                WHEN ranked.location_index > 1 THEN 'location_variant'
                ELSE 'original'
            END),
            true
        ) AS instance,
        ranked.wave_index
    FROM ranked
    JOIN component_values USING (component_id)
), result AS (
    SELECT
        COALESCE(jsonb_agg(instance ORDER BY node_id), '[]'::jsonb) AS listing_instances,
        COALESCE(max(wave_index), 0)::integer AS posting_wave_count
    FROM annotated
)
SELECT
    result.listing_instances,
    result.posting_wave_count,
    GREATEST(result.posting_wave_count - 1, 0)::integer AS repost_count
FROM result;
$$;

REVOKE ALL ON FUNCTION public.calculate_listing_posting_waves(jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.calculate_listing_posting_waves(jsonb) TO service_role;

CREATE TABLE IF NOT EXISTS public.job_listing_archive (
    provider text NOT NULL,
    source_job_id text NOT NULL,
    canonical_job_id text NOT NULL REFERENCES public.jobs(job_id) ON DELETE CASCADE,
    observed_at timestamptz,
    source_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (provider, source_job_id)
);

CREATE TABLE IF NOT EXISTS public.job_resume_links (
    canonical_job_id text NOT NULL REFERENCES public.jobs(job_id) ON DELETE CASCADE,
    customized_resume_id uuid NOT NULL,
    source_job_id text,
    PRIMARY KEY (canonical_job_id, customized_resume_id)
);

ALTER TABLE public.job_resume_links
    DROP CONSTRAINT IF EXISTS job_resume_links_customized_resume_id_fkey;

CREATE TABLE IF NOT EXISTS public.job_repost_merge_plan (
    source_job_id text PRIMARY KEY,
    survivor_job_id text NOT NULL,
    match_method text NOT NULL,
    match_similarity numeric,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (source_job_id <> survivor_job_id)
);

ALTER TABLE public.job_listing_archive ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.job_resume_links ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.job_repost_merge_plan ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION public.replace_historical_repost_plan(p_plan jsonb)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    plan_count integer;
BEGIN
    -- Staging and merging must observe one complete plan at a time.
    PERFORM pg_advisory_xact_lock(
        hashtextextended('historical-repost-plan-global', 0)
    );

    IF p_plan IS NULL OR jsonb_typeof(p_plan) <> 'array' THEN
        RAISE EXCEPTION 'Historical repost merge plan must be a JSON array';
    END IF;

    IF jsonb_array_length(p_plan) > 50000 THEN
        RAISE EXCEPTION 'Historical repost merge plan exceeds 50000 rows';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM jsonb_array_elements(p_plan) AS plan(item)
        WHERE jsonb_typeof(item) <> 'object'
           OR jsonb_typeof(item->'source_job_id') IS DISTINCT FROM 'string'
           OR btrim(item->>'source_job_id') = ''
           OR jsonb_typeof(item->'survivor_job_id') IS DISTINCT FROM 'string'
           OR btrim(item->>'survivor_job_id') = ''
           OR jsonb_typeof(item->'match_method') IS DISTINCT FROM 'string'
           OR btrim(item->>'match_method') = ''
           OR (
               item ? 'match_similarity'
               AND jsonb_typeof(item->'match_similarity') NOT IN ('number', 'null')
           )
           OR item->>'source_job_id' = item->>'survivor_job_id'
    ) THEN
        RAISE EXCEPTION 'Historical repost merge plan contains an invalid row';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM jsonb_array_elements(p_plan) AS plan(item)
        GROUP BY item->>'source_job_id'
        HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION 'Historical repost merge plan contains duplicate source jobs';
    END IF;

    DELETE FROM public.job_repost_merge_plan;

    INSERT INTO public.job_repost_merge_plan (
        source_job_id,
        survivor_job_id,
        match_method,
        match_similarity
    )
    SELECT
        item->>'source_job_id',
        item->>'survivor_job_id',
        item->>'match_method',
        (item->>'match_similarity')::numeric
    FROM jsonb_array_elements(p_plan) AS plan(item);

    GET DIAGNOSTICS plan_count = ROW_COUNT;
    RETURN plan_count;
END;
$$;

ALTER FUNCTION public.replace_historical_repost_plan(jsonb) OWNER TO postgres;
REVOKE ALL ON FUNCTION public.replace_historical_repost_plan(jsonb) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.replace_historical_repost_plan(jsonb) TO service_role;
REVOKE ALL ON TABLE public.job_repost_merge_plan FROM PUBLIC, anon, authenticated, service_role;

CREATE OR REPLACE FUNCTION public.merge_historical_repost_plan()
RETURNS TABLE(merged_groups integer, deleted_rows integer)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    survivor text;
    group_count integer := 0;
    deleted_count integer := 0;
    affected integer;
BEGIN
    -- Serialize execution with atomic plan replacement.
    PERFORM pg_advisory_xact_lock(
        hashtextextended('historical-repost-plan-global', 0)
    );

    -- Keyword facts are moved below; serialize that mutation with delta/rebuild RPCs.
    PERFORM pg_advisory_xact_lock(
        hashtextextended('keyword-insights-aggregate-global', 0)
    );

    -- Lock every source and survivor in one deterministic order before reading
    -- or deleting any merge inputs. Concurrent scraper updates then wait for
    -- this transaction instead of updating a row that is about to be removed.
    PERFORM jobs.job_id
    FROM public.jobs AS jobs
    WHERE jobs.job_id IN (
        SELECT source_job_id
        FROM public.job_repost_merge_plan
        UNION
        SELECT survivor_job_id
        FROM public.job_repost_merge_plan
    )
    ORDER BY jobs.job_id
    FOR UPDATE OF jobs;

    IF EXISTS (
        SELECT 1
        FROM public.job_repost_merge_plan p
        LEFT JOIN public.jobs source ON source.job_id = p.source_job_id
        LEFT JOIN public.jobs target ON target.job_id = p.survivor_job_id
        WHERE source.job_id IS NULL OR target.job_id IS NULL
    ) THEN
        RAISE EXCEPTION 'Merge plan contains missing source or survivor jobs';
    END IF;

    IF EXISTS (
        SELECT 1 FROM public.job_repost_merge_plan p
        JOIN public.job_repost_merge_plan nested ON nested.source_job_id = p.survivor_job_id
    ) THEN
        RAISE EXCEPTION 'Merge plan contains survivor chains';
    END IF;

    FOR survivor IN
        SELECT DISTINCT survivor_job_id FROM public.job_repost_merge_plan ORDER BY survivor_job_id
    LOOP
        IF EXISTS (
            SELECT 1
            FROM public.jobs source
            JOIN public.jobs target ON target.job_id = survivor
            WHERE source.job_id IN (
                SELECT source_job_id FROM public.job_repost_merge_plan WHERE survivor_job_id = survivor
            )
              AND source.archetype IS DISTINCT FROM target.archetype
        ) THEN
            RAISE EXCEPTION 'Merge group % contains conflicting archetypes', survivor;
        END IF;

        IF EXISTS (
            SELECT 1
            FROM public.jobs source
            JOIN public.jobs target ON target.job_id = survivor
            JOIN public.job_repost_merge_plan plan
              ON plan.source_job_id = source.job_id
             AND plan.survivor_job_id = survivor
            WHERE plan.match_method = 'body_hash_fuzzy_title'
              AND (
                  source.provider IS DISTINCT FROM target.provider
                  OR source.company IS NULL
                  OR target.company IS NULL
                  OR source.description_fingerprint IS NULL
                  OR source.description_fingerprint IS DISTINCT FROM target.description_fingerprint
              )
        ) THEN
            RAISE EXCEPTION 'Merge group % failed exact description identity validation', survivor;
        END IF;

        INSERT INTO public.job_listing_archive (
            provider, source_job_id, canonical_job_id, observed_at, source_snapshot
        )
        SELECT
            j.provider,
            j.job_id,
            survivor,
            COALESCE(j.last_seen_at, j.scraped_at),
            to_jsonb(j)
        FROM public.jobs j
        WHERE j.job_id = survivor
           OR j.job_id IN (
               SELECT source_job_id FROM public.job_repost_merge_plan WHERE survivor_job_id = survivor
           )
        ON CONFLICT (provider, source_job_id) DO UPDATE SET
            canonical_job_id = EXCLUDED.canonical_job_id,
            observed_at = EXCLUDED.observed_at,
            source_snapshot = EXCLUDED.source_snapshot;

        INSERT INTO public.job_listing_archive (
            provider, source_job_id, canonical_job_id, observed_at, source_snapshot
        )
        SELECT DISTINCT ON (j.provider, instance->>'job_id')
            j.provider,
            instance->>'job_id',
            survivor,
            COALESCE((instance->>'scraped_at')::timestamptz, j.last_seen_at, j.scraped_at),
            instance
        FROM public.jobs j
        CROSS JOIN LATERAL jsonb_array_elements(COALESCE(j.listing_instances, '[]'::jsonb)) instance
        WHERE (j.job_id = survivor OR j.job_id IN (
            SELECT source_job_id FROM public.job_repost_merge_plan WHERE survivor_job_id = survivor
        ))
          AND instance->>'job_id' IS NOT NULL
        ORDER BY j.provider, instance->>'job_id', COALESCE((instance->>'scraped_at')::timestamptz, j.last_seen_at, j.scraped_at) DESC
        ON CONFLICT (provider, source_job_id) DO UPDATE SET
            canonical_job_id = EXCLUDED.canonical_job_id,
            observed_at = GREATEST(public.job_listing_archive.observed_at, EXCLUDED.observed_at),
            source_snapshot = public.job_listing_archive.source_snapshot || EXCLUDED.source_snapshot;

        INSERT INTO public.job_resume_links (canonical_job_id, customized_resume_id, source_job_id)
        SELECT survivor, j.customized_resume_id, j.job_id
        FROM public.jobs j
        WHERE j.customized_resume_id IS NOT NULL
          AND (j.job_id = survivor OR j.job_id IN (
              SELECT source_job_id FROM public.job_repost_merge_plan WHERE survivor_job_id = survivor
          ))
        ON CONFLICT (canonical_job_id, customized_resume_id) DO UPDATE SET
            source_job_id = EXCLUDED.source_job_id;

        INSERT INTO public.job_keyword_insights (
            job_id, keyword, category, analyzed_at, archetype, provider
        )
        SELECT
            survivor, keyword, category, max(analyzed_at), archetype,
            (array_agg(provider ORDER BY analyzed_at DESC) FILTER (WHERE provider IS NOT NULL))[1]
        FROM public.job_keyword_insights
        WHERE job_id IN (
            SELECT source_job_id FROM public.job_repost_merge_plan WHERE survivor_job_id = survivor
        )
        GROUP BY keyword, category, archetype
        ON CONFLICT (job_id, archetype, keyword, category) DO UPDATE SET
            analyzed_at = GREATEST(public.job_keyword_insights.analyzed_at, EXCLUDED.analyzed_at),
            provider = COALESCE(EXCLUDED.provider, public.job_keyword_insights.provider);

        DELETE FROM public.job_keyword_insights
        WHERE job_id IN (
            SELECT source_job_id FROM public.job_repost_merge_plan WHERE survivor_job_id = survivor
        );

        WITH members AS (
            SELECT j.*
            FROM public.jobs j
            WHERE j.job_id = survivor OR j.job_id IN (
                SELECT source_job_id FROM public.job_repost_merge_plan WHERE survivor_job_id = survivor
            )
        ), aggregate_values AS (
            SELECT
                (array_agg(company ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE company IS NOT NULL))[1] company,
                (array_agg(job_title ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE job_title IS NOT NULL))[1] job_title,
                (array_agg(level ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE level IS NOT NULL))[1] level,
                string_agg(DISTINCT btrim(location), '; ' ORDER BY btrim(location)) FILTER (WHERE btrim(COALESCE(location, '')) <> '') location,
                (array_agg(description ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE description IS NOT NULL))[1] description,
                (array_agg(status ORDER BY CASE status WHEN 'offer' THEN 4 WHEN 'interviewing' THEN 3 WHEN 'applied' THEN 2 ELSE 1 END DESC, COALESCE(application_date, scraped_at) DESC) FILTER (WHERE status IS NOT NULL))[1] status,
                bool_or(is_active) is_active,
                min(application_date) application_date,
                max(resume_score) resume_score,
                string_agg(DISTINCT notes, E'\n\n' ORDER BY notes) FILTER (WHERE notes IS NOT NULL AND btrim(notes) <> '') notes,
                min(scraped_at) scraped_at,
                max(last_checked) last_checked,
                bool_or(is_interested) is_interested,
                (array_agg(customized_resume_id ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE customized_resume_id IS NOT NULL))[1] customized_resume_id,
                max(posted_at) posted_at,
                bool_and(is_filtered) is_filtered,
                (array_agg(filter_reason ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE filter_reason IS NOT NULL))[1] filter_reason,
                bool_or(is_entry_level_filtered) is_entry_level_filtered,
                max(insights_analyzed_at) insights_analyzed_at,
                max(insights_reanalyzed_at) insights_reanalyzed_at,
                (array_agg(search_query ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE search_query IS NOT NULL))[1] search_query,
                (array_agg(archetype ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE archetype IS NOT NULL))[1] archetype,
                (array_agg(filter_profile ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE filter_profile IS NOT NULL))[1] filter_profile,
                (array_agg(canonical_key ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE canonical_key IS NOT NULL))[1] canonical_key,
                (array_agg(resume_score_stage ORDER BY CASE resume_score_stage WHEN 'final' THEN 2 WHEN 'initial' THEN 1 ELSE 0 END DESC, COALESCE(last_seen_at, scraped_at) DESC))[1] resume_score_stage,
                min(first_seen_at) first_seen_at,
                max(last_seen_at) last_seen_at,
                max(last_seen_posted_at) last_seen_posted_at,
                (array_agg(posted_relative_text ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE posted_relative_text IS NOT NULL))[1] posted_relative_text,
                (array_agg(applicant_count ORDER BY COALESCE(detail_metadata_checked_at, last_seen_at, scraped_at) DESC) FILTER (WHERE applicant_count IS NOT NULL))[1] applicant_count,
                (array_agg(applicant_count_text ORDER BY COALESCE(detail_metadata_checked_at, last_seen_at, scraped_at) DESC) FILTER (WHERE applicant_count_text IS NOT NULL))[1] applicant_count_text,
                (array_agg(applicant_count_type ORDER BY COALESCE(detail_metadata_checked_at, last_seen_at, scraped_at) DESC) FILTER (WHERE applicant_count_type IS NOT NULL))[1] applicant_count_type,
                (array_agg(salary_text ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE salary_text IS NOT NULL))[1] salary_text,
                (array_agg(salary_min ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE salary_min IS NOT NULL))[1] salary_min,
                (array_agg(salary_max ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE salary_max IS NOT NULL))[1] salary_max,
                (array_agg(salary_currency ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE salary_currency IS NOT NULL))[1] salary_currency,
                (array_agg(recruiter_name ORDER BY COALESCE(detail_metadata_checked_at, last_seen_at, scraped_at) DESC) FILTER (WHERE recruiter_name IS NOT NULL))[1] recruiter_name,
                (array_agg(recruiter_profile_url ORDER BY COALESCE(detail_metadata_checked_at, last_seen_at, scraped_at) DESC) FILTER (WHERE recruiter_profile_url IS NOT NULL))[1] recruiter_profile_url,
                (array_agg(recruiter_identifier ORDER BY COALESCE(detail_metadata_checked_at, last_seen_at, scraped_at) DESC) FILTER (WHERE recruiter_identifier IS NOT NULL))[1] recruiter_identifier,
                max(detail_metadata_checked_at) detail_metadata_checked_at,
                (array_agg(location_province_code ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE location_province_code IS NOT NULL))[1] location_province_code,
                (array_agg(location_scope ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE location_scope IS NOT NULL))[1] location_scope,
                (array_agg(location_metro ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE location_metro IS NOT NULL))[1] location_metro
            FROM members
        ), listing_values AS (
            SELECT
                (array_agg(source_job_id ORDER BY observed_at, source_job_id))[1] original_job_id,
                (array_agg(source_job_id ORDER BY observed_at DESC, source_job_id DESC))[1] latest_job_id,
                count(*)::integer seen_count,
                jsonb_agg(
                    jsonb_strip_nulls(jsonb_build_object(
                        'job_id', source_job_id,
                        'scraped_at', COALESCE(source_snapshot->>'scraped_at', observed_at::text),
                        'last_seen_at', COALESCE(source_snapshot->>'last_seen_at', source_snapshot->>'scraped_at', observed_at::text),
                        'scrape_run_id', source_snapshot->>'scrape_run_id',
                        'location', source_snapshot->>'location',
                        'posted_at', source_snapshot->>'posted_at',
                        'posted_relative_text', source_snapshot->>'posted_relative_text',
                        'applicant_count', source_snapshot->'applicant_count',
                        'applicant_count_text', source_snapshot->>'applicant_count_text',
                        'applicant_count_type', source_snapshot->>'applicant_count_type',
                        'salary_text', source_snapshot->>'salary_text',
                        'salary_min', source_snapshot->'salary_min',
                        'salary_max', source_snapshot->'salary_max',
                        'salary_currency', source_snapshot->>'salary_currency',
                        'recruiter_name', source_snapshot->>'recruiter_name',
                        'recruiter_profile_url', source_snapshot->>'recruiter_profile_url',
                        'recruiter_identifier', source_snapshot->>'recruiter_identifier',
                        'detail_metadata_checked_at', source_snapshot->>'detail_metadata_checked_at'
                    ))
                    ORDER BY observed_at, source_job_id
                ) raw_listing_instances
            FROM public.job_listing_archive
            WHERE canonical_job_id = survivor
        ), listing_waves AS (
            SELECT waves.*
            FROM listing_values lv
            CROSS JOIN LATERAL public.calculate_listing_posting_waves(lv.raw_listing_instances) waves
        )
        UPDATE public.jobs target SET
            company = a.company,
            job_title = a.job_title,
            level = a.level,
            location = a.location,
            description = a.description,
            status = a.status,
            is_active = a.is_active,
            application_date = a.application_date,
            resume_score = a.resume_score,
            notes = a.notes,
            scraped_at = a.scraped_at,
            last_checked = a.last_checked,
            job_state = CASE WHEN a.is_active THEN 'new' ELSE target.job_state END,
            is_interested = a.is_interested,
            customized_resume_id = a.customized_resume_id,
            posted_at = a.posted_at,
            is_filtered = a.is_filtered,
            filter_reason = CASE WHEN a.is_filtered THEN a.filter_reason ELSE NULL END,
            is_entry_level_filtered = a.is_entry_level_filtered,
            insights_analyzed_at = a.insights_analyzed_at,
            insights_reanalyzed_at = a.insights_reanalyzed_at,
            search_query = a.search_query,
            archetype = a.archetype,
            filter_profile = a.filter_profile,
            canonical_key = a.canonical_key,
            resume_score_stage = a.resume_score_stage,
            original_job_id = l.original_job_id,
            latest_job_id = l.latest_job_id,
            first_seen_at = a.first_seen_at,
            last_seen_at = a.last_seen_at,
            last_seen_posted_at = a.last_seen_posted_at,
            posted_relative_text = a.posted_relative_text,
            applicant_count = a.applicant_count,
            applicant_count_text = a.applicant_count_text,
            applicant_count_type = a.applicant_count_type,
            salary_text = a.salary_text,
            salary_min = a.salary_min,
            salary_max = a.salary_max,
            salary_currency = a.salary_currency,
            recruiter_name = a.recruiter_name,
            recruiter_profile_url = a.recruiter_profile_url,
            recruiter_identifier = a.recruiter_identifier,
            seen_count = l.seen_count,
            posting_wave_count = w.posting_wave_count,
            repost_count = w.repost_count,
            listing_instances = w.listing_instances,
            detail_metadata_checked_at = a.detail_metadata_checked_at
        FROM aggregate_values a, listing_values l, listing_waves w
        WHERE target.job_id = survivor;

        DELETE FROM public.jobs
        WHERE job_id IN (
            SELECT source_job_id FROM public.job_repost_merge_plan WHERE survivor_job_id = survivor
        );
        GET DIAGNOSTICS affected = ROW_COUNT;
        deleted_count := deleted_count + affected;
        group_count := group_count + 1;
    END LOOP;

    IF EXISTS (
        SELECT 1
        FROM public.job_listing_archive a
        LEFT JOIN public.jobs j ON j.job_id = a.canonical_job_id
        WHERE j.job_id IS NULL
    ) THEN
        RAISE EXCEPTION 'Archived listing references a missing canonical job';
    END IF;

    -- Keep aggregate keyword counts transactionally consistent with the facts
    -- moved above. The transaction-wide advisory lock acquired at entry is the
    -- same lock used by all incremental keyword fact writers and rebuilds.
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

    DELETE FROM public.job_repost_merge_plan;
    RETURN QUERY SELECT group_count, deleted_count;
END;
$$;

ALTER FUNCTION public.merge_historical_repost_plan() OWNER TO postgres;
REVOKE ALL ON FUNCTION public.merge_historical_repost_plan() FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.merge_historical_repost_plan() TO service_role;

-- SECURITY DEFINER RPCs are the only supported mutation paths. ALL also
-- removes stale TRUNCATE, REFERENCES, and TRIGGER grants from deployments that
-- previously granted full table access.
REVOKE ALL PRIVILEGES ON TABLE public.keyword_insights, public.job_keyword_insights
FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT ON TABLE public.keyword_insights, public.job_keyword_insights
TO anon, authenticated, service_role;

COMMIT;
