BEGIN;

ALTER TABLE public.jobs
    ADD COLUMN IF NOT EXISTS posting_wave_count integer NOT NULL DEFAULT 1;

COMMENT ON COLUMN public.jobs.seen_count IS
    'Distinct source listing-ID count retained for compatibility.';
COMMENT ON COLUMN public.jobs.posting_wave_count IS
    'Maximum chronological posting-wave count for one normalized location in listing_instances.';
COMMENT ON COLUMN public.jobs.repost_count IS
    'Confirmed later posting waves: greatest(posting_wave_count - 1, 0). Simultaneous and location variants do not increment it.';
COMMENT ON COLUMN public.jobs.listing_instances IS
    'Per-source-listing history including location, posting-wave identity, and variant classification.';

CREATE OR REPLACE FUNCTION public.calculate_listing_posting_waves(instances jsonb)
RETURNS TABLE(listing_instances jsonb, posting_wave_count integer, repost_count integer)
LANGUAGE sql
IMMUTABLE
SET search_path = public
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

REVOKE ALL ON FUNCTION public.calculate_listing_posting_waves(jsonb) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.calculate_listing_posting_waves(jsonb) TO service_role;

COMMIT;
