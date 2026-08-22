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
    customized_resume_id uuid NOT NULL REFERENCES public.customized_resumes(id) ON DELETE CASCADE,
    source_job_id text,
    PRIMARY KEY (canonical_job_id, customized_resume_id)
);

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

CREATE OR REPLACE FUNCTION public.merge_historical_repost_plan()
RETURNS TABLE(merged_groups integer, deleted_rows integer)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    survivor text;
    group_count integer := 0;
    deleted_count integer := 0;
    affected integer;
BEGIN
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
                (array_agg(location ORDER BY COALESCE(last_seen_at, scraped_at) DESC) FILTER (WHERE location IS NOT NULL))[1] location,
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
                        'posted_at', source_snapshot->>'posted_at',
                        'posted_relative_text', source_snapshot->>'posted_relative_text',
                        'applicant_count', source_snapshot->'applicant_count',
                        'applicant_count_text', source_snapshot->>'applicant_count_text',
                        'applicant_count_type', source_snapshot->>'applicant_count_type',
                        'salary_text', source_snapshot->>'salary_text',
                        'recruiter_name', source_snapshot->>'recruiter_name',
                        'recruiter_profile_url', source_snapshot->>'recruiter_profile_url',
                        'recruiter_identifier', source_snapshot->>'recruiter_identifier',
                        'detail_metadata_checked_at', source_snapshot->>'detail_metadata_checked_at'
                    ))
                    ORDER BY observed_at, source_job_id
                ) listing_instances
            FROM public.job_listing_archive
            WHERE canonical_job_id = survivor
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
            repost_count = GREATEST(l.seen_count - 1, 0),
            listing_instances = l.listing_instances,
            detail_metadata_checked_at = a.detail_metadata_checked_at
        FROM aggregate_values a, listing_values l
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

    TRUNCATE public.job_repost_merge_plan;
    RETURN QUERY SELECT group_count, deleted_count;
END;
$$;

REVOKE ALL ON FUNCTION public.merge_historical_repost_plan() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.merge_historical_repost_plan() TO service_role;
