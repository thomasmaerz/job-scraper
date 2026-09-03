import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _publication_function_body(sql, name, tag):
    definitions = re.findall(
        rf"CREATE OR REPLACE FUNCTION public\.{name}\b.*?AS \${tag}\$(.*?)\${tag}\$;",
        sql,
        re.DOTALL,
    )
    body = re.sub(r"--.*", "", definitions[-1])
    return re.sub(r"\s+", "", body)


def _function_body(sql, name):
    definition = re.search(
        rf"CREATE OR REPLACE FUNCTION public\.{name}\b.*?AS \$(\w*)\$(.*?)\$\1\$;",
        sql,
        re.DOTALL,
    )
    assert definition, f"missing public.{name} definition"
    body = re.sub(r"--.*", "", definition.group(2))
    return re.sub(r"\s+", "", body)


def _jobs_columns(init_sql):
    create_table = re.search(
        r'CREATE TABLE IF NOT EXISTS "public"\."jobs" \((.*?)\n\);',
        init_sql,
        re.DOTALL,
    ).group(1)
    columns = set(re.findall(r'^\s*"([a-z_]+)"\s+', create_table, re.MULTILINE))
    columns.update(re.findall(r'ADD COLUMN IF NOT EXISTS "([a-z_]+)"', init_sql))
    return columns


def _freehire_view_source_columns(sql):
    view = re.search(
        r'CREATE OR REPLACE VIEW\s+(?:"public"|public)\.(?:"freehire_jobs"|freehire_jobs)\s+AS\s+'
        r'SELECT(.*?)FROM\s+(?:"public"|public)\.(?:"jobs"|jobs)(.*?);',
        sql,
        re.DOTALL | re.IGNORECASE,
    )
    select, filters = view.groups()
    aliases = set(re.findall(r'\bAS\s+"?([a-z_]+)"?', select, re.IGNORECASE))
    references = re.sub(r"'[^']*'", "", select + filters)
    identifiers = {
        quoted or bare
        for quoted, bare in re.findall(r'"([a-z_]+)"|\b([a-z_]+)\b', references)
    }
    return identifiers - aliases - {"and", "coalesce", "is", "not", "null", "where"}


def test_init_sql_documents_existing_scrape_run_state_contract():
    sql = (ROOT / "supabase_setup" / "init.sql").read_text()

    assert 'CREATE TABLE IF NOT EXISTS "public"."scrape_run_state" (' in sql
    assert '"last_successful_scrape_at" timestamp with time zone' in sql
    assert (
        'DROP POLICY IF EXISTS "service_role_singleton_watermark_access" '
        'ON "public"."scrape_run_state";'
    ) in sql
    assert re.search(
        r'CREATE POLICY "service_role_singleton_watermark_access"\s+'
        r'ON "public"\."scrape_run_state"\s+'
        r'FOR ALL\s+TO service_role\s+'
        r'USING \("id" = 1\)\s+'
        r'WITH CHECK \("id" = 1\);',
        sql,
    )
    assert 'REVOKE ALL ON TABLE "public"."scrape_run_state" FROM PUBLIC, anon, authenticated;' in sql
    assert 'GRANT ALL ON TABLE "public"."scrape_run_state" TO service_role;' in sql


def test_scrape_run_state_migration_is_idempotent_and_private():
    sql = (ROOT / "supabase_setup" / "add_scrape_run_state.sql").read_text()
    assert "CREATE TABLE IF NOT EXISTS public.scrape_run_state" in sql
    assert "ON CONFLICT (id) DO NOTHING" in sql
    assert "ALTER TABLE public.scrape_run_state ENABLE ROW LEVEL SECURITY" in sql
    assert (
        "DROP POLICY IF EXISTS service_role_singleton_watermark_access "
        "ON public.scrape_run_state;"
    ) in sql
    assert re.search(
        r"CREATE POLICY service_role_singleton_watermark_access\s+"
        r"ON public\.scrape_run_state\s+"
        r"FOR ALL\s+TO service_role\s+"
        r"USING \(id = 1\)\s+"
        r"WITH CHECK \(id = 1\);",
        sql,
    )
    assert "REVOKE ALL ON TABLE public.scrape_run_state FROM PUBLIC, anon, authenticated" in sql
    assert "GRANT ALL ON TABLE public.scrape_run_state TO service_role" in sql


def test_scrape_success_rpc_is_secure_idempotent_and_service_role_only():
    for filename in ("init.sql", "add_scrape_run_state.sql"):
        sql = (ROOT / "supabase_setup" / filename).read_text()
        normalized = sql.replace('"', "").lower()

        assert "create or replace function public.record_scrape_success" in normalized
        assert "p_finished_at timestamp with time zone" in normalized or (
            "p_finished_at timestamptz" in normalized
        )
        assert "returns timestamp with time zone" in normalized or "returns timestamptz" in normalized
        assert "security definer" in normalized
        assert "set search_path = pg_catalog" in normalized
        assert "insert into public.scrape_run_state (id, last_successful_scrape_at)" in normalized
        assert "values (1, p_finished_at)" in normalized
        assert "on conflict (id) do update" in normalized
        assert "returning last_successful_scrape_at into persisted_at" in normalized
        assert re.search(
            r"revoke all on function public\.record_scrape_success\("
            r"(?:timestamp with time zone|timestamptz)\) from public, anon, authenticated;",
            normalized,
        )
        assert re.search(
            r"grant execute on function public\.record_scrape_success\("
            r"(?:timestamp with time zone|timestamptz)\) to service_role;",
            normalized,
        )
        assert "p_run_id" not in normalized


def test_scrape_run_state_policy_does_not_broaden_other_internal_tables():
    for filename in ("init.sql", "add_scrape_run_state.sql"):
        sql = (ROOT / "supabase_setup" / filename).read_text()
        policies = re.findall(
            r"CREATE POLICY\s+\"?service_role_singleton_watermark_access\"?\s+"
            r"ON\s+([^\s;]+)",
            sql,
            re.IGNORECASE,
        )

        assert policies in (["public.scrape_run_state"], ['"public"."scrape_run_state"'])


def test_scrape_run_state_policy_precedes_idempotent_singleton_seed():
    for filename in ("init.sql", "add_scrape_run_state.sql"):
        sql = (ROOT / "supabase_setup" / filename).read_text()
        seed = (
            'INSERT INTO "public"."scrape_run_state" ("id")'
            if filename == "init.sql"
            else "INSERT INTO public.scrape_run_state (id)"
        )

        assert sql.index("CREATE POLICY") < sql.index(seed)


def test_cleanup_sql_drops_job_keyword_insights_table():
    sql = (ROOT / "supabase_setup" / "cleanup.sql").read_text()

    assert 'DROP TABLE IF EXISTS "public"."job_keyword_insights" CASCADE;' in sql
    assert 'DROP TABLE IF EXISTS "public"."keyword_insights" CASCADE;' in sql
    assert 'DROP TABLE IF EXISTS "public"."jobs" CASCADE;' in sql
    assert (
        'DROP FUNCTION IF EXISTS "public"."replace_historical_repost_plan"(jsonb) CASCADE;'
        in sql
    )
    assert (
        'DROP FUNCTION IF EXISTS "public"."replace_job_keyword_facts_and_refresh_aggregates"'
        '(text[], text, jsonb) CASCADE;'
    ) in sql
    assert 'DROP FUNCTION IF EXISTS "public"."rebuild_keyword_insights_atomic"() CASCADE;' in sql
    assert 'DROP FUNCTION IF EXISTS "public"."update_last_updated_column"() CASCADE;' in sql


def test_add_job_insights_sql_is_self_contained_for_last_updated_trigger():
    sql = (ROOT / "supabase_setup" / "add_job_insights.sql").read_text()

    assert 'CREATE OR REPLACE FUNCTION "public"."update_last_updated_column"() RETURNS "trigger"' in sql


def test_incremental_keyword_insights_rpc_is_bounded_atomic_and_service_role_only():
    migration = (ROOT / "supabase_setup" / "add_incremental_keyword_insights_rpc.sql").read_text()
    init = (ROOT / "supabase_setup" / "init.sql").read_text()

    for sql in (migration, init):
        assert "CREATE OR REPLACE FUNCTION public.rebuild_keyword_insights_atomic()" in sql
        assert "CREATE OR REPLACE FUNCTION public.replace_job_keyword_facts_and_refresh_aggregates(" in sql
        assert "SECURITY DEFINER" in sql
        assert "SET search_path = pg_catalog, public" in sql
        assert "cardinality(p_job_ids) > 1000" in sql
        assert "jsonb_array_length(p_facts) > 50000" in sql
        assert sql.count("hashtextextended('keyword-insights-aggregate-global', 0)") >= 3
        assert sql.index("SELECT public.rebuild_keyword_insights_atomic();") < sql.index(
            "CREATE OR REPLACE FUNCTION public.replace_job_keyword_facts_and_refresh_aggregates("
        )
        assert "DELETE FROM public.keyword_insights;" in sql
        assert "GROUP BY archetype, COALESCE(provider, 'unknown'), keyword, category" in sql
        assert "SET LOCAL statement_timeout = '15min';" in sql
        assert "LOCK TABLE public.job_keyword_insights, public.keyword_insights" in sql
        assert "IN ACCESS EXCLUSIVE MODE;" in sql
        assert "GREATEST(0, count + (v_delta->>'delta')::integer)" in sql
        assert "AND count = 0" in sql
        assert (
            "REVOKE ALL ON FUNCTION public.replace_job_keyword_facts_and_refresh_aggregates"
            "(text[], text, jsonb) FROM PUBLIC, anon, authenticated;"
        ) in sql
        assert (
            "GRANT EXECUTE ON FUNCTION public.replace_job_keyword_facts_and_refresh_aggregates"
            "(text[], text, jsonb) TO service_role;"
        ) in sql
        assert (
            "REVOKE ALL ON FUNCTION public.rebuild_keyword_insights_atomic()"
            "\nFROM PUBLIC, anon, authenticated;"
        ) in sql
        assert "GRANT EXECUTE ON FUNCTION public.rebuild_keyword_insights_atomic() TO service_role;" in sql
        assert "ALTER FUNCTION public.rebuild_keyword_insights_atomic() OWNER TO postgres;" in sql
        normalized = re.sub(r'["\s]', "", sql).lower()
        assert (
            "revokeallprivilegesontablepublic.keyword_insights,"
            "public.job_keyword_insightsfrompublic,anon,authenticated,service_role;"
            in normalized
        )
        assert (
            "grantselectontablepublic.keyword_insights,public.job_keyword_insights"
            "toanon,authenticated,service_role;"
            in normalized
        )


def test_historical_merge_serializes_keyword_fact_moves_with_aggregate_writers():
    sql = (ROOT / "supabase_setup" / "merge_historical_reposts.sql").read_text()
    body = _function_body(sql, "merge_historical_repost_plan")

    assert body.index("hashtextextended('keyword-insights-aggregate-global',0)") < body.index(
        "INSERTINTOpublic.job_keyword_insights"
    )
    assert body.index("INSERTINTOpublic.job_keyword_insights") < body.index(
        "DELETEFROMpublic.keyword_insights"
    )
    assert body.index("DELETEFROMpublic.keyword_insights") < body.index(
        "RETURNQUERYSELECTgroup_count,deleted_count"
    )
    assert "FROMpublic.job_keyword_insightsGROUPBYarchetype," in body


def test_historical_merge_staging_and_job_mutations_are_race_safe():
    sql = (ROOT / "supabase_setup" / "merge_historical_reposts.sql").read_text()
    replace_body = _function_body(sql, "replace_historical_repost_plan")
    merge_body = _function_body(sql, "merge_historical_repost_plan")
    plan_lock = "hashtextextended('historical-repost-plan-global',0)"

    assert plan_lock in replace_body
    assert replace_body.index(plan_lock) < replace_body.index(
        "DELETEFROMpublic.job_repost_merge_plan"
    )
    assert "jsonb_typeof(p_plan)<>'array'" in replace_body
    assert "jsonb_array_length(p_plan)>50000" in replace_body
    assert "HAVINGcount(*)>1" in replace_body
    assert plan_lock in merge_body
    assert merge_body.index(plan_lock) < merge_body.index("FORUPDATEOFjobs")
    assert merge_body.index("FORUPDATEOFjobs") < merge_body.index(
        "SELECTDISTINCTsurvivor_job_id"
    )
    assert "SELECTsource_job_idFROMpublic.job_repost_merge_planUNIONSELECTsurvivor_job_id" in merge_body
    assert "ORDERBYjobs.job_idFORUPDATEOFjobs" in merge_body

    assert "SECURITY DEFINER" in sql[sql.index(
        "CREATE OR REPLACE FUNCTION public.replace_historical_repost_plan"
    ):sql.index("CREATE OR REPLACE FUNCTION public.merge_historical_repost_plan")]
    assert (
        "REVOKE ALL ON FUNCTION public.replace_historical_repost_plan(jsonb) "
        "FROM PUBLIC, anon, authenticated;"
    ) in sql
    assert (
        "GRANT EXECUTE ON FUNCTION public.replace_historical_repost_plan(jsonb) "
        "TO service_role;"
    ) in sql
    assert (
        "REVOKE ALL ON TABLE public.job_repost_merge_plan "
        "FROM PUBLIC, anon, authenticated, service_role;"
    ) in sql


def test_historical_merge_function_is_aligned_between_migration_and_init():
    migration = (ROOT / "supabase_setup" / "merge_historical_reposts.sql").read_text()
    init = (ROOT / "supabase_setup" / "init.sql").read_text()

    assert _function_body(migration, "merge_historical_repost_plan") == _function_body(
        init, "merge_historical_repost_plan"
    )
    assert _function_body(migration, "replace_historical_repost_plan") == _function_body(
        init, "replace_historical_repost_plan"
    )


def test_keyword_insight_tables_are_select_only_for_api_roles_in_all_deploy_paths():
    filenames = (
        "init.sql",
        "add_job_insights.sql",
        "add_incremental_keyword_insights_rpc.sql",
        "merge_historical_reposts.sql",
    )

    for filename in filenames:
        sql = (ROOT / "supabase_setup" / filename).read_text()
        normalized = re.sub(r'["\s]', "", sql).lower()
        assert not re.search(
            r"grantall(?:privileges)?ontablepublic\.(?:job_)?keyword_insights",
            normalized,
        ), filename
        assert (
            "revokeallprivilegesontablepublic.keyword_insights,"
            "public.job_keyword_insightsfrompublic,anon,authenticated,service_role;"
            in normalized
        ), filename
        assert (
            "grantselectontablepublic.keyword_insights,public.job_keyword_insights"
            "toanon,authenticated,service_role;"
            in normalized
        ), filename

    all_sql = "\n".join(
        path.read_text() for path in (ROOT / "supabase_setup").glob("*.sql")
    )
    assert not re.search(
        r'GRANT\s+ALL(?:\s+PRIVILEGES)?\s+ON\s+(?:TABLE\s+)?'
        r'(?:(?:"?public"?)\.)?"?(?:job_)?keyword_insights"?',
        all_sql,
        re.IGNORECASE,
    )


def test_init_sql_uses_if_not_exists_for_setup_indexes():
    sql = (ROOT / "supabase_setup" / "init.sql").read_text()

    assert 'CREATE INDEX IF NOT EXISTS "idx_jobs_insights_analyzed_at"' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_jobs_insights_reanalyzed_at"' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_keyword_insights_category"' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_keyword_insights_count"' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_job_keyword_insights_job_id"' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_job_keyword_insights_keyword_category"' in sql


def test_relist_and_freehire_security_and_snapshot_guards():
    relist = (ROOT / "supabase_setup" / "add_same_id_relist_tracking.sql").read_text()
    freehire = (ROOT / "supabase_setup" / "add_freehire_compat.sql").read_text()
    merge = (ROOT / "supabase_setup" / "merge_historical_reposts.sql").read_text()
    init = (ROOT / "supabase_setup" / "init.sql").read_text()

    assert "extensions.digest(" in relist
    assert "pending_relist_on date" in relist
    assert "REVOKE UPDATE, DELETE ON TABLE public.listing_observations FROM service_role" in relist
    assert "p_expected_source_snapshot jsonb" in freehire
    assert "p_expected_source_snapshot <@ to_jsonb(j)" in freehire
    assert "BEFORE UPDATE ON public.jobs" in freehire
    assert not re.search(r"\bTRUNCATE\s+(?:TABLE\s+)?public\.jobs\b", merge)
    assert "LOCK TABLE public.base_resume" not in init
    for required in (
        "calculate_listing_posting_waves",
        "apply_linkedin_relist_projection",
        "apply_same_id_relist_repair",
        "claim_freehire_compat_job",
        "persist_freehire_compat_result",
        "apply_freehire_compat_metadata",
        "jobs_invalidate_freehire_compat_input",
    ):
        assert required in init


def test_same_id_repair_rpc_has_jsonb_cas_and_service_role_only_execute():
    init = (ROOT / "supabase_setup" / "init.sql").read_text()
    migration = (ROOT / "supabase_setup" / "add_same_id_relist_tracking.sql").read_text()
    backfill = (ROOT / "backfill_same_id_relists.py").read_text()

    for sql in (init, migration):
        assert "CREATE OR REPLACE FUNCTION public.apply_same_id_relist_repair(" in sql
        assert "SECURITY DEFINER" in sql
        assert "listing_instances IS NOT DISTINCT FROM p_expected_listing_instances" in sql
        assert "last_seen_at IS NOT DISTINCT FROM p_expected_last_seen_at" in sql
        assert "REVOKE ALL ON FUNCTION public.apply_same_id_relist_repair(text, jsonb, timestamptz, jsonb) FROM PUBLIC, anon, authenticated;" in sql
        assert "GRANT EXECUTE ON FUNCTION public.apply_same_id_relist_repair(text, jsonb, timestamptz, jsonb) TO service_role;" in sql

    assert 'supabase_utils.supabase.rpc("apply_same_id_relist_repair"' in backfill
    assert '.eq("listing_instances"' not in backfill
    assert "json.dumps" not in backfill


def test_publication_snapshot_contract_is_idempotent_bounded_and_least_privilege():
    migration = (ROOT / "supabase_setup" / "add_freehire_publication_snapshots.sql").read_text()
    init = (ROOT / "supabase_setup" / "init.sql").read_text()

    for sql in (migration, init):
        assert "CREATE TABLE IF NOT EXISTS public.freehire_publication_state" in sql
        assert "CREATE TABLE IF NOT EXISTS public.freehire_publication_generations" in sql
        assert "CREATE TABLE IF NOT EXISTS public.freehire_publication_snapshots" in sql
        assert "PRIMARY KEY (generation, canonical_job_id)" in sql
        assert "import_hash text NOT NULL" in sql
        assert "payload jsonb NOT NULL" in sql
        assert "SECURITY DEFINER" in sql
        assert "SET search_path = pg_catalog" in sql
        assert "LOCK TABLE public.jobs IN SHARE MODE" in sql
        assert "source_count = 0" in sql
        assert "same-watermark publication generation % failed integrity validation" in sql
        assert "p_source_scrape_watermark > authoritative_watermark" in sql
        assert "p_source_scrape_watermark < authoritative_watermark" in sql
        assert "LIMIT LEAST(p_page_size, 1000)" in sql
        assert "CREATE OR REPLACE FUNCTION public.prune_freehire_publication_generations" in sql
        assert "DEFERRABLE INITIALLY DEFERRED" in sql
        assert "NOT VALID" in sql
        assert "p_max_generations integer DEFAULT 3" in sql
        assert "SET statement_timeout = '60s'" in sql
        assert "FROM public.freehire_jobs AS source" in sql
        assert "pg_catalog.to_jsonb(source)" in sql
        assert "CREATE ROLE freehire_publication_reader" in sql
        assert "FROM PUBLIC, anon, authenticated, service_role, freehire_publication_reader" in sql
        assert "GRANT USAGE ON SCHEMA public TO freehire_publication_reader" in sql
        assert "finalize_freehire_publication(timestamptz) FROM freehire_publication_reader" in sql
        assert "get_freehire_publication_state() TO service_role" not in sql
        assert "get_freehire_publication_page(bigint, text, integer) TO service_role" not in sql

    assert "SET LOCAL lock_timeout = '10s'" in migration
    assert "SET LOCAL statement_timeout = '5min'" in migration
    assert "SET LOCAL idle_in_transaction_session_timeout = '6min'" in migration
    assert init.index('CREATE OR REPLACE VIEW "public"."freehire_jobs"') < init.index(
        "CREATE TABLE IF NOT EXISTS public.freehire_publication_state"
    )


def test_finalize_contract_is_transactional_idempotent_and_updates_state_after_copy():
    sql = (ROOT / "supabase_setup" / "add_freehire_publication_snapshots.sql").read_text()
    idempotency = "current_state.source_scrape_watermark = p_source_scrape_watermark"
    copy = "INSERT INTO public.freehire_publication_snapshots"
    completion = "INSERT INTO public.freehire_publication_generations"
    update = "UPDATE public.freehire_publication_state AS state"

    assert sql.index(idempotency) < sql.index(copy)
    assert sql.index(copy) < sql.index(completion, sql.index(copy)) < sql.index(update)
    finalize_body = _publication_function_body(
        sql, "finalize_freehire_publication", "function"
    )
    assert "DELETE FROM public.freehire_publication_generations" not in finalize_body
    assert "FOR UPDATE" in sql
    assert "p_source_scrape_watermark IS NULL" in sql
    assert "source scrape watermark cannot move backwards" in sql
    assert "GET DIAGNOSTICS copied_count = ROW_COUNT" in sql


def test_init_and_standalone_publication_rpc_definitions_are_exactly_aligned():
    migration = (ROOT / "supabase_setup" / "add_freehire_publication_snapshots.sql").read_text()
    init = (ROOT / "supabase_setup" / "init.sql").read_text()

    for name in (
        "finalize_freehire_publication",
        "get_freehire_publication_state",
        "get_freehire_publication_page",
    ):
        assert _publication_function_body(
            migration, name, "function"
        ) == _publication_function_body(init, name, "publication_hardened")


def test_init_sql_adds_job_archetype_provenance_columns():
    sql = (ROOT / "supabase_setup" / "init.sql").read_text()

    assert 'ADD COLUMN IF NOT EXISTS "search_query" text' in sql
    assert 'ADD COLUMN IF NOT EXISTS "archetype" text' in sql
    assert 'ADD COLUMN IF NOT EXISTS "filter_profile" text' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_jobs_archetype"' in sql


def test_add_job_insights_sql_scopes_fact_and_aggregate_tables_by_archetype():
    sql = (ROOT / "supabase_setup" / "add_job_insights.sql").read_text()

    assert '"archetype" text NOT NULL' in sql
    assert 'PRIMARY KEY ("archetype", "provider", "keyword", "category")' in sql
    assert 'PRIMARY KEY ("job_id", "archetype", "keyword", "category")' in sql


def test_add_job_insights_sql_migrates_existing_keyword_insights_table():
    sql = (ROOT / "supabase_setup" / "add_job_insights.sql").read_text()

    assert 'ALTER TABLE "public"."keyword_insights"' in sql
    assert 'ADD COLUMN IF NOT EXISTS "archetype" text' in sql
    assert 'ADD COLUMN IF NOT EXISTS "provider" text' in sql
    assert "UPDATE \"public\".\"keyword_insights\"" in sql
    assert "SET \"archetype\" = 'software_tpm'" in sql
    assert "SET \"provider\" = 'unknown'" in sql
    assert 'ALTER COLUMN "provider" SET NOT NULL' in sql
    assert 'DROP CONSTRAINT IF EXISTS "keyword_insights_pkey"' in sql
    assert 'ADD CONSTRAINT "keyword_insights_pkey" PRIMARY KEY ("archetype", "provider", "keyword", "category")' in sql


def test_add_job_insights_sql_migrates_existing_job_keyword_insights_table():
    sql = (ROOT / "supabase_setup" / "add_job_insights.sql").read_text()

    assert 'ALTER TABLE "public"."job_keyword_insights"' in sql
    assert 'ADD COLUMN IF NOT EXISTS "archetype" text' in sql
    assert 'ADD COLUMN IF NOT EXISTS "provider" text' in sql
    assert "UPDATE \"public\".\"job_keyword_insights\"" in sql
    assert "SET \"archetype\" = 'software_tpm'" in sql
    assert 'DROP CONSTRAINT IF EXISTS "job_keyword_insights_pkey"' in sql
    assert 'ADD CONSTRAINT "job_keyword_insights_pkey" PRIMARY KEY ("job_id", "archetype", "keyword", "category")' in sql


def test_init_sql_scopes_insight_tables_by_archetype_for_fresh_installs():
    sql = (ROOT / "supabase_setup" / "init.sql").read_text()

    assert 'CREATE TABLE IF NOT EXISTS "public"."keyword_insights" (' in sql
    assert 'CREATE TABLE IF NOT EXISTS "public"."job_keyword_insights" (' in sql
    assert '"archetype" text NOT NULL' in sql
    assert 'ADD CONSTRAINT "keyword_insights_pkey" PRIMARY KEY ("archetype", "provider", "keyword", "category")' in sql
    assert 'ADD CONSTRAINT "job_keyword_insights_pkey" PRIMARY KEY ("job_id", "archetype", "keyword", "category")' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_keyword_insights_archetype_category"' in sql
    assert 'CREATE INDEX IF NOT EXISTS "idx_job_keyword_insights_archetype"' in sql
    assert 'ADD CONSTRAINT "keyword_insights_pkey" PRIMARY KEY ("keyword", "category")' not in sql
    assert 'ADD CONSTRAINT "job_keyword_insights_pkey" PRIMARY KEY ("job_id", "keyword", "category")' not in sql


def test_init_sql_actionable_rpcs_exclude_filtered_jobs_consistently():
    sql = (ROOT / "supabase_setup" / "init.sql").read_text()

    assert 'AND j.is_filtered = FALSE\n        AND j.customized_resume_id IS NOT NULL' in sql
    assert 'AND j.is_filtered = FALSE\n        AND j.resume_score >= 50' in sql
    assert sql.count('AND j.is_filtered = FALSE') >= 3


def test_top_scored_jobs_rpc_has_only_canonical_seven_arg_signature():
    init_sql = (ROOT / "supabase_setup" / "init.sql").read_text()
    migration_sql = (
        ROOT / "supabase_setup" / "drop_legacy_top_scored_jobs_overload.sql"
    ).read_text()
    function_name = '"public"."get_top_scored_jobs_custom_sort"'
    legacy_named_signature = (
        f'{function_name}("p_page_number" integer, "p_page_size" integer, '
        '"p_provider" "text", "p_min_score" integer, "p_max_score" integer, '
        '"p_is_interested_option" "text")'
    )
    canonical_definition = (
        f'CREATE OR REPLACE FUNCTION {function_name}("p_page_number" integer, '
        '"p_page_size" integer, "p_provider" "text" DEFAULT NULL::"text", '
        '"p_min_score" integer DEFAULT 50, "p_max_score" integer DEFAULT 100, '
        '"p_is_interested_option" "text" DEFAULT NULL::"text", '
        '"p_search_query" "text" DEFAULT NULL::"text")'
    )
    drop_legacy_signature = (
        f'DROP FUNCTION IF EXISTS {function_name}'
        '(integer, integer, text, integer, integer, text);'
    )

    assert init_sql.count(canonical_definition) == 1
    assert legacy_named_signature not in init_sql
    assert drop_legacy_signature in init_sql
    assert drop_legacy_signature in migration_sql
    assert migration_sql.startswith("BEGIN;")
    assert migration_sql.rstrip().endswith("COMMIT;")


def test_posting_wave_migration_adds_count_and_shared_calculator():
    sql = (ROOT / "supabase_setup" / "add_posting_wave_semantics.sql").read_text()

    assert "ADD COLUMN IF NOT EXISTS posting_wave_count" in sql
    assert "CREATE OR REPLACE FUNCTION public.calculate_listing_posting_waves" in sql
    assert "instance->>'location'" in sql
    assert "instance->>'scrape_run_id'" in sql
    assert "GREATEST(result.posting_wave_count - 1, 0)" in sql
    assert "GRANT EXECUTE ON FUNCTION public.calculate_listing_posting_waves(jsonb) TO service_role" in sql


def test_historical_merge_uses_location_and_posting_wave_calculator():
    sql = (ROOT / "supabase_setup" / "merge_historical_reposts.sql").read_text()

    assert "'location', source_snapshot->>'location'" in sql
    assert "public.calculate_listing_posting_waves(lv.raw_listing_instances)" in sql
    assert "posting_wave_count = w.posting_wave_count" in sql
    assert "repost_count = w.repost_count" in sql
    assert "GREATEST(l.seen_count - 1, 0)" not in sql


def test_same_id_relist_schema_is_idempotent_append_only_and_service_role_only():
    init_sql = (ROOT / "supabase_setup" / "init.sql").read_text()
    migration_sql = (ROOT / "supabase_setup" / "add_same_id_relist_tracking.sql").read_text()

    for sql in (init_sql, migration_sql):
        assert 'CREATE TABLE IF NOT EXISTS "public"."ingestion_runs"' in sql or "CREATE TABLE IF NOT EXISTS public.ingestion_runs" in sql
        assert 'CREATE TABLE IF NOT EXISTS "public"."listing_observations"' in sql or "CREATE TABLE IF NOT EXISTS public.listing_observations" in sql
        assert 'CREATE TABLE IF NOT EXISTS "public"."listing_content_versions"' in sql or "CREATE TABLE IF NOT EXISTS public.listing_content_versions" in sql
        assert 'CREATE TABLE IF NOT EXISTS "public"."listing_relist_events"' in sql or "CREATE TABLE IF NOT EXISTS public.listing_relist_events" in sql
        assert 'CREATE TABLE IF NOT EXISTS "public"."listing_states"' in sql or "CREATE TABLE IF NOT EXISTS public.listing_states" in sql
        assert "same_id_relist_count" in sql
        assert "relisted at least once" in sql
        assert "ENABLE ROW LEVEL SECURITY" in sql
        assert "listing_observations_append_only" in sql
        assert "query_scope" in sql
        assert "pending_relist_on" in sql
        assert "SET search_path = pg_catalog, public" in sql
        assert (
            "UNIQUE (provider, source_job_id, ingestion_run_id, query_scope, result)" in sql
            or 'UNIQUE ("provider", "source_job_id", "ingestion_run_id", "query_scope", "result")' in sql
        )
        assert "listing_relist_events_idempotency" in sql
        assert "service_role" in sql


def test_cleanup_drops_same_id_relist_ledger_tables():
    sql = (ROOT / "supabase_setup" / "cleanup.sql").read_text()
    for table in (
        "listing_relist_events",
        "listing_content_versions",
        "listing_observations",
        "listing_states",
        "ingestion_runs",
    ):
        assert f'DROP TABLE IF EXISTS "public"."{table}" CASCADE;' in sql


def test_freehire_contract_migration_has_pinned_checks_and_safe_publication_view():
    sql = (ROOT / "supabase_setup" / "add_freehire_compat.sql").read_text()
    init_sql = (ROOT / "supabase_setup" / "init.sql").read_text()
    assert 'ADD COLUMN IF NOT EXISTS "freehire_category" text' in init_sql
    assert 'CREATE OR REPLACE VIEW "public"."freehire_jobs"' in init_sql
    assert "ADD COLUMN IF NOT EXISTS freehire_category text" in sql
    assert "'software_engineering'" in sql and "'customer_success'" in sql
    assert "freehire_seniority IN ('', 'intern', 'junior', 'middle', 'senior', 'lead', 'staff', 'principal', 'c_level')" in sql
    assert "freehire_compat_status IN ('pending', 'processing', 'current', 'failed')" in sql
    assert "CREATE OR REPLACE VIEW public.freehire_jobs" in sql
    assert "COALESCE(latest_job_id, job_id) AS live_listing_id" in sql
    assert "freehire_compat_status = 'current'" in sql
    for private_field in ("application_date", "resume_score", "is_interested", "customized_resume_id"):
        view = sql.split("CREATE OR REPLACE VIEW public.freehire_jobs AS", 1)[1]
        assert private_field not in view


def test_freehire_view_sources_exist_in_jobs_ddl_and_match_init():
    init_sql = (ROOT / "supabase_setup" / "init.sql").read_text()
    migration_sql = (ROOT / "supabase_setup" / "add_freehire_compat.sql").read_text()
    jobs_columns = _jobs_columns(init_sql)
    init_sources = _freehire_view_source_columns(init_sql)
    migration_sources = _freehire_view_source_columns(migration_sql)

    assert init_sources <= jobs_columns, init_sources - jobs_columns
    assert migration_sources <= jobs_columns, migration_sources - jobs_columns
    assert migration_sources == init_sources


def test_remediation_migrations_are_transactional_and_define_atomic_rpcs():
    freehire = (ROOT / "supabase_setup" / "add_freehire_compat.sql").read_text()
    relists = (ROOT / "supabase_setup" / "add_same_id_relist_tracking.sql").read_text()
    init = (ROOT / "supabase_setup" / "init.sql").read_text()
    for sql in (freehire, relists, init):
        assert sql.lstrip().startswith("BEGIN;")
        assert sql.rstrip().endswith("COMMIT;")
    assert "claim_freehire_compat_job" in freehire
    assert "persist_freehire_compat_result" in freehire
    assert "jobs_invalidate_freehire_compat_input" in freehire
    assert "apply_linkedin_relist_projection" in relists
    assert '"replace_base_resume"' in init
