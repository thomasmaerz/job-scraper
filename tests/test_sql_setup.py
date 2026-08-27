import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def test_cleanup_sql_drops_job_keyword_insights_table():
    sql = (ROOT / "supabase_setup" / "cleanup.sql").read_text()

    assert 'DROP TABLE IF EXISTS "public"."job_keyword_insights" CASCADE;' in sql
    assert 'DROP TABLE IF EXISTS "public"."keyword_insights" CASCADE;' in sql
    assert 'DROP TABLE IF EXISTS "public"."jobs" CASCADE;' in sql
    assert 'DROP FUNCTION IF EXISTS "public"."update_last_updated_column"() CASCADE;' in sql


def test_add_job_insights_sql_is_self_contained_for_last_updated_trigger():
    sql = (ROOT / "supabase_setup" / "add_job_insights.sql").read_text()

    assert 'CREATE OR REPLACE FUNCTION "public"."update_last_updated_column"() RETURNS "trigger"' in sql


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
    assert "TRUNCATE" not in merge
    assert "LOCK TABLE public.base_resume" not in init
    for required in (
        "calculate_listing_posting_waves",
        "apply_linkedin_relist_projection",
        "claim_freehire_compat_job",
        "persist_freehire_compat_result",
        "apply_freehire_compat_metadata",
        "jobs_invalidate_freehire_compat_input",
    ):
        assert required in init


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
