-- Job insights analysis schema migration.
-- Run this once in Supabase SQL Editor for existing databases.

ALTER TABLE "public"."jobs"
ADD COLUMN IF NOT EXISTS "insights_analyzed_at" timestamp with time zone;

CREATE TABLE IF NOT EXISTS "public"."keyword_insights" (
    "keyword" text NOT NULL,
    "category" text NOT NULL,
    "count" integer DEFAULT 0 NOT NULL,
    "last_updated" timestamp with time zone DEFAULT now(),
    CONSTRAINT "keyword_insights_pkey" PRIMARY KEY ("keyword", "category"),
    CONSTRAINT "keyword_insights_category_check"
        CHECK ("category" IN ('skill', 'technology', 'certification', 'attribute')),
    CONSTRAINT "keyword_insights_count_check"
        CHECK ("count" >= 0)
);

CREATE TABLE IF NOT EXISTS "public"."job_keyword_insights" (
    "job_id" text NOT NULL,
    "keyword" text NOT NULL,
    "category" text NOT NULL,
    "analyzed_at" timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT "job_keyword_insights_pkey" PRIMARY KEY ("job_id", "keyword", "category"),
    CONSTRAINT "job_keyword_insights_category_check"
        CHECK ("category" IN ('skill', 'technology', 'certification', 'attribute'))
);

CREATE INDEX IF NOT EXISTS "idx_jobs_insights_analyzed_at"
ON "public"."jobs" USING btree ("insights_analyzed_at");

CREATE INDEX IF NOT EXISTS "idx_keyword_insights_category"
ON "public"."keyword_insights" USING btree ("category");

CREATE INDEX IF NOT EXISTS "idx_keyword_insights_count"
ON "public"."keyword_insights" USING btree ("count" DESC);

CREATE INDEX IF NOT EXISTS "idx_job_keyword_insights_job_id"
ON "public"."job_keyword_insights" USING btree ("job_id");

CREATE INDEX IF NOT EXISTS "idx_job_keyword_insights_keyword_category"
ON "public"."job_keyword_insights" USING btree ("keyword", "category");

CREATE OR REPLACE TRIGGER "update_keyword_insights_last_updated"
BEFORE UPDATE ON "public"."keyword_insights"
FOR EACH ROW EXECUTE FUNCTION "public"."update_last_updated_column"();

ALTER TABLE "public"."keyword_insights" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "public"."job_keyword_insights" ENABLE ROW LEVEL SECURITY;

GRANT ALL ON TABLE "public"."keyword_insights" TO "anon";
GRANT ALL ON TABLE "public"."keyword_insights" TO "authenticated";
GRANT ALL ON TABLE "public"."keyword_insights" TO "service_role";
GRANT ALL ON TABLE "public"."job_keyword_insights" TO "anon";
GRANT ALL ON TABLE "public"."job_keyword_insights" TO "authenticated";
GRANT ALL ON TABLE "public"."job_keyword_insights" TO "service_role";
