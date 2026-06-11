"""
backfill_filter.py
------------------
One-shot script to backfill is_filtered / is_entry_level_filtered / filter_reason
against the entire jobs table.

Run after applying the DB migration in the Supabase SQL Editor:

    ALTER TABLE public.jobs
        ADD COLUMN IF NOT EXISTS is_filtered            boolean DEFAULT false,
        ADD COLUMN IF NOT EXISTS filter_reason          text,
        ADD COLUMN IF NOT EXISTS is_entry_level_filtered boolean DEFAULT false;

    CREATE INDEX IF NOT EXISTS idx_jobs_is_filtered
        ON public.jobs (is_filtered);
    CREATE INDEX IF NOT EXISTS idx_jobs_is_entry_level_filtered
        ON public.jobs (is_entry_level_filtered);

Usage (with env vars set):
    SUPABASE_URL=https://... SUPABASE_SERVICE_ROLE_KEY=... python backfill_filter.py

Or with a .env file — python-dotenv is loaded via config.py.
"""

import logging
import supabase_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    logging.info("=== Starting full-DB filter backfill ===")
    flagged = supabase_utils.flag_filtered_jobs()
    logging.info(f"=== Backfill complete. Total newly flagged: {flagged} ===")
