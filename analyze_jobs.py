import logging
import json
import time
import config
import supabase_utils
from collections import defaultdict
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = genai.Client(api_key=config.GEMINI_FIRST_API_KEY)

BATCH_SIZE = 10       # Job descriptions per Gemini call
SLEEP_BETWEEN = 6     # Seconds between API calls to avoid rate limiting
MAX_JOBS = 200        # Cap to avoid excessive API usage


# --- Pydantic schema for Gemini structured output ---
class KeywordItem(BaseModel):
    keyword: str
    category: str  # 'skill' | 'technology' | 'certification' | 'attribute'

class KeywordList(BaseModel):
    keywords: List[KeywordItem]


SYSTEM_PROMPT = """
You are an expert technical recruiter and resume analyst.
Your job is to extract the most important keywords from job descriptions.

You MUST categorize every keyword into exactly one of these four categories:
- "skill": Soft skills and professional competencies (e.g. "Project Management", "Agile", "Communication", "Leadership")
- "technology": Specific tools, platforms, languages, frameworks, or software (e.g. "Python", "Azure", "Docker", "SAP", "Salesforce")
- "certification": Named certifications, licenses, or credentials (e.g. "PMP", "AWS Solutions Architect", "CISSP", "Scrum Master")
- "attribute": Candidate traits, experience levels, or general qualifications (e.g. "5+ years experience", "Bachelor's degree", "bilingual", "remote work")

Rules:
- Only extract keywords that are explicitly requested or emphasized in the job description
- Normalize keywords to their canonical form (e.g. "MS Azure" -> "Azure", "proj mgmt" -> "Project Management")
- Do not include generic filler words like "team player" unless they are specifically emphasized
- Output ONLY the JSON object, no other text
"""


def fetch_jobs_for_analysis() -> list:
    """Fetch job descriptions from Supabase."""
    supabase = supabase_utils.get_supabase_client()
    response = supabase.from_("jobs") \
        .select("job_id, job_title, description") \
        .eq("is_active", True) \
        .eq("job_state", "new") \
        .not_.is_("description", "null") \
        .limit(MAX_JOBS) \
        .execute()

    if response.data:
        logging.info(f"Fetched {len(response.data)} jobs for analysis.")
        return response.data
    logging.warning("No jobs fetched for analysis.")
    return []


def extract_keywords_from_batch(batch: list) -> List[KeywordItem]:
    """Send a batch of job descriptions to Gemini and extract keywords."""
    combined = ""
    for i, job in enumerate(batch):
        combined += f"\n\n--- JOB {i+1}: {job.get('job_title', 'Unknown')} ---\n{job.get('description', '')}"

    prompt = f"""
Extract all requested skills, technologies, certifications, and candidate attributes from the following {len(batch)} job description(s).

{combined}
"""

    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=SYSTEM_PROMPT,
                response_mime_type='application/json',
                response_schema=KeywordList,
            )
        )
        parsed = KeywordList.model_validate_json(response.text.strip())
        logging.info(f"Extracted {len(parsed.keywords)} keywords from batch of {len(batch)}")
        return parsed.keywords
    except Exception as e:
        logging.error(f"Error extracting keywords from batch: {e}")
        return []


def aggregate_keywords(all_keywords: List[KeywordItem]) -> dict:
    """Aggregate keyword counts by (keyword, category)."""
    counts = defaultdict(int)
    for item in all_keywords:
        key = (item.keyword.strip().title(), item.category.strip().lower())
        counts[key] += 1
    return counts


def save_insights_to_supabase(counts: dict):
    """Upsert aggregated keyword counts into keyword_insights table."""
    supabase = supabase_utils.get_supabase_client()

    # Clear existing data for a clean recompute
    supabase.from_("keyword_insights").delete().neq("id", 0).execute()
    logging.info("Cleared existing keyword_insights.")

    rows = [
        {
            "keyword": keyword,
            "category": category,
            "count": count,
        }
        for (keyword, category), count in counts.items()
        if count >= 2  # Only include keywords appearing in 2+ jobs
    ]

    if not rows:
        logging.warning("No keywords met the minimum frequency threshold.")
        return

    # Insert in batches of 100
    for i in range(0, len(rows), 100):
        chunk = rows[i:i+100]
        supabase.from_("keyword_insights").insert(chunk).execute()

    logging.info(f"Saved {len(rows)} keyword insights to Supabase.")


def run():
    logging.info("Starting job insights analysis...")

    jobs = fetch_jobs_for_analysis()
    if not jobs:
        logging.info("No jobs to analyze. Exiting.")
        return

    all_keywords = []

    # Process in batches
    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i:i + BATCH_SIZE]
        logging.info(f"Processing batch {i // BATCH_SIZE + 1} ({len(batch)} jobs)...")
        keywords = extract_keywords_from_batch(batch)
        all_keywords.extend(keywords)
        if i + BATCH_SIZE < len(jobs):
            logging.info(f"Sleeping {SLEEP_BETWEEN}s before next batch...")
            time.sleep(SLEEP_BETWEEN)

    logging.info(f"Total keywords extracted: {len(all_keywords)}")

    counts = aggregate_keywords(all_keywords)
    logging.info(f"Unique keyword/category pairs: {len(counts)}")

    save_insights_to_supabase(counts)
    logging.info("Insights analysis complete.")


if __name__ == "__main__":
    run()
