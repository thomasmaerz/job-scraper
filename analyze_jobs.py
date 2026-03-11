Skip to content
thomasmaerz
job-scraper
Repository navigation
Code
Pull requests
Actions
Projects
Security
Insights
Settings
Files
Go to file
t
.github
supabase_setup
.gitignore
LICENSE
README.md
analyze_jobs.py
config.py
custom_resume_generator.py
job_manager.py
models.py
parse_resume_with_ai.py
pdf_generator.py
requirements.txt
resume.pdf
resume_parser.py
score_jobs.py
scraper.py
supabase_utils.py
user_agents.py
job-scraper
/
analyze_jobs.py
in
main

Edit

Preview
Indent mode

Spaces
Indent size

4
Line wrap mode

No wrap
Editing analyze_jobs.py file contents
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
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
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = genai.Client(api_key=config.GEMINI_FIRST_API_KEY)

BATCH_SIZE = 10       # Job descriptions per Gemini call
SLEEP_BETWEEN = 6     # Seconds between API calls to avoid rate limiting
MAX_JOBS = 200        # Cap per run to avoid excessive API usage


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


def fetch_unanalyzed_jobs() -> list:
    """Fetch only jobs that have not yet been analyzed."""
    response = supabase_utils.supabase.table(config.SUPABASE_TABLE_NAME) \
        .select("job_id, job_title, description") \
        .eq("is_active", True) \
        .eq("job_state", "new") \
        .is_("insights_analyzed_at", None) \
        .not_.is_("description", None) \
        .limit(MAX_JOBS) \
        .execute()

    if response.data:
        logging.info(f"Fetched {len(response.data)} unanalyzed jobs.")
        return response.data
    logging.info("No new unanalyzed jobs found.")
    return []


def extract_keywords_from_batch(batch: list) -> List[KeywordItem]:
Use Control + Shift + m to toggle the tab key moving focus. Alternatively, use esc then tab to move to the next interactive element on the page.
Editing job-scraper/analyze_jobs.py at main · thomasmaerz/job-scraper
