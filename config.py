import os
from dotenv import load_dotenv

load_dotenv()

# --- DO NOT MODIFY THE BELOW SECTION ---

# =================================================================
# 1. CORE SYSTEM CONFIGURATION (Do Not Modify)
# =================================================================
SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_TABLE_NAME: str = "jobs"
SUPABASE_CUSTOMIZED_RESUMES_TABLE_NAME = "customized_resumes"
SUPABASE_STORAGE_BUCKET="personalized_resumes"
SUPABASE_RESUME_STORAGE_BUCKET="resumes"
SUPABASE_BASE_RESUME_TABLE_NAME = "base_resume"
BASE_RESUME_PATH = "resume.json"

# API keys — set only the key(s) needed for your chosen provider.
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_FIRST_API_KEY")

# =================================================================
# 2. USER PREFERENCES (Editable)
# =================================================================

# --- LLM Settings ---
# Use any model supported by LiteLLM (gemini, openai/gpt-4o-mini, groq/llama-3.3-70b-versatile)
# Full list of supported models & naming: https://docs.litellm.ai/docs/providers
LLM_MODEL = "gemini"

# --- Search Configuration ---
LINKEDIN_SEARCH_QUERIES = ["IT Project Manager", "Technical Project Manager", "Information Technology Project Manager", "Technical Program Manager"]
LINKEDIN_LOCATION = "Canada"
LINKEDIN_GEO_ID = 101174742      # Canada
LINKEDIN_JOB_TYPE = "F" # F=Full-time, C=Contract, P=Part-time, T=Temporary, I=Internship
LINKEDIN_JOB_POSTING_DATE = "r86400" # r86400=Past 24h, r604800=Past week
LINKEDIN_F_WT = 1,2 # 1=Onsite, 2=Remote, 3=Hybrid

CAREERS_FUTURE_SEARCH_QUERIES = []
CAREERS_FUTURE_SEARCH_CATEGORIES = []
CAREERS_FUTURE_SEARCH_EMPLOYMENT_TYPES = []

# --- Filter Configuration ---
# Jobs matching these patterns are flagged is_filtered=True before LLM scoring.
# All regex patterns are case-insensitive. Edit lists here to tune without touching code.

# Block entire company by name (exact substring match, case-insensitive).
COMPANY_BLOCKLIST = [
    r"jobgether",
]

# Block on job title. Matched before reading description (cheaper).
TITLE_BLOCKLIST = [
    r"\bconstruction\b",          # "Construction and Redevelopment", "Industrial Construction"
    r"\bland development\b",      # WSP Senior PM, Land Development
    r"\bm&e\b",                   # EBC Senior PM - M&E (Mechanical & Electrical)
    r"\bsubcontract\b",           # L3Harris Subcontract Management Lead
    r"\bsales manager\b",         # Nice Group Regional Sales Manager
    r"\baccount manager\b",       # Varicent Technical Account Manager
    r"\bcustomer success\b",      # AON3D Director, Customer Success
    r"\bICI\b",                   # Clark Builders (Institutional/Commercial/Industrial construction)
    r"\bclinical\b",              # Worldwide Clinical Trials
    r"\bCNS\b",                   # Central Nervous System — clinical research
]

# Entry-level / below-PM titles. Sets is_entry_level_filtered=True (and is_filtered=True).
# UI exposes these separately — pivot from IT PM to coordinator is feasible; to Sr construction PM is not.
TITLE_ENTRY_LEVEL_BLOCKLIST = [
    r"\bcoordinator\b",
    r"\bjr\.?\s+(project|program)\b",
    r"\bjunior\b",
    r"\bassistant project manager\b",
    r"\bstaff engineer\b",
]

# Block on full job description. Catches generic titles ("Senior Project Manager") at non-IT firms.
DESC_BLOCKLIST = [
    # Construction / civil / physical infrastructure
    r"construction firm",
    r"construction company",
    r"infrastructure construction",
    r"industrial construction",
    r"\bEPCM?\b",                           # Engineering, Procurement, Construction (Management)
    r"\bEPC environment\b",
    r"civil engineering",
    r"natural and built assets",            # Arcadis signature phrase
    r"\bsubtrades?\b",                      # construction subcontracting term
    # subcontractor alone is too broad — tech companies (e.g. Bosch security systems) also use the term
    # for field installation vendors. Require co-occurrence with an unambiguous physical construction signal.
    r"(?s)\bsubcontractor.{0,3000}\b(?:general contractor|preconstruction|site inspection|specialty contractor|construction management)|\b(?:general contractor|preconstruction|site inspection|specialty contractor|construction management).{0,3000}\bsubcontractor",
    r"\bpreconstruction\b",                 # physical construction phase term — never in IT PM context
    r"\bgeneral contractor\b",              # unambiguous — no IT PM role describes their employer this way
    r"\bMEP\b",                             # Mechanical, Electrical, Plumbing
    r"nuclear facility|nuclear mega.?project",
    r"parliamentary precinct",              # House of Commons real property programs
    r"real property programs",
    r"\bProcore\b",                         # construction PM software — never appears in IT PM roles
    r"tenant improvement",                  # commercial fit-out construction (TI projects)
    r"building restoration",                # building science domain — restoration/envelope consulting
    r"construction administration",         # managing contractors during build phase
    r"\bshop drawings\b",                   # construction document type, never IT
    r"mechanical construction",
    r"plumbing and mechanical",
    # Non-IT sectors
    r"aerospace.*defense|defense.*aerospace",
    r"water purif|waste\s?water treatment",
    r"multifamily property",
    r"property management company",
    r"resource.sector client",
    r"pharmaceutical advertising",
    r"healthcare communications agency",
]

# --- Processing Limits ---
SCRAPING_SOURCES = ["linkedin"] # "linkedin", "careers_future"
JOBS_TO_SCORE_PER_RUN = 5
JOBS_TO_CUSTOMIZE_PER_RUN = 1
MAX_JOBS_PER_SEARCH = {
    "linkedin": None,
    "careers_future": 10,
}

# =================================================================
# 3. ADVANCED SYSTEM SETTINGS (Modify with Caution)
# =================================================================
LLM_MAX_RPM = 10
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 10
LLM_DAILY_REQUEST_BUDGET = 0
LLM_REQUEST_DELAY_SECONDS = 8

JOB_SCORING_MODEL_CHAIN = [
    "gemini/gemini-3.1-flash-lite",
    "gemini/gemma-4-31b-it",
    "gemini/gemini-3-flash-preview",
    "gemini/gemma-4-26b-a4b-it",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
]

JOB_INSIGHTS_MODEL_CHAIN = [
    "gemini/gemini-3.1-flash-lite",
    "gemini/gemma-4-26b-a4b-it",
    "gemini/gemma-4-31b-it",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-3-flash-preview",
]

# --- Job Insights Analysis ---
JOB_INSIGHTS_MAX_JOBS = 200
JOB_INSIGHTS_BATCH_SIZE = 10
JOB_INSIGHTS_SLEEP_SECONDS = 6
JOB_INSIGHTS_MAX_RETRIES = 3
JOB_INSIGHTS_DB_PAGE_SIZE = 1000
JOB_INSIGHTS_UPSERT_BATCH_SIZE = 500

LINKEDIN_MAX_START = 30
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 15

JOB_EXPIRY_DAYS = 30
JOB_CHECK_DAYS = 3
JOB_DELETION_DAYS = 60
JOB_CHECK_LIMIT = 50
ACTIVE_CHECK_TIMEOUT = 20
ACTIVE_CHECK_MAX_RETRIES = 2
ACTIVE_CHECK_RETRY_DELAY = 10
