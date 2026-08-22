import logging

import analyze_jobs
import config
from llm_client import LLMClient


EXPECTED_JOB_IDS = [
    "4446829925",
    "4446841210",
    "4456262665",
    "4456284311",
    "4457597152",
    "4456223861",
    "4457392370",
    "4456297456",
    "4454654037",
    "4454639757",
]


def main():
    db = analyze_jobs._get_db()
    batch = analyze_jobs.fetch_unanalyzed_jobs(
        db=db,
        limit=config.JOB_INSIGHTS_BATCH_SIZE,
        archetype="software_tpm",
        backfill_all=True,
    )
    job_ids = [str(job["job_id"]) for job in batch]
    if job_ids != EXPECTED_JOB_IDS:
        raise ValueError(f"Unexpected batch; refusing diagnostic request: {job_ids}")

    prompt_lines = [
        "Extract keywords for each job posting individually.",
        "Return only structured JSON.",
    ]
    for job in batch:
        prompt_lines.append(f"Job ID: {job.get('job_id', '')}")
        prompt_lines.append(f"Title: {job.get('job_title', '')}")
        prompt_lines.append(f"Description: {job.get('description', '')}")
        prompt_lines.append("")

    client = LLMClient(
        model="gemini/gemma-4-26b-a4b-it",
        api_key=config.LLM_API_KEY,
        max_rpm=config.LLM_MAX_RPM,
        max_retries=0,
        retry_base_delay=0,
        daily_budget=0,
        request_delay=0,
    )
    raw_response = client.generate_content(
        prompt="\n".join(prompt_lines),
        system_prompt=analyze_jobs.SYSTEM_PROMPT,
        reasoning_effort="low",
        response_format=analyze_jobs.JobKeywordResultList,
    )
    parsed = analyze_jobs.parse_keyword_response(raw_response)
    missing_job_ids = [job_id for job_id in EXPECTED_JOB_IDS if job_id not in parsed]
    logging.info("Diagnostic batch job IDs: %s", job_ids)
    logging.info("Raw LLM response:\n%s", raw_response)
    logging.info("Returned job IDs: %s", list(parsed))
    logging.info("Missing job IDs: %s", missing_job_ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
