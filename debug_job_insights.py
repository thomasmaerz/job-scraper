import logging
import os

import analyze_jobs
import config
from llm_client import LLMClient


def main():
    job_id = os.environ["JOB_ID"]
    db = analyze_jobs._get_db()
    job = (
        db.table(config.SUPABASE_TABLE_NAME)
        .select("job_id, job_title, description, archetype, provider")
        .eq("job_id", job_id)
        .single()
        .execute()
        .data
    )
    if not job:
        raise ValueError(f"Job not found: {job_id}")

    client = LLMClient(
        model="gemini/gemma-4-26b-a4b-it",
        api_key=config.LLM_API_KEY,
        max_rpm=config.LLM_MAX_RPM,
        max_retries=0,
        retry_base_delay=0,
        daily_budget=0,
        request_delay=0,
    )
    prompt = "\n".join(
        [
            "Extract keywords for this job posting.",
            "Return only structured JSON.",
            f"Job ID: {job['job_id']}",
            f"Title: {job.get('job_title', '')}",
            f"Description: {job.get('description', '')}",
        ]
    )
    raw_response = client.generate_content(
        prompt=prompt,
        system_prompt=analyze_jobs.SYSTEM_PROMPT,
        response_format=analyze_jobs.JobKeywordResultList,
        reasoning_effort="low",
    )
    logging.info("Raw LLM response for job %s:\n%s", job_id, raw_response)
    logging.info("Parsed result: %s", analyze_jobs.parse_keyword_response(raw_response))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
