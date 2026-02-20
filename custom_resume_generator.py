import logging
import io
import supabase_utils
import config
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any
import json
import pdf_generator
import re
import asyncio
from google import genai
from google.genai import types
from models import (
    Education, Experience, Project, Certification, Links, Resume,
    SummaryOutput, SkillsOutput, ExperienceListOutput, SingleExperienceOutput,
    ProjectListOutput, SingleProjectOutput, ValidationResponse
)
import time

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Gemini Client ---
client = genai.Client(api_key=config.GEMINI_SECOND_API_KEY)

# --- Content budget constants (controls 2-page fit) ---
MAX_SUMMARY_WORDS = 60          # ~3 lines at normal font
MAX_SKILLS = 12                 # 4 rows of 3 columns
MAX_EXPERIENCE_ITEMS = 5        # Limit number of roles shown
MAX_BULLETS_PER_ROLE = 4        # Max bullet points per experience/project
MAX_BULLET_WORDS = 20           # Hard cap: ensures single line at 10pt font
MAX_PROJECTS = 2                # Projects section is often cut first


def trim_to_single_line_bullet(text: str) -> str:
    """Truncates a bullet string to MAX_BULLET_WORDS words."""
    words = text.strip().split()
    if len(words) <= MAX_BULLET_WORDS:
        return text.strip()
    return " ".join(words[:MAX_BULLET_WORDS]).rstrip(",;") + "."


def enforce_content_budget(resume: Resume) -> Resume:
    """
    Trims resume content to fit within a 2-page budget BEFORE PDF generation.
    Modifies a deep copy and returns it.
    """
    r = resume.model_copy(deep=True)

    # Trim summary
    if r.summary:
        words = r.summary.split()
        if len(words) > MAX_SUMMARY_WORDS:
            r.summary = " ".join(words[:MAX_SUMMARY_WORDS]).rstrip(",;") + "..."

    # Trim skills
    if r.skills and len(r.skills) > MAX_SKILLS:
        r.skills = r.skills[:MAX_SKILLS]

    # Trim experience: limit roles and bullets per role
    if r.experience:
        r.experience = r.experience[:MAX_EXPERIENCE_ITEMS]
        for exp in r.experience:
            if exp.description:
                bullets = _split_to_bullets(exp.description)
                bullets = bullets[:MAX_BULLETS_PER_ROLE]
                bullets = [trim_to_single_line_bullet(b) for b in bullets]
                exp.description = "\n".join(f"• {b}" for b in bullets)

    # Trim projects
    if r.projects:
        r.projects = r.projects[:MAX_PROJECTS]
        for proj in r.projects:
            if proj.description:
                bullets = _split_to_bullets(proj.description)
                bullets = bullets[:MAX_BULLETS_PER_ROLE]
                bullets = [trim_to_single_line_bullet(b) for b in bullets]
                proj.description = "\n".join(f"• {b}" for b in bullets)

    return r


def _split_to_bullets(description: str) -> list:
    """Splits a description into individual bullet strings (without prefix)."""
    if '\n' in description:
        bullets = []
        for line in description.split('\n'):
            line = line.strip().lstrip('•-').strip()
            if line:
                bullets.append(line)
        return bullets
    # Sentence split fallback
    abbrevs = {
        "e.g.": "EG", "i.e.": "IE", "etc.": "ETC", "vs.": "VS",
        "Mr.": "MR", "Mrs.": "MRS", "Ms.": "MS", "Dr.": "DR",
        "St.": "ST", "Ph.D.": "PHD", "U.S.": "US", "U.K.": "UK",
    }
    text = description.strip()
    for k, v in abbrevs.items():
        text = text.replace(k, f"TEMP_{v}")
    sentences = [s.strip() for s in text.split('. ') if s.strip()]
    result = []
    for s in sentences:
        for k, v in abbrevs.items():
            s = s.replace(f"TEMP_{v.replace('.','')}", k)
        if s and not s[-1] in '.!?':
            s += '.'
        result.append(s)
    return result


# --- LLM Personalization Function ---
def extract_json_from_text(text: str) -> str:
    fenced_match = re.search(r"```(?:json)?\s*(\[\s*{.*?}\s*\]|\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        json_candidate = fenced_match.group(1).strip()
    else:
        loose_match = re.search(r"(\[\s*{.*?}\s*\]|\[.*?\]|\{.*?\})", text, re.DOTALL)
        if loose_match:
            json_candidate = loose_match.group(1).strip()
        else:
            json_candidate = text.strip()
    try:
        parsed = json.loads(json_candidate)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to extract valid JSON: {e}\nRaw candidate:\n{json_candidate}")


async def personalize_section_with_llm(
    section_name: str,
    section_content: Any,
    full_resume: Resume,
    job_details: Dict[str, Any]
) -> Any:
    """
    Uses Gemini to personalize a specific section of the resume for the given job.
    """
    if not section_content:
        logging.warning(f"Skipping personalization for empty section: {section_name}")
        return section_content

    output_model_map = {
        "summary": (SummaryOutput, "summary"),
        "skills": (SkillsOutput, "skills"),
        "experience": (SingleExperienceOutput, "experience"),
        "projects": (SingleProjectOutput, "project"),
    }

    if section_name not in output_model_map:
        logging.error(f"Unsupported section_name for LLM personalization: {section_name}")
        return section_content

    OutputModel, output_key = output_model_map[section_name]

    resume_context_dict = full_resume.model_dump(exclude={section_name})
    resume_context = json.dumps(resume_context_dict, indent=2)

    if isinstance(section_content, list) and section_content and hasattr(section_content[0], 'model_dump'):
        serializable_section_content = [item.model_dump() for item in section_content]
    else:
        serializable_section_content = section_content

    prompts = []

    prompt_intro = f"""
    **Task:** Enhance the specified resume section for the target job application.

    **Target Job**
    - Title: {job_details['job_title']}
    - Company: {job_details['company']}
    - Seniority Level: {job_details['level']}
    - Job Description: {job_details['description']}

    ---

    **Full Resume Context (excluding the section being edited):**
    {resume_context}

    **Resume Section to Enhance:** {section_name}
    """

    system_prompt = f"""
    You are an expert resume writer and a precise JSON generation assistant.
    Your primary function is to enhance specified sections of a resume to better align with a target job description, based on the provided resume context and original section content.

    **CRITICAL OUTPUT REQUIREMENTS:**
    1.  You MUST ALWAYS output a single, valid JSON object.
    2.  Your entire response MUST be *only* the JSON object.
    3.  Do NOT include any introductory text, explanations, apologies, markdown formatting (like ```json or ```), or any text outside of the JSON structure itself.

    **CORE RESUME WRITING PRINCIPLES:**
    1.  **Adhere to Instructions:** Meticulously follow all specific instructions provided in the user prompt for the given section.
    2.  **No Fabrication:** NEVER invent new information, skills, projects, job titles, or responsibilities not explicitly found in the original resume materials. Rephrasing and emphasizing existing facts is allowed; fabrication is strictly forbidden.
    3.  **Relevance:** Focus on aligning the candidate's existing experience and skills with the target job.
    4.  **Fact-Based:** All enhancements must be grounded in the provided "Full Resume Context" or "Original Content of This Section."

    **BULLET POINT LENGTH RULE (CRITICAL):**
    - Every bullet point in experience and project descriptions MUST be a single line only.
    - A single line means NO MORE THAN {MAX_BULLET_WORDS} WORDS per bullet point.
    - Do NOT write multi-sentence bullets. One concise thought per bullet only.
    - If you cannot express a point in {MAX_BULLET_WORDS} words, cut it — do not exceed the limit.
    - Bad example (too long): "Led cross-functional team to migrate legacy on-premises ERP system to SAP S/4HANA cloud, reducing infrastructure costs by 30% and improving system reliability."
    - Good example: "Led ERP migration to SAP S/4HANA cloud, cutting infrastructure costs 30%."
    """

    specific_instructions = ""

    if section_name == "summary":
        specific_instructions = f"""
        **Original Content of This Section:**
        {json.dumps(serializable_section_content, indent=2)}

        ---
        **Instructions:**
        - Rewrite **only** the summary to be concise, impactful, and highly relevant to the Target Job.
        - **CRITICAL LENGTH LIMIT: The summary MUST be {MAX_SUMMARY_WORDS} words or fewer.** Be ruthless about brevity.
        - **CRITICAL: The core professional identity and experience level from the "Original Content of This Section" MUST be preserved.**
        - Highlight 2-3 key qualifications that ALIGN with the "Job Description," based only on facts in the resume.
        - Use strong action verbs and keywords from the "Job Description" where appropriate.
        - **ABSOLUTELY DO NOT INVENT new information, skills, projects, job titles, or responsibilities.**
        ---
        **Expected JSON Output Structure:** {{"summary": "A dynamic and results-oriented professional with X years of experience..."}}
        """
        prompts.append(prompt_intro + specific_instructions)

    elif section_name == "experience":
        for exp_item_content in serializable_section_content:
            specific_instructions = f"""
            **Original Content of This Specific Experience Item:**
            {json.dumps(exp_item_content, indent=2)}

            ---
            **Instructions for this experience item:**
            - Enhance the 'description' field ONLY. All other fields MUST remain UNCHANGED.
            - Write EXACTLY {MAX_BULLETS_PER_ROLE} bullet points or fewer for this role — no more.
            - **CRITICAL: Each bullet point MUST be {MAX_BULLET_WORDS} words or fewer. Single line only. No exceptions.**
            - Start each bullet with a strong past-tense action verb.
            - Integrate relevant keywords from the Target Job Description naturally.
            - Quantify achievements where possible based on original content only.
            - Do NOT invent skills or experiences.
            - Format the description as newline-separated bullets, e.g.: "Delivered X\\nManaged Y\\nImproved Z"
            ---
            **Expected JSON Output Structure:** {{"experience": {{"job_title": "Original Job Title", "company": "Original Company", "dates": "Original Dates", "description": "Bullet 1\\nBullet 2\\nBullet 3", "location": "Original Location"}}}}
            """
            prompts.append(prompt_intro + specific_instructions)

    elif section_name == "projects":
        for project_item_content in serializable_section_content:
            specific_instructions = f"""
            **Original Content of This Specific Project Item:**
            {json.dumps(project_item_content, indent=2)}

            ---
            **Instructions for this project item:**
            - Enhance the 'description' field ONLY. All other fields MUST remain UNCHANGED.
            - Write EXACTLY {MAX_BULLETS_PER_ROLE} bullet points or fewer — no more.
            - **CRITICAL: Each bullet point MUST be {MAX_BULLET_WORDS} words or fewer. Single line only. No exceptions.**
            - Do NOT invent skills or experiences.
            - Format the description as newline-separated bullets, e.g.: "Built X\\nDeployed Y"
            ---
            **Expected JSON Output Structure (for this single project item):** {{"project": {{"name": "Original Project Name", "technologies": ["Tech1"], "description": "Bullet 1\\nBullet 2", "link": "Original Link"}}}}
            """
            prompts.append(prompt_intro + specific_instructions)

    elif section_name == "skills":
        specific_instructions = f"""
        **Original Content of This Section (Candidate's Initial Skills List):**
        {json.dumps(serializable_section_content, indent=2)}

        ---
        **Instructions for Generating the Curated Skills List:**

        **1. Identify Candidate's Actual Skills:**
        - Review the 'Full Resume Context' and 'Original Content of This Section'.
        - Only include skills *explicitly written* in the resume materials.
        - **CRITICAL RULE: DO NOT infer, assume, or invent any skills.**

        **2. Select and Refine for the Target Job:**
        - Select only the most relevant skills to the Target Job Description.
        - **CRITICAL LENGTH LIMIT: Output between 5 and {MAX_SKILLS} skills maximum.**
        - Prioritize skills directly mentioned in the Job Description AND confirmed in the resume.
        - Avoid redundancy.
        ---
        **Expected JSON Output Structure:** {{"skills": ["Python", "Azure", "Docker", "Agile", "SQL"]}}
        """
        prompts.append(prompt_intro + specific_instructions)

    logging.info(f"Number of prompts: {len(prompts)}")

    responses = []
    for prompt in prompts:
        logging.info(f"Sending prompt to Gemini for section: {section_name}")
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    system_instruction=system_prompt,
                    response_mime_type='application/json',
                    response_schema=OutputModel,
                )
            )
            llm_output = response.text.strip()
            logging.info(f"Received response from Gemini for section: {section_name}")
            try:
                parsed_response_model = OutputModel.model_validate_json(llm_output)
                responses.append(parsed_response_model)
            except ValidationError as e:
                logging.error(f"Failed to validate LLM JSON output for {section_name}: {e}")
                logging.error(f"LLM Raw Output: {llm_output}")
                return section_content
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM JSON output for {section_name}: {e}")
                return section_content
        except Exception as e:
            logging.error(f"Error calling Gemini for section {section_name}: {e}")
            return section_content

    logging.info(f"Received {len(responses)} responses from Gemini for section: {section_name}")

    if section_name in ("summary", "skills"):
        return getattr(responses[0], output_key)
    elif section_name == "experience":
        return [getattr(r, output_key) for r in responses]
    elif section_name == "projects":
        return [getattr(r, output_key) for r in responses]


async def validate_customization(
    section_name: str,
    original_content: Any,
    customized_content: Any,
    full_original_resume: Resume,
    job_details: Dict[str, Any]
) -> (bool, str):

    resume_context_dict = full_original_resume.model_dump(exclude={section_name})
    resume_context = json.dumps(resume_context_dict, indent=2)

    if isinstance(original_content, list) and original_content and hasattr(original_content[0], 'model_dump'):
        serializable_original_content = [item.model_dump() for item in original_content]
    else:
        serializable_original_content = original_content

    if isinstance(customized_content, list) and customized_content and hasattr(customized_content[0], 'model_dump'):
        serializable_customized_content = [item.model_dump() for item in customized_content]
    else:
        serializable_customized_content = customized_content

    system_prompt = f"""
    You are a meticulous Resume Fact-Checker.
    Your primary function is to compare an "Original Resume Section" with a "Customized Resume Section" and determine if the customized version introduces any information, skills, experiences, or qualifications that are NOT supported by or cannot be reasonably inferred from the original section or the broader original resume context.

    **CRITICAL OUTPUT REQUIREMENTS:**
    1.  You MUST ALWAYS output a single, valid JSON object.
    2.  Your entire response MUST be *only* the JSON object.
    3.  Do NOT include any introductory text, explanations, apologies, markdown formatting, or any text outside of the JSON structure itself.
    4.  The JSON object MUST contain exactly two keys:
        - "is_valid": A boolean (true if faithful and accurate; false otherwise).
        - "reason": A string explaining your decision.
    """

    user_prompt = f"""
    **Task:** Evaluate the "Customized Resume Section" against the "Original Resume Section" and "Original Full Resume Context."

    **Target Job Details:**
    - Title: {job_details['job_title']}
    - Company: {job_details['company']}
    - Seniority Level: {job_details['level']}
    - Job Description: {job_details['description']}

    ---
    **Original Full Resume Context (excluding this section):**
    {resume_context}

    ---
    **Original Resume Section ("{section_name}"):**
    {json.dumps(serializable_original_content, indent=2)}

    ---
    **Customized Resume Section ("{section_name}"):**
    {json.dumps(serializable_customized_content, indent=2)}

    ---
    **Evaluation Criteria:**
    1.  **Factual Accuracy:** Does the customized section add or alter core facts not supported by the original? Has the primary job title or professional identity been fundamentally changed without explicit support in the original resume? Rephrasing IS acceptable. Fabrication IS NOT.
    2.  **Skill Consistency:** Are any new skills mentioned verifiably present in the original resume context?
    3.  **Preservation of Core Meaning:** Does the customized section fundamentally change the nature or primary professional identity of the original?

    Based on all provided information, output your JSON response.
    """

    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=system_prompt,
                response_mime_type='application/json',
                response_schema=ValidationResponse,
            )
        )
        llm_output = response.text.strip()
        try:
            parsed = ValidationResponse.model_validate_json(llm_output)
            logging.info(f"Validation response: {parsed}")
            return parsed.is_valid, parsed.reason
        except ValidationError as e:
            logging.error(f"Failed to validate validation schema: {e}")
            return False, "Failed to validate LLM JSON output against validation schema."
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse validation JSON: {e}")
            return False, "Failed to parse LLM JSON output."
    except Exception as e:
        logging.error(f"Error calling Gemini for validation: {e}")
        return False, "Error calling Gemini or processing response."


# --- Main Processing Logic ---
async def process_job(job_details: Dict[str, Any], base_resume_details: Resume):
    """
    Processes a single job: personalizes resume, enforces content budget, generates PDF, uploads, updates status.
    """
    job_id = job_details.get("job_id")
    if not job_id:
        logging.error("Job details missing job_id.")
        return

    logging.info(f"--- Starting processing for job_id: {job_id} ---")

    try:
        personalized_resume_data = base_resume_details.model_copy(deep=True)
        any_validation_failed = False

        sections_to_personalize = {
            "summary": base_resume_details.summary,
            "experience": base_resume_details.experience,
            "projects": base_resume_details.projects,
            "skills": base_resume_details.skills,
        }

        sleep_time = 6

        for section_name, section_content in sections_to_personalize.items():
            if any_validation_failed:
                logging.warning(f"Skipping further personalization for job_id {job_id} due to prior validation failure.")
                break

            if section_content:
                logging.info(f"Waiting {sleep_time}s before next request...")
                time.sleep(sleep_time)

                logging.info(f"Personalizing section: {section_name} for job_id: {job_id}")
                personalized_content = await personalize_section_with_llm(
                    section_name, section_content, base_resume_details, job_details
                )

                logging.info(f"Waiting {sleep_time}s before validation...")
                time.sleep(sleep_time)

                logging.info(f"Validating section: {section_name} for job_id: {job_id}")
                is_valid, reason = await validate_customization(
                    section_name, section_content, personalized_content,
                    base_resume_details, job_details
                )

                if is_valid:
                    logging.info(f"Section {section_name} validation passed.")
                    setattr(personalized_resume_data, section_name, personalized_content)
                    sections_to_personalize[section_name] = personalized_content
                else:
                    logging.warning(f"VALIDATION FAILED for section {section_name}, job_id {job_id}. Reason: {reason}")
                    logging.warning(f"Halting resume generation for job_id {job_id}.")
                    any_validation_failed = True
                    break

                logging.info(f"Finished section: {section_name} for job_id: {job_id}")
            else:
                logging.info(f"Skipping empty section: {section_name} for job_id: {job_id}")

        if any_validation_failed:
            logging.info(f"--- Aborting PDF generation for job_id: {job_id} due to validation failure. ---")
            return

        # Enforce 2-page content budget BEFORE generating PDF
        logging.info(f"Enforcing content budget for job_id: {job_id}")
        budget_trimmed_resume = enforce_content_budget(personalized_resume_data)

        # Generate PDF
        logging.info(f"Generating PDF for job_id: {job_id}")
        try:
            pdf_bytes = pdf_generator.create_resume_pdf(budget_trimmed_resume)
            if not pdf_bytes:
                raise ValueError("PDF generation returned empty bytes.")
            logging.info(f"PDF generation complete for job_id: {job_id}")
        except Exception as e:
            logging.error(f"Failed to generate PDF for job_id {job_id}: {e}")
            return

        # Upload PDF to Supabase Storage
        destination_path = f"personalized_resumes/resume_{job_id}.pdf"
        logging.info(f"Uploading PDF to {destination_path} for job_id: {job_id}")
        resume_link = supabase_utils.upload_customized_resume_to_storage(pdf_bytes, destination_path)

        if not resume_link:
            logging.error(f"Failed to upload resume PDF for job_id: {job_id}")
            return

        logging.info(f"Successfully uploaded PDF for job_id: {job_id}. Link: {resume_link}")

        # Save to Supabase
        logging.info("Adding customized resume to Supabase")
        customized_resume_id = supabase_utils.save_customized_resume(budget_trimmed_resume, resume_link)

        logging.info(f"Updating job record for job_id: {job_id}")
        update_success = supabase_utils.update_job_with_resume_link(job_id, customized_resume_id, new_status="resume_generated")

        if update_success:
            logging.info(f"Successfully updated job record for job_id: {job_id}")
        else:
            logging.error(f"Failed to update job record for job_id: {job_id}")

        logging.info(f"--- Finished processing for job_id: {job_id} ---")

    except Exception as e:
        logging.error(f"Unexpected error processing job_id {job_id}: {e}", exc_info=True)


async def run_job_processing_cycle():
    """Fetches top jobs and processes them one by one."""
    logging.info("Starting new job processing cycle...")

    user_email = config.LINKEDIN_EMAIL
    if not user_email:
        logging.error("LINKEDIN_EMAIL not set in config. Cannot fetch base resume.")
        return

    logging.info(f"Fetching base resume for user: {user_email}")
    raw_resume_details = supabase_utils.get_resume_custom_fields_by_email(user_email)

    if not raw_resume_details:
        logging.error(f"Could not find base resume for user: {user_email}. Aborting cycle.")
        return

    try:
        for key in ['skills', 'experience', 'education', 'projects', 'certifications', 'languages']:
            if raw_resume_details.get(key) is None:
                raw_resume_details[key] = []
        base_resume_details = Resume(**raw_resume_details)
        logging.info("Successfully parsed base resume.")
    except Exception as e:
        logging.error(f"Error parsing base resume: {e}")
        logging.error(f"Raw data: {raw_resume_details}")
        return

    jobs_limit = 2
    logging.info(f"Fetching top {jobs_limit} scored jobs...")
    jobs_to_process = supabase_utils.get_top_scored_jobs_for_resume_generation(limit=jobs_limit)

    if not jobs_to_process:
        logging.info("No new jobs found to process in this cycle.")
        return

    logging.info(f"Found {len(jobs_to_process)} jobs to process.")

    for job_details in jobs_to_process:
        await process_job(job_details, base_resume_details)

    logging.info("Finished job processing cycle.")


# --- Script Entry Point ---
if __name__ == "__main__":
    logging.info("Script started.")
    try:
        asyncio.run(run_job_processing_cycle())
        logging.info("Resume processing completed successfully.")
    except Exception as e:
        logging.error(f"Error during task execution: {e}", exc_info=True)
