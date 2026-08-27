import pdfplumber
import config
import json
import logging
import models
import re
import sys
import time
from llm_client import primary_client
from pydantic import ValidationError


logger = logging.getLogger(__name__)

TRUNCATED_OUTPUT = "truncated_output"
INVALID_JSON = "invalid_json"
SCHEMA_VALIDATION_FAILED = "schema_validation_failed"


class ResumeParseError(ValueError):
    def __init__(self, category, stage, detail):
        super().__init__(detail)
        self.category = category
        self.stage = stage
        self.detail = detail


def _find_object_end(text, start):
    stack = []
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            expected = "{" if character == "}" else "["
            if not stack or stack[-1] != expected:
                return index + 1
            stack.pop()
            if not stack:
                return index + 1

    return None


def extract_first_json_object(response_text):
    """Extract the first complete, valid top-level JSON object from LLM text."""
    if not isinstance(response_text, str) or not response_text.strip():
        raise ResumeParseError(INVALID_JSON, "extraction", "empty response")

    search_from = 0
    found_object_start = False
    last_decode_error = None

    while True:
        start = response_text.find("{", search_from)
        if start == -1:
            break

        following_text = response_text[start + 1:].lstrip()
        if following_text and not following_text.startswith(('"', '}')):
            search_from = start + 1
            continue

        found_object_start = True
        end = _find_object_end(response_text, start)
        if end is None:
            raise ResumeParseError(
                TRUNCATED_OUTPUT,
                "extraction",
                "JSON object was not closed before the response ended",
            )

        candidate = response_text[start:end]
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError as error:
            last_decode_error = error
            search_from = end
            continue

        if isinstance(decoded, dict):
            return candidate
        search_from = end

    if found_object_start and last_decode_error is not None:
        detail = f"JSON decoding failed at position {last_decode_error.pos}"
    else:
        detail = "response did not contain a JSON object"
    raise ResumeParseError(INVALID_JSON, "json_decode", detail)


def _sanitize_payload_snippet(payload, limit=160):
    if not isinstance(payload, str):
        return "<empty>"

    snippet = " ".join(payload.split())
    snippet = re.sub(r"[\w@.+/-]+", "<text>", snippet)
    return snippet[:limit]


def _log_parse_failure(error, payload, attempt, max_retries):
    response_length = len(payload) if isinstance(payload, str) else 0
    model_name = getattr(primary_client, "model", "unknown")
    snippet = _sanitize_payload_snippet(payload)
    logger.warning(
        "Resume parse failed: category=%s stage=%s attempt=%s/%s "
        "response_length=%s model=%s payload_snippet=%s",
        error.category,
        error.stage,
        attempt,
        max_retries,
        response_length,
        model_name,
        snippet,
        extra={
            "event": "resume_parse_failure",
            "parse_category": error.category,
            "parse_stage": error.stage,
            "attempt": attempt,
            "max_attempts": max_retries,
            "response_length": response_length,
            "llm_model": model_name,
            "payload_snippet": snippet,
        },
    )

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a given PDF file.

    Args:
        pdf_path (str): The file path to the PDF resume.

    Returns:
        str: The extracted text content from the PDF.
    """
    print(f"Extracting text from: {pdf_path}")
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract the visible text
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            
            # Extract embedded hyperlinks which are not captured by extract_text()
            if page.hyperlinks:
                for link in page.hyperlinks:
                    uri = link.get("uri")
                    if uri:
                        text += f"Embedded Link: {uri}\n"
    return text

def parse_resume_with_ai(resume_text):
    """
    Send resume text to an AI model and get structured information back.
    
    Args:
        resume_text (str): The plain text extracted from the resume
        
    Returns:
        str: JSON string of structured resume information
    """
    print("Processing resume with AI model...")

    prompt = f"""Extract and return the structured resume information from the text below. 
    Only use what is explicitly stated in the text and do not infer or invent any details.
    
    CRITICAL: If any information is missing or not available in the text, use "NA" for that field. 
    This applies to all fields (e.g., summary, dates, location, links, etc.). 
    Do NOT leave fields empty or use empty strings.

    Resume text:
    {resume_text}
    """

    response_text = primary_client.generate_content(
        prompt=prompt,
        response_format=models.Resume,
    )
    return response_text

def replace_empty_with_na(data):
    """
    Recursively replaces empty strings or None values in a dictionary or list with "NA".
    """
    if isinstance(data, dict):
        return {k: replace_empty_with_na(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_empty_with_na(i) for i in data]
    elif data == "" or data is None:
        return "NA"
    return data

def parse_and_validate_resume(resume_text, max_retries=config.MAX_RETRIES):
    """
    Attempts to parse resume text with AI, with retry logic for JSON errors or empty responses.
    
    Args:
        resume_text (str): The extracted text from the resume.
        max_retries (int): Maximum number of attempts.
        
    Returns:
        dict: The structured resume data with empty values replaced by "NA".
    """
    for attempt in range(max_retries):
        parsed_resume_details_str = parse_resume_with_ai(resume_text)

        try:
            json_object = extract_first_json_object(parsed_resume_details_str)
            resume_data_dict = replace_empty_with_na(json.loads(json_object))
            validated = models.Resume.model_validate(resume_data_dict)
            meaningful_fields = (
                validated.name,
                validated.email,
                validated.phone,
                validated.summary,
                validated.skills,
                validated.education,
                validated.experience,
                validated.projects,
                validated.certifications,
            )
            if not any(value and value != "NA" for value in meaningful_fields):
                raise ResumeParseError(
                    SCHEMA_VALIDATION_FAILED,
                    "usability_validation",
                    "Resume payload contains no usable identity, contact, or history data",
                )
            return replace_empty_with_na(validated.model_dump())
        except ValidationError as error:
            parse_error = ResumeParseError(
                SCHEMA_VALIDATION_FAILED,
                "schema_validation",
                f"Resume validation failed with {error.error_count()} error(s)",
            )
        except ResumeParseError as error:
            parse_error = error

        _log_parse_failure(
            parse_error,
            parsed_resume_details_str,
            attempt + 1,
            max_retries,
        )
        print(
            f"Attempt {attempt + 1}: Resume parse failed "
            f"({parse_error.category}). Retrying..."
        )
        if attempt < max_retries - 1:
            time.sleep(config.RETRY_DELAY_SECONDS)

    print(f"ERROR: Failed to parse resume after {max_retries} attempts.")
    sys.exit(1)

def main():
    """
    Main function to orchestrate the resume parsing process.
    Downloads the resume PDF from Supabase Storage, parses it with AI, 
    and saves the structured data to both local file and Supabase DB.
    """
    import io
    import os
    import supabase_utils

    pdf_file_path = "./resume.pdf"

    # 1. Try to download resume PDF from Supabase Storage
    pdf_bytes = supabase_utils.download_resume_from_storage("resume.pdf")

    if pdf_bytes:
        print("Successfully downloaded resume.pdf from Supabase Storage.")
        # Write to a temporary local file for pdfplumber
        with open(pdf_file_path, 'wb') as f:
            f.write(pdf_bytes)
    elif os.path.exists(pdf_file_path):
        print(f"Supabase Storage download failed. Using local file: {pdf_file_path}")
    else:
        print("ERROR: Could not find resume.pdf in Supabase Storage or locally.")
        print("Please upload your resume.pdf to the 'resumes' bucket in your Supabase Storage dashboard.")
        raise RuntimeError("Could not find resume.pdf in storage or locally")

    # 2. Extract text from PDF
    resume_text = extract_text_from_pdf(pdf_file_path)
    if not resume_text:
        print("Failed to extract text. Exiting.")
        raise RuntimeError("Resume PDF contains no extractable text")

    # 3. Parse resume text with AI
    resume_data_dict = parse_and_validate_resume(resume_text)

    # 4. Save parsed data to Supabase base_resume table
    save_success = supabase_utils.save_base_resume(resume_data_dict)
    if save_success:
        print("Successfully saved parsed resume to Supabase database.")
    else:
        raise RuntimeError("Failed to save parsed resume to Supabase database")

    # 5. Also save to local JSON file (for development/fallback)
    output_path = config.BASE_RESUME_PATH
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(resume_data_dict, f, indent=4)
        print(f"Successfully saved parsed resume to local file: {output_path}")
    except Exception as e:
        print(f"Error saving resume to {output_path}: {e}")

    # 6. Clean up the temporary PDF file (don't leave sensitive data on disk in CI)
    if pdf_bytes and os.path.exists(pdf_file_path):
        try:
            os.remove(pdf_file_path)
            print(f"Cleaned up temporary file: {pdf_file_path}")
        except Exception as e:
            print(f"Warning: Could not clean up {pdf_file_path}: {e}")

    print("\nResume processing finished.")


if __name__ == "__main__":
    print("Starting resume processing...")
    main()
