import pytest
import json
import logging
from unittest.mock import patch
from resume_parser import (
    INVALID_JSON,
    SCHEMA_VALIDATION_FAILED,
    TRUNCATED_OUTPUT,
    ResumeParseError,
    extract_first_json_object,
    parse_and_validate_resume,
)

def test_parse_and_validate_resume_success():
    mock_data = {"name": "John Doe", "experience": []}
    mock_json = json.dumps(mock_data)
    
    with patch('resume_parser.parse_resume_with_ai', return_value=mock_json):
        result = parse_and_validate_resume("some text")
        assert result["name"] == mock_data["name"]
        assert result["experience"] == mock_data["experience"]

def test_parse_and_validate_resume_retry_then_success():
    mock_data = {"name": "John Doe", "experience": []}
    mock_json = json.dumps(mock_data)
    
    with patch('resume_parser.parse_resume_with_ai') as mock_parse:
        # First call returns None, second returns valid JSON
        mock_parse.side_effect = [None, mock_json]
        
        # We need to patch time.sleep to avoid waiting during tests
        with patch('time.sleep'):
            result = parse_and_validate_resume("some text")
            assert result["name"] == mock_data["name"]
            assert result["experience"] == mock_data["experience"]
            assert mock_parse.call_count == 2

def test_parse_and_validate_resume_json_error_retry():
    mock_data = {"name": "John Doe", "experience": []}
    mock_json = json.dumps(mock_data)
    
    with patch('resume_parser.parse_resume_with_ai') as mock_parse:
        # First call returns malformed JSON, second returns valid JSON
        mock_parse.side_effect = ["{malformed}", mock_json]
        
        with patch('time.sleep'):
            result = parse_and_validate_resume("some text")
            assert result["name"] == mock_data["name"]
            assert result["experience"] == mock_data["experience"]
            assert mock_parse.call_count == 2


def test_parse_and_validate_resume_accepts_fenced_json_with_preamble():
    response = '''Here is the parsed resume:
```json
{"name": "John Doe", "summary": "Uses {structured} data"}
```
This is the requested output.'''

    with patch('resume_parser.parse_resume_with_ai', return_value=response):
        result = parse_and_validate_resume("some text")

    assert result["name"] == "John Doe"
    assert result["summary"] == "Uses {structured} data"


def test_extract_first_json_object_uses_first_valid_top_level_object():
    response = 'Preamble {not JSON}. Result: {"name": "First"} {"name": "Second"}'

    extracted = extract_first_json_object(response)

    assert json.loads(extracted) == {"name": "First"}


@pytest.mark.parametrize("response", [
    '{"name": "John Doe"',
    '{"name": "John Doe", "links": {"github": "example"}',
])
def test_extract_first_json_object_categorizes_truncated_output(response):
    with pytest.raises(ResumeParseError) as exc_info:
        extract_first_json_object(response)

    assert exc_info.value.category == TRUNCATED_OUTPUT


def test_extract_first_json_object_categorizes_invalid_json():
    with pytest.raises(ResumeParseError) as exc_info:
        extract_first_json_object('{"name": invalid}')

    assert exc_info.value.category == INVALID_JSON


def test_parse_and_validate_resume_retries_schema_validation_failure(caplog):
    invalid_schema = json.dumps({"name": "John Doe", "skills": "Python"})
    valid_data = {"name": "John Doe", "skills": ["Python"]}

    with patch('resume_parser.parse_resume_with_ai') as mock_parse:
        mock_parse.side_effect = [invalid_schema, json.dumps(valid_data)]
        with patch('time.sleep'), caplog.at_level(logging.WARNING):
            result = parse_and_validate_resume("some text")

    assert result["name"] == valid_data["name"]
    assert result["skills"] == valid_data["skills"]
    record = next(record for record in caplog.records if record.event == "resume_parse_failure")
    assert record.parse_category == SCHEMA_VALIDATION_FAILED
    assert record.parse_stage == "schema_validation"


def test_parse_failure_diagnostic_is_structured_and_payload_is_bounded(caplog):
    response = '{"name": invalid, "private": "' + ("secret-value-" * 30) + '"}'

    with patch('resume_parser.parse_resume_with_ai', return_value=response):
        with patch('time.sleep'), patch('sys.exit'), caplog.at_level(logging.WARNING):
            parse_and_validate_resume("some text", max_retries=1)

    record = next(record for record in caplog.records if record.event == "resume_parse_failure")
    assert record.parse_category == INVALID_JSON
    assert record.parse_stage == "json_decode"
    assert record.response_length == len(response)
    assert record.llm_model
    assert len(record.payload_snippet) <= 160
    assert response not in record.getMessage()
    assert "secret-value" not in record.payload_snippet

def test_parse_and_validate_resume_failure_exits():
    with patch('resume_parser.parse_resume_with_ai', return_value=None):
        with patch('time.sleep'):
            with patch('sys.exit') as mock_exit:
                parse_and_validate_resume("some text", max_retries=2)
                mock_exit.assert_called_once_with(1)

def test_parse_and_validate_resume_replaces_empty_with_na():
    mock_data = {"name": "", "summary": None, "skills": ["Python", ""]}
    expected_data = {"name": "NA", "summary": "NA", "skills": ["Python", "NA"]}
    mock_json = json.dumps(mock_data)
    
    with patch('resume_parser.parse_resume_with_ai', return_value=mock_json):
        result = parse_and_validate_resume("some text")
        assert {key: result[key] for key in expected_data} == expected_data


@pytest.mark.parametrize("payload", [{}, {"unexpected": "payload"}, {"name": "NA"}])
def test_parse_and_validate_resume_rejects_unusable_or_unknown_payload(payload):
    with patch('resume_parser.parse_resume_with_ai', return_value=json.dumps(payload)):
        with patch('time.sleep'):
            with pytest.raises(SystemExit):
                parse_and_validate_resume("some text", max_retries=1)


def test_main_fails_when_database_save_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("supabase_utils.download_resume_from_storage", lambda _name: b"pdf")
    monkeypatch.setattr("supabase_utils.save_base_resume", lambda _payload: False)
    monkeypatch.setattr("resume_parser.extract_text_from_pdf", lambda _path: "resume text")
    monkeypatch.setattr("resume_parser.parse_and_validate_resume", lambda _text: {"name": "Jane"})

    from resume_parser import main

    with pytest.raises(RuntimeError, match="Failed to save parsed resume"):
        main()
