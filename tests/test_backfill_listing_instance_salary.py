import backfill_listing_instance_salary


def test_build_payload_fills_only_missing_structured_salary_fields():
    payload, recovered = backfill_listing_instance_salary.build_payload({
        "listing_instances": [
            {"job_id": "1", "salary_text": "$80,000-$95,000 CAD"},
            {"job_id": "2", "salary_text": "$100,000-$120,000 USD", "salary_min": 99999},
            {"job_id": "3", "salary_text": None},
        ]
    })

    assert recovered == 2
    assert payload["listing_instances"][0]["salary_min"] == 80000
    assert payload["listing_instances"][0]["salary_max"] == 95000
    assert payload["listing_instances"][0]["salary_currency"] == "CAD"
    assert payload["listing_instances"][1]["salary_min"] == 99999
    assert payload["listing_instances"][1]["salary_currency"] == "USD"
    assert payload["listing_instances"][0]["salary_metadata_source"] == "salary_text_parser"


def test_build_payload_returns_none_when_nothing_is_recoverable():
    payload, recovered = backfill_listing_instance_salary.build_payload({
        "listing_instances": [{"job_id": "1", "salary_text": None}]
    })

    assert payload is None
    assert recovered == 0
