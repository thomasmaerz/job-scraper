import supabase_utils


def test_normalize_title_handles_clear_cut_abbreviations():
    assert supabase_utils.normalize_title("Sr. Project Manager") == "senior project manager"
    assert supabase_utils.normalize_title("Technical   Project-Manager") == "technical project manager"


def test_normalize_title_does_not_corrupt_embedded_tokens():
    assert supabase_utils.normalize_title("SRE Manager") == "sre manager"


def test_normalize_location_collapses_formatting_noise():
    assert supabase_utils.normalize_location(" Toronto , Ontario  , Canada ") == "toronto ontario canada"


def test_build_canonical_key_uses_normalized_parts():
    key = supabase_utils.build_canonical_key(
        provider="linkedin",
        company="Chandos Construction",
        title="Sr. Project Manager",
        location="Chalk River, Ontario, Canada",
    )
    assert key == "linkedin|chandos construction|senior project manager|chalk river ontario canada"


def test_normalize_company_treats_separator_variants_consistently():
    assert supabase_utils.normalize_company("Foo-Bar") == "foo bar"
    assert supabase_utils.normalize_company("Foo/Bar") == "foo bar"
    assert supabase_utils.normalize_company("Foo Bar") == "foo bar"


def test_build_canonical_key_uses_normalized_company_separator_variants():
    dash_key = supabase_utils.build_canonical_key(
        provider="linkedin",
        company="Foo-Bar",
        title="Sr. Project Manager",
        location="Toronto / Ontario - Canada",
    )
    slash_key = supabase_utils.build_canonical_key(
        provider="linkedin",
        company="Foo/Bar",
        title="Senior Project Manager",
        location="Toronto, Ontario, Canada",
    )
    space_key = supabase_utils.build_canonical_key(
        provider="linkedin",
        company="Foo Bar",
        title="Sr Project Manager",
        location="Toronto Ontario Canada",
    )

    assert dash_key == slash_key == space_key == "linkedin|foo bar|senior project manager|toronto ontario canada"


def test_description_fingerprint_ignores_minor_formatting_changes():
    sentence = "We are Chandos. Inclusion, collaboration, innovation, and continuous improvement drive every project we deliver. "
    a = sentence * 6
    b = (sentence.replace(". ", "\n\n").replace(", ", ",  ").replace(" improvement", " improvement!") * 6)

    fingerprint_a = supabase_utils.make_description_fingerprint(a)
    fingerprint_b = supabase_utils.make_description_fingerprint(b)

    assert fingerprint_a is not None
    assert fingerprint_a == fingerprint_b


def test_description_fingerprint_normalizes_unicode_punctuation_for_long_equivalents():
    plain = (
        "We're building client-focused teams that solve complex problems with care, speed, and accountability. "
        "Our people partner across design, delivery, and operations to keep commitments clear and work moving forward. "
    ) * 4
    formatted = (
        "We’re building client—focused teams that solve complex problems with care, speed, and accountability.\n"
        "• Our people partner across design, delivery, and operations to keep commitments clear and work moving forward.\n\n"
    ) * 4

    fingerprint_plain = supabase_utils.make_description_fingerprint(plain)
    fingerprint_formatted = supabase_utils.make_description_fingerprint(formatted)

    assert fingerprint_plain is not None
    assert fingerprint_formatted is not None
    assert fingerprint_plain == fingerprint_formatted
