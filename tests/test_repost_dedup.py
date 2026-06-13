import supabase_utils


def test_normalize_title_handles_clear_cut_abbreviations():
    assert supabase_utils.normalize_title("Sr. Project Manager") == "senior project manager"
    assert supabase_utils.normalize_title("Technical   Project-Manager") == "technical project manager"


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


def test_description_fingerprint_ignores_minor_formatting_changes():
    a = "We are Chandos. Inclusion, collaboration, innovation."
    b = "We are Chandos\n\nInclusion, collaboration, innovation!"
    assert supabase_utils.make_description_fingerprint(a) == supabase_utils.make_description_fingerprint(b)
