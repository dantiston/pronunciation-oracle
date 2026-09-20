from pronunciation_oracle.text_norm import normalize_word, tokenize


def test_normalize_strips_punctuation_and_casefolds():
    assert normalize_word("Pikachu!") == "pikachu"
    assert normalize_word("PIKACHU,") == "pikachu"
    assert normalize_word("pikachu") == "pikachu"


def test_normalize_keeps_internal_apostrophes_and_hyphens():
    assert normalize_word("don't") == "don't"
    assert normalize_word("Team-Rocket") == "team-rocket"


def test_normalize_strips_leading_trailing_hyphen_apostrophe():
    assert normalize_word("'quote'") == "quote"
    assert normalize_word("-dash-") == "dash"


def test_normalize_unicode_accents_casefold():
    assert normalize_word("Pokémon") == normalize_word("pokémon")


def test_tokenize_splits_on_whitespace_and_drops_empties():
    assert tokenize("  Hey   Pikachu!  Use Thunderbolt!  ") == ["Hey", "Pikachu!", "Use", "Thunderbolt!"]


def test_tokenize_empty_string():
    assert tokenize("") == []
    assert tokenize("   ") == []
