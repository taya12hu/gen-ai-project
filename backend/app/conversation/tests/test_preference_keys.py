"""
Tests for the preference key vocabulary.

Pure logic, no database. These cover the boundary that decides whether a
remembered fact can ever be acted on: query understanding picks the key
itself, and before this vocabulary existed a model answering "diet" instead
of "dietary" produced a preference that was stored, shown to the user in the
preferences panel, and then silently never applied to any search.
"""

from app.chat.service import VIBE_PREFERENCE_KEYS
from app.conversation.preferences import (
    PREFERENCE_KEYS,
    normalize_preference_key,
    normalize_preferences,
)


def test_canonical_keys_pass_through_unchanged():
    for key in PREFERENCE_KEYS:
        assert normalize_preference_key(key) == key


def test_known_aliases_map_to_their_canonical_key():
    assert normalize_preference_key("diet") == "dietary"
    assert normalize_preference_key("dietary_restriction") == "dietary"
    assert normalize_preference_key("ambiance") == "ambience"
    assert normalize_preference_key("atmosphere") == "ambience"


def test_key_matching_ignores_case_spacing_and_separators():
    assert normalize_preference_key("Dietary") == "dietary"
    assert normalize_preference_key("  DIETARY RESTRICTION ") == "dietary"
    assert normalize_preference_key("dietary-restriction") == "dietary"


def test_an_unknown_key_is_rejected_rather_than_stored():
    """The whole point: storing it would show the user a fact the system has
    already decided it will never use."""
    assert normalize_preference_key("budget") is None
    assert normalize_preference_key("favourite_colour") is None


def test_normalize_drops_unknown_keys_and_keeps_the_rest():
    result = normalize_preferences({"diet": "vegetarian", "budget": "1000", "vibe": "lively"})
    assert result == {"dietary": "vegetarian", "vibe": "lively"}


def test_values_are_trimmed_and_empty_ones_discarded():
    assert normalize_preferences({"dietary": "  vegetarian  "}) == {"dietary": "vegetarian"}
    assert normalize_preferences({"dietary": "   "}) == {}


def test_values_are_left_as_free_text():
    """Only the key is a closed vocabulary - values are whatever the user
    actually said, and can't be enumerated up front."""
    stated = "no beef, and I avoid shellfish"
    assert normalize_preferences({"dietary": stated}) == {"dietary": stated}


def test_retrieval_and_storage_agree_on_the_vocabulary():
    """These were two separately-maintained lists, and a key in one but not
    the other was exactly how a stored fact became unusable."""
    assert VIBE_PREFERENCE_KEYS == frozenset(PREFERENCE_KEYS)
