"""
Phase 4 - Preference Input Layer.

Defines the accepted shape of a user's restaurant preferences and validates
and normalizes it: enforces sane ranges for price/rating (when given - both
are optional), trims whitespace, and case-corrects place/cuisines against
values that actually exist in the data (Phase 3). This exists so a malformed
or unrecognized preference (typo'd place, made-up cuisine, negative price)
is rejected here rather than silently producing an empty result set later
in the retrieval phase (Phase 5).
"""

from pydantic import BaseModel, Field, field_validator


class UnknownValueError(ValueError):
    """Raised when a place/cuisine doesn't match anything in the dataset."""


class UserPreferences(BaseModel):
    place: str
    cuisines: list[str] = Field(min_length=1)
    max_price: float | None = Field(default=None, gt=0, description="Max budget for two people")
    min_rating: float | None = Field(
        default=None, ge=0, le=5, description="Minimum acceptable rating"
    )

    @field_validator("place")
    @classmethod
    def place_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("cuisines")
    @classmethod
    def cuisines_not_blank(cls, v: list[str]) -> list[str]:
        cleaned = [c.strip() for c in v]
        if any(not c for c in cleaned):
            raise ValueError("cuisines must not contain blank values")
        return cleaned


def normalize_preferences(
    place: str,
    cuisines: list[str],
    max_price: float | None,
    min_rating: float | None,
    known_places: set[str],
    known_cuisines: set[str],
) -> UserPreferences:
    """
    Validate types/ranges via UserPreferences, then case-insensitively match
    place/cuisines against known values, mapping to the canonical casing
    stored in the database. Raises UnknownValueError if no match is found.
    """
    prefs = UserPreferences(
        place=place, cuisines=cuisines, max_price=max_price, min_rating=min_rating
    )

    place_lookup = {p.lower(): p for p in known_places}
    cuisine_lookup = {c.lower(): c for c in known_cuisines}

    canonical_place = place_lookup.get(prefs.place.lower())
    if canonical_place is None:
        raise UnknownValueError(f"Unknown place: '{prefs.place}'")

    canonical_cuisines = []
    for cuisine in prefs.cuisines:
        canonical = cuisine_lookup.get(cuisine.lower())
        if canonical is None:
            raise UnknownValueError(f"Unknown cuisine: '{cuisine}'")
        canonical_cuisines.append(canonical)

    return prefs.model_copy(update={"place": canonical_place, "cuisines": canonical_cuisines})
