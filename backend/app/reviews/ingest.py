"""
Review Ingestion & Embedding (step 1: ingest).

Re-derives individual reviews from the raw reviews_list column that
app.data.cleaning drops (see data/CLEANING.md). To make sure every review
lands on the *same* restaurant that cleaning step kept, this script re-runs
its exact parsing/dropna/dedup steps (see app/data/cleaning.py) on the raw
CSV - including carrying reviews_list through untouched - so the surviving
(name, place) rows are identical to restaurants_clean.csv, row for row.
Each review is then matched to restaurants.id via that same (name, place)
key and inserted with a NULL embedding; embed.py fills the embeddings in a
second pass so this step (parsing ~50k raw rows) stays fast and independent
of the ML model load.

reviews_list format (verified against the full raw file): a stringified
Python list of (rating, review_text) tuples, e.g.
    "[('Rated 4.0', 'RATED\\n  A beautiful place to dine in...'), ...]"
rating is occasionally None (83 of ~1.32M raw rows) for a review with no
star rating attached - those reviews are still kept, with a NULL rating.
"""

import ast
import re

import pandas as pd
from psycopg2.extras import execute_values

from app.settings import DATA_DIR
from app.storage.db import get_connection

RAW_CSV = DATA_DIR / "raw" / "train.csv"

# Same critical columns app.data.cleaning requires before a row counts as a
# kept restaurant - must match that module so the dedup output lines up.
CRITICAL_COLUMNS = ["name", "place", "cuisines", "price", "rating"]

RATE_PATTERN = re.compile(r"^\s*(\d(?:\.\d)?)\s*/\s*5\s*$")
REVIEW_RATING_PATTERN = re.compile(r"^Rated\s+(\d(?:\.\d)?)$")
RATED_PREFIX_PATTERN = re.compile(r"^RATED\s*\n?\s*", re.IGNORECASE)

BATCH_SIZE = 1000

# Raw review counts per restaurant are heavily skewed (median 4, but up to
# 1,765 for one outlier) - storing every review for the high-volume tail
# blew well past Supabase's free-tier database size limit for very little
# grounding benefit. Capping keeps the longest, most substantive reviews per
# restaurant (short ones like "good food" carry little signal for chat
# grounding anyway) while leaving the vast majority of restaurants - which
# already have fewer than this many reviews - completely untouched.
REVIEWS_PER_RESTAURANT_CAP = 15


def parse_rating(value) -> float | None:
    if pd.isna(value):
        return None
    match = RATE_PATTERN.match(str(value))
    return float(match.group(1)) if match else None


def parse_price(value) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def parse_cuisines(value) -> str | None:
    if pd.isna(value):
        return None
    parts = [c.strip() for c in str(value).split(",") if c.strip()]
    return ", ".join(parts) if parts else None


def clean_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def parse_reviews(raw_value: str) -> list[tuple[float | None, str]]:
    """One reviews_list cell -> [(review_rating, review_text), ...], skipping blanks.

    The raw Zomato export repeats every review in a restaurant's cell several
    times over (verified: some restaurants have each review tripled) - almost
    certainly an artifact of how the source scrape was assembled, not real
    signal. Deduplicated by cleaned text here, at the point where duplicates
    are first visible, so nothing downstream (ranking, capping, snippet
    selection) has to special-case it.
    """
    if pd.isna(raw_value):
        return []
    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return []

    reviews = []
    seen_text: set[str] = set()
    for rating_str, text in parsed:
        clean = RATED_PREFIX_PATTERN.sub("", str(text)).strip()
        if not clean or clean in seen_text:
            continue
        seen_text.add(clean)
        rating_match = REVIEW_RATING_PATTERN.match(str(rating_str).strip()) if rating_str else None
        rating = float(rating_match.group(1)) if rating_match else None
        reviews.append((rating, clean))
    return reviews


def load_matched_rows() -> pd.DataFrame:
    """Reproduce app.data.cleaning's cleaned+deduped restaurant rows, with reviews_list attached."""
    df = pd.read_csv(RAW_CSV)

    cleaned = pd.DataFrame(
        {
            "name": df["name"].apply(clean_text),
            "place": df["location"].apply(clean_text),
            "cuisines": df["cuisines"].apply(parse_cuisines),
            "price": df["approx_cost(for two people)"].apply(parse_price),
            "rating": df["rate"].apply(parse_rating),
            "reviews_list": df["reviews_list"],
        }
    )

    cleaned = cleaned.dropna(subset=CRITICAL_COLUMNS)
    cleaned = cleaned.drop_duplicates(subset=["name", "place"], keep="first")
    return cleaned.reset_index(drop=True)


def main() -> None:
    matched = load_matched_rows()
    print(f"Matched {len(matched)} restaurants (should equal restaurants_clean.csv row count).")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("select id, name, place from restaurants;")
            restaurant_ids = {(name, place): rid for rid, name, place in cur.fetchall()}

        rows_to_insert: list[tuple[int, float | None, str]] = []
        unmatched = 0
        for _, row in matched.iterrows():
            restaurant_id = restaurant_ids.get((row["name"], row["place"]))
            if restaurant_id is None:
                unmatched += 1
                continue

            reviews = parse_reviews(row["reviews_list"])
            # Longest (most substantive) first, ties broken by rating - same
            # ranking used for the one-off cleanup of the initial over-quota
            # ingest, kept here so a future full rebuild doesn't regress.
            reviews.sort(key=lambda r: (len(r[1]), r[0] if r[0] is not None else -1), reverse=True)
            for review_rating, review_text in reviews[:REVIEWS_PER_RESTAURANT_CAP]:
                rows_to_insert.append((restaurant_id, review_rating, review_text))

        if unmatched:
            print(f"Warning: {unmatched} rows had no matching restaurants row (unexpected).")

        with conn.cursor() as cur:
            cur.execute("truncate table restaurant_reviews restart identity;")
            for i in range(0, len(rows_to_insert), BATCH_SIZE):
                batch = rows_to_insert[i : i + BATCH_SIZE]
                execute_values(
                    cur,
                    "insert into restaurant_reviews (restaurant_id, review_rating, review_text) values %s",
                    batch,
                )
        conn.commit()
        print(f"Inserted {len(rows_to_insert)} reviews across {len(matched) - unmatched} restaurants.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
