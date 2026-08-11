"""
Data Cleaning & Normalization.

Reads the raw dataset produced by acquisition.py (backend/data/raw/train.csv),
normalizes the fields the pipeline depends on (price, place, rating, cuisine),
deduplicates, and writes a clean dataset to backend/data/processed/.

Deduplication note: the raw data repeats each restaurant once per Zomato
listing category (listed_in(type): Buffet/Delivery/Dine-out/...) and per
listed_in(city) grouping, with the same name/location/cuisines/rate/votes
each time. We dedupe on (name, location), which is the right granularity
for recommending a restaurant.
"""

import re

import pandas as pd

from app.settings import DATA_DIR

RAW_CSV = DATA_DIR / "raw" / "train.csv"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_CSV = PROCESSED_DIR / "restaurants_clean.csv"

RATE_PATTERN = re.compile(r"^\s*(\d(?:\.\d)?)\s*/\s*5\s*$")

# Fields required for the pipeline's core filtering (price, place, rating, cuisine)
# plus 'name' as the identifier. Rows missing any of these can't be matched or shown.
CRITICAL_COLUMNS = ["name", "place", "cuisines", "price", "rating"]


def parse_rating(value) -> float | None:
    """'4.1/5' / '3.9 /5' -> 4.1. 'NEW', '-', NaN -> None."""
    if pd.isna(value):
        return None
    match = RATE_PATTERN.match(str(value))
    if not match:
        return None
    return float(match.group(1))


def parse_price(value) -> float | None:
    """'1,200' -> 1200.0. NaN -> None."""
    if pd.isna(value):
        return None
    cleaned = str(value).replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_cuisines(value) -> str | None:
    """'Chinese, North Indian ,Thai' -> 'Chinese, North Indian, Thai'."""
    if pd.isna(value):
        return None
    parts = [c.strip() for c in str(value).split(",") if c.strip()]
    return ", ".join(parts) if parts else None


def clean_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def main() -> None:
    df = pd.read_csv(RAW_CSV)
    rows_raw = len(df)

    cleaned = pd.DataFrame(
        {
            "name": df["name"].apply(clean_text),
            "place": df["location"].apply(clean_text),
            "city": df["listed_in(city)"].apply(clean_text),
            "cuisines": df["cuisines"].apply(parse_cuisines),
            "price": df["approx_cost(for two people)"].apply(parse_price),
            "rating": df["rate"].apply(parse_rating),
            "rest_type": df["rest_type"].apply(clean_text),
            "votes": df["votes"],
        }
    )

    rows_before_dropna = len(cleaned)
    cleaned = cleaned.dropna(subset=CRITICAL_COLUMNS)
    rows_after_dropna = len(cleaned)

    cleaned = cleaned.drop_duplicates(subset=["name", "place"], keep="first")
    rows_after_dedup = len(cleaned)

    cleaned = cleaned.reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_CSV, index=False)

    print(f"Raw rows:                          {rows_raw}")
    print(f"After parsing (pre dropna):         {rows_before_dropna}")
    print(f"After dropping missing critical fields: {rows_after_dropna}")
    print(f"After dedup on (name, place):        {rows_after_dedup}")
    print(f"Saved cleaned dataset -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
