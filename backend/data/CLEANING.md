# Phase 2 — Data Cleaning & Normalization

**Input:** `phase1_data_acquisition/data/raw/train.csv` (51,717 raw rows)
**Output:** `data/processed/restaurants_clean.csv`
**Script:** `clean_data.py`

## Transformations

| Output column | From | Transformation |
|---|---|---|
| `name` | `name` | Trimmed |
| `place` | `location` | Trimmed |
| `city` | `listed_in(city)` | Trimmed |
| `cuisines` | `cuisines` | Split on comma, trimmed each, rejoined (`"Chinese, North Indian ,Thai"` → `"Chinese, North Indian, Thai"`) |
| `price` | `approx_cost(for two people)` | Comma stripped, parsed to float (`"1,200"` → `1200.0`) |
| `rating` | `rate` | Parsed `"X.X/5"` / `"X.X /5"` → float. `"NEW"`, `"-"`, blank → `None` |
| `rest_type` | `rest_type` | Trimmed (kept as supporting context, not a filter field) |
| `votes` | `votes` | Unchanged |

Dropped columns (not needed for filtering or recommendation): `url`, `address`, `online_order`, `book_table`, `phone`, `dish_liked`, `reviews_list`, `menu_item`, `listed_in(type)`.

## Key data-quality findings

- **`rate` formatting was inconsistent**: some rows used `"4.1/5"`, others `"3.9 /5"` (extra space) — the parser strips whitespace before matching, so both normalize correctly.
- **`rate` missing/placeholder values**: `NaN` (7,775 rows), `"NEW"` (2,208 rows — not yet rated), `"-"` (69 rows) — all normalized to `None` and dropped as a critical field.
- **Heavy duplication**: the raw data repeats each restaurant once per Zomato `listed_in(type)` category (Buffet/Cafes/Delivery/Desserts/Dine-out/Drinks & nightlife/Pubs and bars) crossed with `listed_in(city)` grouping. Same restaurant → same `name`/`location`/`cuisines`/`rate`/`votes` across those repeated rows. Deduplicated on `(name, place)`, keeping the first occurrence.

## Row counts

| Stage | Rows |
|---|---|
| Raw | 51,717 |
| After dropping rows missing a critical field (`name`, `place`, `cuisines`, `price`, `rating`) | 41,410 |
| After deduplication on `(name, place)` | **9,216** |

## Output schema sanity

- `price` range: ₹40 – ₹6,000
- `rating` range: 1.8 – 4.9
- No nulls in `name`, `place`, `city`, `cuisines`, `price`, `rating`, `votes`. `rest_type` has a small number of nulls (33) — acceptable since it's supporting context, not a filter field.
