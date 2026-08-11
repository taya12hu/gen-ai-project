# Raw Dataset Schema — Phase 1

**Source:** [ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)
**Split:** `train` only — 51,717 rows
**Saved to:** `data/raw/train.csv` (raw, untouched copy)

## Columns

| Column | Type | Notes |
|---|---|---|
| `url` | string | Zomato listing URL |
| `address` | string | Full address |
| `name` | string | Restaurant name |
| `online_order` | string | Yes/No |
| `book_table` | string | Yes/No |
| `rate` | string | e.g. `"4.1/5"` — needs parsing to numeric. Some values may be `"NEW"` or missing. |
| `votes` | int64 | Number of votes |
| `phone` | string | Phone number(s) |
| `location` | string | Neighborhood/area, e.g. `"Banashankari"` |
| `rest_type` | string | Restaurant type, e.g. `"Casual Dining"` |
| `dish_liked` | string | Free-text popular dishes |
| `cuisines` | string | Comma-separated, e.g. `"North Indian, Mughlai, Chinese"` |
| `approx_cost(for two people)` | string | Numeric-as-string, e.g. `"800"` — has commas in some values, needs parsing |
| `reviews_list` | string | Free-text review blob |
| `menu_item` | string | Free-text menu items |
| `listed_in(type)` | string | Listing category (Dine-out, Delivery, etc.) |
| `listed_in(city)` | string | City-level area grouping |

## Mapping to Required Preferences

| Preference | Source column(s) |
|---|---|
| **Price** | `approx_cost(for two people)` |
| **Place** | `location` (neighborhood) and/or `listed_in(city)` (broader area) |
| **Rating** | `rate` (strip `"/5"`, handle `"NEW"`/missing) |
| **Cuisine** | `cuisines` (comma-separated, needs splitting) |

This confirms all four required preference fields are present in the raw data. Cleaning/normalization (parsing `rate` and `approx_cost`, splitting `cuisines`, deduplicating `location`/`listed_in(city)`) is handled in Phase 2.
