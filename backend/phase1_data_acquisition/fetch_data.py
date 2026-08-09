"""
Phase 1 - Data Acquisition.

Pulls the Zomato restaurant recommendation dataset from Hugging Face
(ManikaSaini/zomato-restaurant-recommendation) and persists a raw,
untouched local copy under phase1_data_acquisition/data/raw/.
"""

import json
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "ManikaSaini/zomato-restaurant-recommendation"
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_ID)

    schema_info = {}
    for split_name, split_data in dataset.items():
        out_path = RAW_DIR / f"{split_name}.csv"
        split_data.to_csv(out_path, index=False)

        schema_info[split_name] = {
            "num_rows": split_data.num_rows,
            "columns": {
                name: str(feature) for name, feature in split_data.features.items()
            },
        }
        print(f"Saved split '{split_name}' ({split_data.num_rows} rows) -> {out_path}")

    schema_path = RAW_DIR / "schema.json"
    schema_path.write_text(json.dumps(schema_info, indent=2))
    print(f"Saved schema info -> {schema_path}")


if __name__ == "__main__":
    main()
