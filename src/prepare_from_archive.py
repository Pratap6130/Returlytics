from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = BASE_DIR / "archive"
OUTPUT_PATH = BASE_DIR / "data" / "raw" / "data.csv"


def parse_price(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("â‚¹", "", regex=False)
        .str.replace(r"[^0-9.]", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_count(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def build_dataset() -> pd.DataFrame:
    csv_files = sorted(ARCHIVE_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {ARCHIVE_DIR}")

    frames = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception:
            continue

        lower_cols = {c.lower(): c for c in df.columns}
        needed = ["main_category", "sub_category", "discount_price", "actual_price"]
        if not all(col in lower_cols for col in needed):
            continue

        cat_col = lower_cols["main_category"]
        sub_col = lower_cols["sub_category"]
        disc_col = lower_cols["discount_price"]
        act_col = lower_cols["actual_price"]
        rating_col = lower_cols.get("ratings")
        rating_count_col = lower_cols.get("no_of_ratings")

        chunk = pd.DataFrame(
            {
                "product_category": df[sub_col].fillna(df[cat_col]).fillna("unknown"),
                "main_category": df[cat_col].fillna("unknown"),
                "discount_price": parse_price(df[disc_col]),
                "actual_price": parse_price(df[act_col]),
                "ratings": pd.to_numeric(df[rating_col], errors="coerce") if rating_col else np.nan,
                "no_of_ratings": parse_count(df[rating_count_col]) if rating_count_col else np.nan,
            }
        )
        frames.append(chunk)

    if not frames:
        raise ValueError("No usable CSV schema found in archive files")

    all_df = pd.concat(frames, ignore_index=True)

    all_df["product_price"] = all_df["actual_price"].fillna(all_df["discount_price"])
    all_df["product_price"] = all_df["product_price"].fillna(all_df["product_price"].median())
    all_df["discount_applied"] = (all_df["actual_price"] - all_df["discount_price"]).clip(lower=0)
    all_df["discount_applied"] = all_df["discount_applied"].fillna(0.0)

    rng = np.random.default_rng(42)

    all_df["order_quantity"] = rng.choice([1, 2, 3, 4], size=len(all_df), p=[0.62, 0.24, 0.1, 0.04])
    all_df["user_age"] = np.clip(rng.normal(loc=33, scale=10, size=len(all_df)).round(), 18, 70).astype(int)
    all_df["user_gender"] = rng.choice(["Male", "Female", "Other"], size=len(all_df), p=[0.48, 0.48, 0.04])
    all_df["payment_method"] = rng.choice(["Card", "COD", "UPI", "NetBanking"], size=len(all_df), p=[0.42, 0.22, 0.28, 0.08])
    all_df["shipping_method"] = rng.choice(["Standard", "Express", "Same-Day"], size=len(all_df), p=[0.64, 0.3, 0.06])

    rating = all_df["ratings"].fillna(4.0)
    rating_count = all_df["no_of_ratings"].fillna(50)
    discount_ratio = (all_df["discount_applied"] / all_df["product_price"].replace(0, np.nan)).fillna(0)

    # Synthetic label for demo/training pipeline validation only
    return_prob = (
        0.08
        + np.where(rating < 3.5, 0.14, 0.0)
        + np.where(discount_ratio > 0.35, 0.05, 0.0)
        + np.where(rating_count < 25, 0.03, 0.0)
        + np.where(all_df["main_category"].str.lower().isin(["fashion", "clothing", "shoes"]), 0.04, 0.0)
    )
    return_prob = np.clip(return_prob, 0.03, 0.6)
    all_df["return_status"] = np.where(rng.random(len(all_df)) < return_prob, "returned", "not returned")

    result = all_df[
        [
            "product_category",
            "product_price",
            "order_quantity",
            "user_age",
            "user_gender",
            "payment_method",
            "shipping_method",
            "discount_applied",
            "return_status",
        ]
    ].copy()

    result = result.dropna(subset=["product_category", "product_price", "return_status"])
    result["product_price"] = result["product_price"].astype(float).clip(lower=1.0)
    result["discount_applied"] = result["discount_applied"].astype(float).clip(lower=0.0)
    result["order_quantity"] = result["order_quantity"].astype(int)
    result["user_age"] = result["user_age"].astype(int)

    return result


def main() -> None:
    df = build_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    label_dist = df["return_status"].value_counts(normalize=True).round(4).to_dict()
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution: {label_dist}")


if __name__ == "__main__":
    main()
