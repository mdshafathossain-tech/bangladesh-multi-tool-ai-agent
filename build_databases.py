#!/usr/bin/env python3
"""
build_databases.py
===================

Downloads three Bangladesh-related datasets from the HuggingFace Hub and
converts each of them into its own standalone SQLite database:

    1. Mahadih534/Institutional-Information-of-Bangladesh -> institutions.db  (table: institutions)
    2. Mahadih534/all-bangladeshi-hospitals               -> hospitals.db     (table: hospitals)
    3. Mahadih534/Bangladeshi-Restaurant-Data              -> restaurants.db   (table: restaurants)

For each dataset the script:
    * Downloads the data via the `datasets` library (falls back to reading a
      raw CSV directly from the Hub if that fails).
    * Cleans column names -> lowercase, spaces/special characters -> underscores.
    * Infers an explicit SQL column type (INTEGER, REAL, or TEXT) for every
      column based on the underlying pandas dtype (with a small manual
      override table for well-known columns such as `capacity` and `rating`).
    * Creates a fresh SQLite database file with an explicit `CREATE TABLE`
      statement (so column types are guaranteed, not just SQLite's default
      dynamic typing) and bulk-loads the cleaned data into it.

Usage
-----
    python build_databases.py

Requirements
------------
    pip install pandas datasets huggingface_hub

Notes
-----
This script is intentionally defensive: if one dataset fails to download or
convert, it prints a clear error and moves on to the next one rather than
crashing the whole run.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DatasetConfig:
    """Everything needed to pull one HF dataset into one SQLite database."""

    repo_id: str                     # HuggingFace dataset repo id
    db_filename: str                 # output .db file
    table_name: str                  # SQL table name
    # Optional manual overrides for specific *cleaned* column names.
    # Anything not listed here is auto-inferred from the pandas dtype.
    type_overrides: dict = field(default_factory=dict)


DATASETS: list[DatasetConfig] = [
    DatasetConfig(
        repo_id="Mahadih534/Institutional-Information-of-Bangladesh",
        db_filename="institutions.db",
        table_name="institutions",
        type_overrides={
            "capacity": "INTEGER",
            "rating": "REAL",
        },
    ),
    DatasetConfig(
        repo_id="Mahadih534/all-bangladeshi-hospitals",
        db_filename="hospitals.db",
        table_name="hospitals",
        type_overrides={
            "capacity": "INTEGER",
            "beds": "INTEGER",
            "rating": "REAL",
        },
    ),
    DatasetConfig(
        repo_id="Mahadih534/Bangladeshi-Restaurant-Data",
        db_filename="restaurants.db",
        table_name="restaurants",
        type_overrides={
            "rating": "REAL",
            "capacity": "INTEGER",
        },
    ),
]


# --------------------------------------------------------------------------- #
# Column name cleaning
# --------------------------------------------------------------------------- #

def clean_column_name(raw_name: str) -> str:
    """
    Normalize a raw column name into a safe, predictable snake_case identifier.

    Examples
    --------
    "Institution Name"   -> "institution_name"
    "Phone-Number"        -> "phone_number"
    "Rating (out of 5)"   -> "rating_out_of_5"
    "  Type  "            -> "type"
    """
    name = str(raw_name).strip().lower()
    # Replace anything that isn't a letter, digit, or underscore with an underscore.
    name = re.sub(r"[^a-z0-9]+", "_", name)
    # Collapse repeated underscores and strip leading/trailing ones.
    name = re.sub(r"_+", "_", name).strip("_")
    # Guard against an empty/blank result (e.g. a column that was all symbols).
    if not name:
        name = "unnamed_column"
    # SQL identifiers shouldn't start with a digit.
    if name[0].isdigit():
        name = f"col_{name}"
    return name


def deduplicate_columns(columns: list[str]) -> list[str]:
    """If cleaning produces duplicate names, disambiguate them with a suffix."""
    seen: dict[str, int] = {}
    result = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
    return result


# --------------------------------------------------------------------------- #
# Type inference
# --------------------------------------------------------------------------- #

def infer_sql_type(series: pd.Series) -> str:
    """
    Map a pandas Series' dtype to an explicit SQLite column type.

    Falls back to TEXT for anything that isn't clearly integer or float,
    which keeps string/mixed/NaN-heavy columns safe.
    """
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        # Store booleans as 0/1 integers.
        return "INTEGER"
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    # Object columns that are *actually* numeric (e.g. "12", "3.5", or NaN
    # mixed in) get a chance at INTEGER/REAL before we give up and use TEXT.
    if pd.api.types.is_object_dtype(dtype):
        non_null = series.dropna()
        if len(non_null) > 0:
            coerced = pd.to_numeric(non_null, errors="coerce")
            if coerced.notna().all():
                # All non-null values are numeric strings.
                if (coerced % 1 == 0).all():
                    return "INTEGER"
                return "REAL"
    return "TEXT"


def build_dtype_map(df: pd.DataFrame, overrides: dict) -> dict:
    """Build the final {column_name: SQL_TYPE} map, applying manual overrides."""
    dtype_map = {}
    for col in df.columns:
        if col in overrides:
            dtype_map[col] = overrides[col]
        else:
            dtype_map[col] = infer_sql_type(df[col])
    return dtype_map


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_dataframe(repo_id: str) -> pd.DataFrame:
    """
    Load a HuggingFace dataset repo into a pandas DataFrame.

    Strategy:
      1. Try the `datasets` library (handles auth, caching, parquet/CSV
         conversion, and multiple files transparently).
      2. Fall back to locating and reading a raw CSV file directly from the
         Hub via `huggingface_hub`, in case the repo isn't a "proper"
         datasets-library-compatible dataset (e.g. it's just loose CSV files).
    """
    # --- Strategy 1: `datasets` library -------------------------------------
    try:
        from datasets import load_dataset

        ds = load_dataset(repo_id, split="train")
        df = ds.to_pandas()
        if df is not None and len(df) > 0:
            return df
        raise ValueError("`datasets` returned an empty split.")
    except Exception as primary_error:
        print(f"  [info] `datasets` library load failed ({primary_error}); "
              f"falling back to direct CSV download...")

    # --- Strategy 2: locate and read a raw CSV via huggingface_hub ---------
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        repo_files = api.list_repo_files(repo_id, repo_type="dataset")
        csv_files = [f for f in repo_files if f.lower().endswith(".csv")]

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in repo '{repo_id}' to fall back on."
            )

        # Prefer a file that looks like the "main" data file, otherwise take
        # the first (often only) CSV in the repo.
        csv_files.sort(key=len)
        target_file = csv_files[0]

        local_path = hf_hub_download(
            repo_id=repo_id, filename=target_file, repo_type="dataset"
        )
        df = pd.read_csv(local_path)
        return df

    except Exception as fallback_error:
        raise RuntimeError(
            f"Could not load dataset '{repo_id}' via either the `datasets` "
            f"library or a direct CSV download. Last error: {fallback_error}"
        ) from fallback_error


# --------------------------------------------------------------------------- #
# Database creation
# --------------------------------------------------------------------------- #

def create_database(df: pd.DataFrame, dtype_map: dict, db_path: str, table_name: str) -> int:
    """
    Create (or overwrite) a SQLite database file containing `table_name`,
    with an explicit CREATE TABLE statement using `dtype_map`, then bulk
    insert the DataFrame's rows.

    Returns the number of rows written.
    """
    # Start from a clean file each run so re-running the script is idempotent.
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        columns_sql = ", ".join(
            f'"{col}" {dtype_map[col]}' for col in df.columns
        )
        create_stmt = f'CREATE TABLE "{table_name}" ({columns_sql});'
        cursor.execute(create_stmt)

        # Prepare data: coerce values to match declared types where sensible,
        # and convert pandas/NumPy NaN to SQL NULL.
        insert_df = df.copy()
        for col in insert_df.columns:
            if dtype_map[col] == "INTEGER":
                insert_df[col] = pd.to_numeric(insert_df[col], errors="coerce")
                # Use pandas' nullable Int64 so missing values become NULL, not 0.
                insert_df[col] = insert_df[col].astype("Int64")
            elif dtype_map[col] == "REAL":
                insert_df[col] = pd.to_numeric(insert_df[col], errors="coerce").astype("float64")
            else:  # TEXT
                insert_df[col] = insert_df[col].apply(
                    lambda v: None if pd.isna(v) else str(v)
                )

        # Convert to a list of tuples with proper Python-native None for NULLs.
        records = insert_df.astype(object).where(pd.notnull(insert_df), None).values.tolist()

        placeholders = ", ".join(["?"] * len(df.columns))
        insert_stmt = f'INSERT INTO "{table_name}" VALUES ({placeholders});'
        cursor.executemany(insert_stmt, records)

        conn.commit()
        return cursor.rowcount if cursor.rowcount != -1 else len(records)
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Per-dataset pipeline
# --------------------------------------------------------------------------- #

def process_dataset(config: DatasetConfig) -> bool:
    """Run the full download -> clean -> type-infer -> load pipeline for one dataset."""
    print(f"\n{'=' * 70}")
    print(f"Processing dataset: {config.repo_id}")
    print(f"{'=' * 70}")

    try:
        print(f"  -> Downloading / reading data from HuggingFace Hub...")
        df = load_dataframe(config.repo_id)
        print(f"  -> Loaded {len(df):,} rows and {len(df.columns)} columns.")

        print(f"  -> Cleaning column names...")
        original_columns = list(df.columns)
        cleaned_columns = deduplicate_columns(
            [clean_column_name(c) for c in original_columns]
        )
        df.columns = cleaned_columns
        for orig, cleaned in zip(original_columns, cleaned_columns):
            if orig != cleaned:
                print(f"       '{orig}' -> '{cleaned}'")

        print(f"  -> Inferring SQL column types...")
        dtype_map = build_dtype_map(df, config.type_overrides)
        for col, sql_type in dtype_map.items():
            print(f"       {col}: {sql_type}")

        db_path = os.path.join(OUTPUT_DIR, config.db_filename)
        print(f"  -> Writing database: {db_path} (table: {config.table_name})")
        row_count = create_database(df, dtype_map, db_path, config.table_name)

        print(f"  [OK] Created '{config.db_filename}' with table "
              f"'{config.table_name}' ({row_count:,} rows, "
              f"{len(dtype_map)} columns).")
        return True

    except Exception as exc:
        print(f"  [ERROR] Failed to process '{config.repo_id}': {exc}")
        print("  --- traceback ---")
        traceback.print_exc()
        print("  -----------------")
        return False


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    print("Bangladesh Multi-Dataset -> SQLite Builder")
    print(f"Output directory: {OUTPUT_DIR}")

    results = {}
    for config in DATASETS:
        results[config.db_filename] = process_dataset(config)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    all_ok = True
    for db_filename, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {db_filename:<25} {status}")
        all_ok = all_ok and success

    if all_ok:
        print("\nAll databases created successfully.")
        return 0
    else:
        print("\nOne or more databases failed to build. See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
