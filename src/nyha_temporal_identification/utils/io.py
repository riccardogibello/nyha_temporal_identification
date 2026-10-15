import json
import os
import shutil
from pathlib import Path
from typing import List, Literal, Optional

import polars as pl
import pyarrow.parquet as pq
from polars import ScanCastOptions

from nyha_temporal_identification.constants import (
    CLEANED_TABLES_FOLDER_NAME,
    COLUMNS_KEY,
)


def save_json_mapping(
    mapping: dict,
    file_path: str,
) -> None:
    """
    This method saves a JSON mapping to the given file path.

    Args:
        mapping (dict): The mapping to be saved.
        file_path (str): The path to the JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(mapping, f, indent=4)


def load_json_mapping(
    file_path: str,
    create_if_not_exists: bool = False,
) -> dict:
    """
    This method loads a JSON mapping from the given file path.

    Args:
        file_path (str): The path to the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.

    Returns:
        dict: The loaded JSON mapping.
    """
    if not os.path.exists(file_path) and not create_if_not_exists:
        raise FileNotFoundError(f"Mapping file not found at: {file_path}")
    elif not os.path.exists(file_path) and create_if_not_exists:
        with open(file_path, "w") as f:
            json.dump({}, f)
    with open(file_path, "r") as f:
        return json.load(f)


def save_table(
    df: pl.DataFrame,
    data_folder_path: str,
    table_name: str,
    use_dataset: bool = True,  # if True, write as Parquet dataset
    reset_previous_data: bool = False,  # if True, clear existing data before saving
) -> None:
    """
    Safely store the cleaned table as a Parquet file or dataset, keeping columns
    consistent with existing tables and applying default column type mapping.

    Args:
        df: The dataframe to save.
        data_folder_path: Base path to the data folder.
        table_name: Name of the table (used as file or dataset folder).
        use_dataset: Whether to write as a Parquet dataset (directory) or single file.
        reset_previous_data: If True, delete the existing file or dataset before saving.
    """

    # Path to cleaned table
    cleaned_table_path = os.path.join(
        data_folder_path,
        CLEANED_TABLES_FOLDER_NAME,
        table_name,
    )

    # If reset_previous_data is True, remove existing file or dataset
    if reset_previous_data:
        if os.path.exists(cleaned_table_path):
            if os.path.isdir(cleaned_table_path):
                shutil.rmtree(cleaned_table_path)
            else:
                os.remove(cleaned_table_path)
    # If the table name specifies a nested structure (e.g. "subdir/table"),
    # we need to make sure the parent directory exists.
    os.makedirs(os.path.dirname(cleaned_table_path), exist_ok=True)

    # If table exists, align columns with existing table (Polars)
    # If not, the table will be created with the current dataframe's schema
    if use_dataset:
        if os.path.exists(cleaned_table_path):
            existing_files = [
                f for f in os.listdir(cleaned_table_path) if f.endswith(".parquet")
            ]
            if existing_files:
                # Read header from first part file using Polars
                existing_header_df = pl.read_parquet(
                    os.path.join(cleaned_table_path, existing_files[0])
                ).head(0)
                header_cols = existing_header_df.columns
                # Add missing columns with None
                for col in header_cols:
                    if col not in df.columns:
                        df = df.with_columns(pl.lit(None).alias(col))
                # Align column order
                df = df.select(header_cols)
        else:
            # Create the directory if it does not exist
            os.makedirs(cleaned_table_path, exist_ok=True)
    else:
        if os.path.exists(cleaned_table_path):
            existing_header_df = pl.read_parquet(cleaned_table_path).head(0)
            header_cols = existing_header_df.columns
            for col in header_cols:
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None).alias(col))
            df = df.select(header_cols)
        # If the file does not exist, it will be created by pq.write_table later

    # Drop duplicate rows
    df = df.unique()

    # Convert to Arrow table
    table = df.to_arrow()

    if use_dataset:
        # Parquet dataset directory: append as new part file
        existing_files = (
            [f for f in os.listdir(cleaned_table_path) if f.endswith(".parquet")]
            if os.path.exists(cleaned_table_path)
            else []
        )
        # If any of the parts contains 0, remove it
        for fname in existing_files:
            part_path = os.path.join(cleaned_table_path, fname)
            part_df = pl.read_parquet(part_path)
            if part_df.shape[0] == 0:
                os.remove(part_path)
        # Recompute existing files after potential removals
        existing_files = (
            [f for f in os.listdir(cleaned_table_path) if f.endswith(".parquet")]
            if os.path.exists(cleaned_table_path)
            else []
        )
        part_numbers = []
        for fname in existing_files:
            try:
                num = int(fname.replace("part-", "").replace(".parquet", ""))
                part_numbers.append(num)
            except Exception:
                continue
        next_part = max(part_numbers) + 1 if part_numbers else 0
        part_path = os.path.join(cleaned_table_path, f"part-{next_part}.parquet")
        pq.write_table(table, part_path)
    else:
        # Single Parquet file: append by reading, concatenating, and rewriting
        if os.path.exists(cleaned_table_path):
            existing_df = pl.read_parquet(cleaned_table_path)
            # Align columns
            for col in existing_df.columns:
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None).alias(col))
            for col in df.columns:
                if col not in existing_df.columns:
                    existing_df = existing_df.with_columns(pl.lit(None).alias(col))
            df = df.select(existing_df.columns)
            combined_df = pl.concat([existing_df, df]).unique()
            table = combined_df.to_arrow()
        pq.write_table(table, cleaned_table_path)


def ensure_output_parquet(
    output_folder: str,
    file_column_mapping: dict[str, dict[str, list[str]]],
    column_types_mapping: dict[str, str],
) -> None:
    """
    Ensure that Parquet outputs exist with the correct schema.

    - If table_name ends with ".parquet": create a single Parquet file
    - Otherwise: create a Parquet dataset directory
    """

    os.makedirs(output_folder, exist_ok=True)

    # Polars dtype mapping
    polars_dtype_mapping = {
        "integer": pl.Int64,
        "float": pl.Float64,
        "string": pl.Utf8,
        "date": pl.Date,
        "datetime": pl.Datetime,
        "time": pl.Time,
    }

    for table_name, table_map in file_column_mapping.items():
        columns = table_map.get(COLUMNS_KEY, [])
        path = os.path.join(output_folder, table_name)

        # -------- Build Polars schema (NO DATA) --------
        schema = {
            col: polars_dtype_mapping[column_types_mapping[col]] for col in columns
        }

        # Empty Polars DataFrame with schema only
        pl_df = pl.DataFrame(schema=schema)

        # Convert schema-only DF to Arrow table
        arrow_table = pl_df.to_arrow()

        if table_name.endswith(".parquet"):
            # ---- Single parquet file ----
            if not os.path.exists(path):
                pq.write_table(arrow_table, path)
        else:
            # ---- Parquet dataset (directory) ----
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                part_path = os.path.join(path, "part-0.parquet")
                pq.write_table(arrow_table, part_path)


def load_parquet_dataset_to_polars(
    parquet_path: str,
    columns: Optional[List[str]] = None,
    header_only: bool = False,
    first_row: Optional[int] = None,
    last_row: Optional[int] = None,
    filters: Optional[List[pl.Expr]] = None,
    filter_operator: Literal["and", "or"] = "and",
) -> pl.DataFrame:
    """
    Load a Parquet file or dataset into a Polars DataFrame with optional filtering.

    Args:
        parquet_path (str): Path to Parquet file or directory.
        columns (List[str], optional): Columns to load.
        header_only (bool): If True, return empty DataFrame with schema only.
        first_row (int, optional): First row index (inclusive).
        last_row (int, optional): Last row index (exclusive).
        filters (List[pl.Expr], optional): List of Polars filter expressions to apply
            before loading data. Filters are applied at scan time for efficiency.
        filter_operator (Literal["and", "or"]): How to combine multiple filters.
            Default is "and".

    Returns:
        pl.DataFrame

    Examples:
        # Load with a single filter
        df = load_parquet_dataset_to_polars(
            "data.parquet",
            filters=[pl.col("age") > 18]
        )

        # Load with multiple filters combined with AND
        df = load_parquet_dataset_to_polars(
            "data.parquet",
            filters=[
                pl.col("age") > 18,
                pl.col("country") == "USA"
            ],
            filter_operator="and"
        )

        # Load with multiple filters combined with OR
        df = load_parquet_dataset_to_polars(
            "data.parquet",
            filters=[
                pl.col("status") == "active",
                pl.col("status") == "pending"
            ],
            filter_operator="or"
        )
    """
    if not os.path.exists(parquet_path):
        return pl.DataFrame()

    lf = pl.scan_parquet(
        parquet_path,
        cast_options=ScanCastOptions(
            integer_cast="upcast",
        ),
    )

    # Apply filters early (before column selection) for efficiency
    if filters is not None and len(filters) > 0:
        combined_filter = filters[0]
        for expr in filters[1:]:
            combined_filter = (
                combined_filter & expr
                if filter_operator == "and"
                else combined_filter | expr
            )
        lf = lf.filter(combined_filter)

    if columns is not None:
        schema_cols = set(lf.collect_schema().names())
        selected_cols = [c for c in columns if c in schema_cols]
        lf = lf.select(selected_cols)

    if header_only:
        schema = lf.collect_schema()
        # Return empty dataframe with correct schema
        return pl.DataFrame(
            {name: pl.Series([], dtype=dtype) for name, dtype in schema.items()}
        )

    if first_row is not None or last_row is not None:
        start = first_row or 0
        if last_row is not None:
            lf = lf.slice(start, last_row - start)
        else:
            lf = lf.slice(start)

    result = lf.collect(engine="streaming")
    return pl.DataFrame(result)


def find_file_in_subfolders(base_folder, target_file=None):
    base = Path(base_folder)
    if not base.exists():
        raise FileNotFoundError(f"Base folder not found: {base_folder}")

    # only list files in the top-level directory (no recursion)
    if target_file is None:
        return [str(p) for p in base.iterdir() if p.is_file()]

    # return first matching top-level file (substring match on name)
    for p in base.iterdir():
        if p.is_file() and target_file in p.name:
            return str(p)
    return None
