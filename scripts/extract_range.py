#!/usr/bin/env python3

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    REPO_ROOT
    / "results"
    / "races"
    / "all_results_cleaned.csv"
)

OUTPUT_FILE = (
    REPO_ROOT
    / "results"
    / "races"
    / "all_results_date_range.csv"
)


# ============================================================
# DATE RANGE
#
# These are supplied by the GitHub Actions workflow.
#
# Defaults are used if the script is run locally without
# environment variables.
# ============================================================

START_DATE = os.getenv(
    "START_DATE",
    "2006-01-01",
)

END_DATE = os.getenv(
    "END_DATE",
    "2015-12-31",
)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 72)
    print("EXTRACT HKJC RESULTS DATE RANGE")
    print("=" * 72)
    print()

    print(f"Input file:  {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Start date:  {START_DATE}")
    print(f"End date:    {END_DATE}")
    print()

    # ========================================================
    # VERIFY INPUT FILE
    # ========================================================

    if not INPUT_FILE.exists():
        print(
            f"ERROR: Input file does not exist: {INPUT_FILE}",
            file=sys.stderr,
        )
        return 1

    # ========================================================
    # PARSE DATE RANGE
    # ========================================================

    try:
        start_date = pd.Timestamp(
            START_DATE
        )

        end_date = pd.Timestamp(
            END_DATE
        )

    except Exception:
        print(
            "ERROR: START_DATE and END_DATE must use YYYY-MM-DD.",
            file=sys.stderr,
        )
        return 1

    if start_date > end_date:
        print(
            "ERROR: START_DATE cannot be after END_DATE.",
            file=sys.stderr,
        )
        return 1

    # ========================================================
    # LOAD SOURCE CSV
    #
    # dtype=str and keep_default_na=False help preserve the
    # source values as strings.
    # ========================================================

    try:
        df = pd.read_csv(
            INPUT_FILE,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )

    except Exception as exc:
        print(
            f"ERROR reading input CSV: {exc}",
            file=sys.stderr,
        )
        return 1

    if df.empty:
        print(
            "ERROR: Input CSV contains no rows.",
            file=sys.stderr,
        )
        return 1

    # ========================================================
    # VERIFY REQUIRED COLUMN
    # ========================================================

    if "race_date" not in df.columns:
        print(
            "ERROR: Required column 'race_date' was not found.",
            file=sys.stderr,
        )
        return 1

    original_columns = list(
        df.columns
    )

    # ========================================================
    # PARSE RACE DATES
    # ========================================================

    race_dates = pd.to_datetime(
        df["race_date"],
        errors="coerce",
    )

    invalid_dates = race_dates.isna()

    if invalid_dates.any():
        print(
            f"ERROR: Found {int(invalid_dates.sum()):,} rows "
            "with invalid race_date values.",
            file=sys.stderr,
        )
        return 1

    # ========================================================
    # FILTER
    #
    # START_DATE and END_DATE are both inclusive.
    #
    # Example:
    #
    # START_DATE=2006-01-01
    # END_DATE=2015-12-31
    #
    # includes all races from 2006 through 2015.
    # ========================================================

    mask = (
        (race_dates >= start_date)
        &
        (race_dates <= end_date)
    )

    filtered = (
        df.loc[
            mask,
            original_columns,
        ]
        .copy()
    )

    if filtered.empty:
        print(
            "ERROR: No rows matched the requested date range.",
            file=sys.stderr,
        )
        return 1

    # ========================================================
    # VERIFY OUTPUT DATE RANGE
    # ========================================================

    filtered_dates = pd.to_datetime(
        filtered["race_date"],
        errors="coerce",
    )

    earliest_date = filtered_dates.min()
    latest_date = filtered_dates.max()

    if earliest_date < start_date:
        print(
            "ERROR: Output contains rows before START_DATE.",
            file=sys.stderr,
        )
        return 1

    if latest_date > end_date:
        print(
            "ERROR: Output contains rows after END_DATE.",
            file=sys.stderr,
        )
        return 1

    # ========================================================
    # WRITE TO TEMPORARY FILE FIRST
    #
    # The source file is never opened for writing.
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_fd, temp_name = tempfile.mkstemp(
        prefix="all_results_date_range_",
        suffix=".csv",
        dir=OUTPUT_FILE.parent,
    )

    os.close(
        temp_fd
    )

    temp_path = Path(
        temp_name
    )

    try:
        filtered.to_csv(
            temp_path,
            index=False,
            lineterminator="\n",
        )

        # ====================================================
        # RELOAD OUTPUT FOR BASIC VALIDATION
        # ====================================================

        check_df = pd.read_csv(
            temp_path,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )

        if list(check_df.columns) != original_columns:
            raise RuntimeError(
                "Output column order does not match input."
            )

        if len(check_df) != len(filtered):
            raise RuntimeError(
                "Output row count does not match filtered row count."
            )

        # ====================================================
        # SAVE NEW OUTPUT FILE
        #
        # all_results_cleaned.csv is NOT modified.
        # ====================================================

        os.replace(
            temp_path,
            OUTPUT_FILE,
        )

    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()

        print(
            f"ERROR writing output CSV: {exc}",
            file=sys.stderr,
        )

        return 1

    # ========================================================
    # SUMMARY
    # ========================================================

    input_size_mb = (
        INPUT_FILE.stat().st_size
        /
        (1024 * 1024)
    )

    output_size_mb = (
        OUTPUT_FILE.stat().st_size
        /
        (1024 * 1024)
    )

    unique_races = (
        filtered["race_id"].nunique()
        if "race_id" in filtered.columns
        else 0
    )

    unique_horses = (
        filtered["horse_id"].nunique()
        if "horse_id" in filtered.columns
        else 0
    )

    print()
    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print()

    print(
        f"Input rows:       {len(df):,}"
    )

    print(
        f"Output rows:      {len(filtered):,}"
    )

    print(
        f"Rows excluded:    {len(df) - len(filtered):,}"
    )

    print(
        f"Unique races:     {unique_races:,}"
    )

    print(
        f"Unique horses:    {unique_horses:,}"
    )

    print(
        f"Earliest race:    {earliest_date.date()}"
    )

    print(
        f"Latest race:      {latest_date.date()}"
    )

    print(
        f"Input size:       {input_size_mb:.2f} MB"
    )

    print(
        f"Output size:      {output_size_mb:.2f} MB"
    )

    print()
    print(
        f"Created: {OUTPUT_FILE}"
    )

    print(
        "Original all_results_cleaned.csv was not modified."
    )

    print()

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
