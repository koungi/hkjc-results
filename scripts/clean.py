#!/usr/bin/env python3

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


# ============================================================
# FILE
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    REPO_ROOT
    / "results"
    / "races"
    / "all_results_cleaned.csv"
)


# ============================================================
# HISTORICAL PAYOUT RULE
#
# Applies ONLY to races BEFORE 10 September 2023.
#
# 1st = 57%
# 2nd = 22%
# 3rd = 11.5%
# 4th = 6%
# 5th = 3.5%
#
# 6th and below = 0%
#
# Races on or after 2023-09-10 are left with their existing
# payout percentage and prize-money-won values.
# ============================================================

PAYOUT_CUTOFF = pd.Timestamp("2023-09-10")

HISTORICAL_PAYOUT_SCHEDULE = {
    1: 0.57,
    2: 0.22,
    3: 0.115,
    4: 0.06,
    5: 0.035,
}


# ============================================================
# ONLY THESE COLUMNS MAY CHANGE
# ============================================================

TARGET_COLUMNS = [
    "prize_payout_percentage",
    "prize_money_won_this_race",
    "career_prize_money_before",
    "career_prize_money_after",
]


REQUIRED_COLUMNS = [
    "race_id",
    "race_date",
    "race_number",
    "horse_id",
    "finishing_position",
    "prize_money_hkd",
    "prize_payout_percentage",
    "prize_money_won_this_race",
    "career_prize_money_before",
    "career_prize_money_after",
]


# ============================================================
# DEAD-HEAT CALCULATION
#
# This preserves the dead-heat method from your original code.
#
# Examples:
#
# Two horses dead heat for 1st:
#
#   57% + 22% = 79%
#   79% / 2 = 39.5% each
#
# Two horses dead heat for 2nd:
#
#   22% + 11.5% = 33.5%
#   33.5% / 2 = 16.75% each
#
# Three horses dead heat for 1st:
#
#   57% + 22% + 11.5% = 90.5%
#   90.5% / 3 = 30.1666667% each
#
# If a dead heat extends beyond 5th, positions outside the
# schedule contribute 0%.
# ============================================================

def calculate_historical_dead_heat_percentages(df):
    payout = pd.Series(
        0.0,
        index=df.index,
        dtype="float64",
    )

    dead_heat_races = set()
    dead_heat_groups = 0

    historical_df = df[
        df["_historical_race"]
    ]

    for race_id, race_group in historical_df.groupby(
        "race_id",
        sort=False,
        dropna=False,
    ):
        if race_group.empty:
            continue

        valid_finishers = race_group[
            race_group["_finish_numeric"].notna()
        ]

        if valid_finishers.empty:
            continue

        for finishing_position, position_group in (
            valid_finishers.groupby(
                "_finish_numeric",
                sort=True,
            )
        ):
            position = int(
                finishing_position
            )

            number_tied = len(
                position_group
            )

            # ------------------------------------------------
            # NORMAL RESULT
            # ------------------------------------------------

            if number_tied == 1:
                percentage = (
                    HISTORICAL_PAYOUT_SCHEDULE.get(
                        position,
                        0.0,
                    )
                )

            # ------------------------------------------------
            # DEAD HEAT
            #
            # Pool the percentages for the positions consumed
            # by the dead heat and divide equally.
            # ------------------------------------------------

            else:
                dead_heat_races.add(
                    race_id
                )

                dead_heat_groups += 1

                combined_percentage = sum(
                    HISTORICAL_PAYOUT_SCHEDULE.get(
                        position + offset,
                        0.0,
                    )
                    for offset in range(
                        number_tied
                    )
                )

                percentage = (
                    combined_percentage
                    /
                    number_tied
                )

            payout.loc[
                position_group.index
            ] = percentage

    return (
        payout,
        dead_heat_races,
        dead_heat_groups,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 72)
    print("HKJC HISTORICAL PRIZE MONEY CORRECTION")
    print("=" * 72)
    print()

    print(
        f"File: {RESULTS_FILE}"
    )

    print(
        "Historical rule applies to: "
        "race_date < 2023-09-10"
    )

    print()

    # ========================================================
    # VERIFY FILE EXISTS
    # ========================================================

    if not RESULTS_FILE.exists():
        print(
            "ERROR: File does not exist:",
            RESULTS_FILE,
            file=sys.stderr,
        )

        return 1

    # ========================================================
    # LOAD CSV
    #
    # All columns are loaded as strings so non-target fields
    # are preserved as closely as possible.
    # ========================================================

    try:
        df = pd.read_csv(
            RESULTS_FILE,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )

    except Exception as exc:
        print(
            f"ERROR reading CSV: {exc}",
            file=sys.stderr,
        )

        return 1

    if df.empty:
        print(
            "ERROR: CSV contains no rows.",
            file=sys.stderr,
        )

        return 1

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns):,}"
    )

    # ========================================================
    # VERIFY REQUIRED COLUMNS
    # ========================================================

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print()
        print(
            "ERROR: Required columns are missing:",
            file=sys.stderr,
        )

        for column in missing_columns:
            print(
                f"  - {column}",
                file=sys.stderr,
            )

        return 1

    # ========================================================
    # VERIFY HORSE ID
    #
    # Career prize money depends on grouping by horse_id.
    # We do not want blank identities being accumulated
    # together accidentally.
    # ========================================================

    blank_horse_ids = (
        df["horse_id"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    if blank_horse_ids.any():
        print()
        print(
            "ERROR: Blank horse_id values found:",
            int(blank_horse_ids.sum()),
            file=sys.stderr,
        )

        print(
            "Career prize money cannot safely be rebuilt.",
            file=sys.stderr,
        )

        return 1

    # ========================================================
    # SAVE ORIGINAL COLUMN ORDER
    # ========================================================

    original_columns = list(
        df.columns
    )

    # ========================================================
    # SAVE A COPY OF ALL NON-TARGET COLUMNS
    #
    # Later we verify that none changed.
    # ========================================================

    non_target_columns = [
        column
        for column in original_columns
        if column not in TARGET_COLUMNS
    ]

    original_non_target = (
        df[
            non_target_columns
        ]
        .copy(deep=True)
    )

    # ========================================================
    # SAVE ORIGINAL ROW ORDER
    # ========================================================

    df["_original_order"] = range(
        len(df)
    )

    # ========================================================
    # TEMPORARY PARSED COLUMNS
    # ========================================================

    df["_race_date_sort"] = pd.to_datetime(
        df["race_date"],
        errors="coerce",
    )

    invalid_dates = (
        df["_race_date_sort"]
        .isna()
    )

    if invalid_dates.any():
        print()
        print(
            "ERROR: Invalid race_date values found:",
            int(invalid_dates.sum()),
            file=sys.stderr,
        )

        return 1

    df["_race_number_sort"] = pd.to_numeric(
        df["race_number"],
        errors="coerce",
    ).fillna(0)

    if "race_index" in df.columns:
        df["_race_index_sort"] = pd.to_numeric(
            df["race_index"],
            errors="coerce",
        ).fillna(0)

    df["_finish_numeric"] = pd.to_numeric(
        df["finishing_position"],
        errors="coerce",
    )

    df["_race_prize_numeric"] = pd.to_numeric(
        df["prize_money_hkd"],
        errors="coerce",
    ).fillna(0.0)

    # ========================================================
    # HISTORICAL RACE MASK
    #
    # STRICTLY BEFORE 10 September 2023.
    #
    # 2023-09-09 -> corrected
    # 2023-09-10 -> untouched
    # ========================================================

    df["_historical_race"] = (
        df["_race_date_sort"]
        <
        PAYOUT_CUTOFF
    )

    historical_mask = (
        df["_historical_race"]
    )

    historical_rows = int(
        historical_mask.sum()
    )

    historical_races = (
        df.loc[
            historical_mask,
            "race_id",
        ]
        .nunique()
    )

    later_rows = (
        len(df)
        -
        historical_rows
    )

    print()
    print(
        f"Historical rows: {historical_rows:,}"
    )

    print(
        f"Historical races: {historical_races:,}"
    )

    print(
        f"Rows on/after cutoff: {later_rows:,}"
    )

    # ========================================================
    # PRESERVE CURRENT VALUES FOR REPORTING
    # ========================================================

    original_payout = pd.to_numeric(
        df["prize_payout_percentage"],
        errors="coerce",
    ).fillna(0.0)

    original_prize_won = pd.to_numeric(
        df["prize_money_won_this_race"],
        errors="coerce",
    ).fillna(0.0)

    # ========================================================
    # CALCULATE CORRECTED HISTORICAL PAYOUT PERCENTAGES
    # ========================================================

    (
        historical_payout,
        dead_heat_races,
        dead_heat_groups,
    ) = calculate_historical_dead_heat_percentages(
        df
    )

    # ========================================================
    # UPDATE HISTORICAL PAYOUT PERCENTAGE
    #
    # ONLY historical rows are modified.
    #
    # Post-cutoff rows retain their existing values.
    # ========================================================

    df.loc[
        historical_mask,
        "prize_payout_percentage",
    ] = (
        historical_payout.loc[
            historical_mask
        ]
        .round(10)
        .astype(str)
        .values
    )

    # ========================================================
    # CALCULATE HISTORICAL PRIZE MONEY WON
    #
    # prize_money_hkd * payout percentage
    # ========================================================

    corrected_historical_prize = (
        df.loc[
            historical_mask,
            "_race_prize_numeric",
        ]
        *
        historical_payout.loc[
            historical_mask
        ]
    ).round(2)

    df.loc[
        historical_mask,
        "prize_money_won_this_race",
    ] = (
        corrected_historical_prize
        .astype(str)
        .values
    )

    # ========================================================
    # IMPORTANT:
    #
    # Races on/after 2023-09-10 have NOT had:
    #
    # prize_payout_percentage
    # prize_money_won_this_race
    #
    # changed.
    # ========================================================

    # ========================================================
    # PREPARE PRIZE MONEY FOR CAREER CALCULATION
    # ========================================================

    df["_prize_won_numeric"] = pd.to_numeric(
        df["prize_money_won_this_race"],
        errors="coerce",
    ).fillna(0.0)

    # ========================================================
    # SORT COMPLETE DATASET CHRONOLOGICALLY BY HORSE
    #
    # Dataset is assumed complete from 2006 through 2025.
    # ========================================================

    sort_columns = [
        "horse_id",
        "_race_date_sort",
        "_race_number_sort",
    ]

    if "_race_index_sort" in df.columns:
        sort_columns.append(
            "_race_index_sort"
        )

    sort_columns.append(
        "race_id"
    )

    df = (
        df.sort_values(
            sort_columns,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # REBUILD COMPLETE CAREER PRIZE MONEY
    # ========================================================

    grouped = df.groupby(
        "horse_id",
        sort=False,
        dropna=False,
    )

    df["_career_after_numeric"] = (
        grouped[
            "_prize_won_numeric"
        ]
        .cumsum()
        .round(2)
    )

    df["_career_before_numeric"] = (
        df["_career_after_numeric"]
        -
        df["_prize_won_numeric"]
    ).round(2)

    df[
        "career_prize_money_before"
    ] = (
        df[
            "_career_before_numeric"
        ]
        .astype(str)
    )

    df[
        "career_prize_money_after"
    ] = (
        df[
            "_career_after_numeric"
        ]
        .astype(str)
    )

    # ========================================================
    # CAREER PRIZE VALIDATION
    #
    # career_after - career_before must equal
    # prize_money_won_this_race
    # ========================================================

    career_difference = (
        df["_career_after_numeric"]
        -
        df["_career_before_numeric"]
        -
        df["_prize_won_numeric"]
    ).abs()

    invalid_career_rows = (
        career_difference
        >
        0.01
    )

    if invalid_career_rows.any():
        print()
        print(
            "ERROR: Career prize validation failed.",
            file=sys.stderr,
        )

        print(
            "Invalid rows:",
            int(invalid_career_rows.sum()),
            file=sys.stderr,
        )

        return 1

    # ========================================================
    # RESTORE ORIGINAL ROW ORDER
    # ========================================================

    df = (
        df.sort_values(
            "_original_order",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # CHANGE SUMMARY
    # ========================================================

    new_payout = pd.to_numeric(
        df["prize_payout_percentage"],
        errors="coerce",
    ).fillna(0.0)

    new_prize_won = pd.to_numeric(
        df["prize_money_won_this_race"],
        errors="coerce",
    ).fillna(0.0)

    payout_changed = (
        (
            new_payout
            -
            original_payout
        ).abs()
        >
        0.000000001
    )

    prize_changed = (
        (
            new_prize_won
            -
            original_prize_won
        ).abs()
        >
        0.01
    )

    print()
    print("=" * 72)
    print("CORRECTION SUMMARY")
    print("=" * 72)

    print(
        "Payout percentage rows changed:",
        f"{int(payout_changed.sum()):,}",
    )

    print(
        "Prize-money rows changed:",
        f"{int(prize_changed.sum()):,}",
    )

    print(
        "Dead-heat races found:",
        f"{len(dead_heat_races):,}",
    )

    print(
        "Dead-heat placing groups:",
        f"{dead_heat_groups:,}",
    )

    # ========================================================
    # DROP TEMPORARY COLUMNS
    # ========================================================

    temporary_columns = [
        "_original_order",
        "_race_date_sort",
        "_race_number_sort",
        "_race_index_sort",
        "_finish_numeric",
        "_race_prize_numeric",
        "_historical_race",
        "_prize_won_numeric",
        "_career_after_numeric",
        "_career_before_numeric",
    ]

    df.drop(
        columns=temporary_columns,
        inplace=True,
        errors="ignore",
    )

    # ========================================================
    # RESTORE EXACT ORIGINAL COLUMN ORDER
    # ========================================================

    df = df[
        original_columns
    ]

    # ========================================================
    # VERIFY NO NON-PRIZE COLUMN CHANGED
    # ========================================================

    current_non_target = (
        df[
            non_target_columns
        ]
        .copy()
    )

    if not current_non_target.equals(
        original_non_target
    ):
        print()
        print(
            "ERROR: A column outside the four permitted "
            "prize columns changed.",
            file=sys.stderr,
        )

        print(
            "The cleaned CSV will NOT be overwritten.",
            file=sys.stderr,
        )

        return 1

    print()
    print(
        "Validation passed:"
    )

    print(
        "No non-prize column values changed."
    )

    # ========================================================
    # WRITE TO TEMPORARY FILE FIRST
    #
    # The real all_results_cleaned.csv is only replaced after
    # the complete write succeeds.
    # ========================================================

    temp_fd, temp_name = tempfile.mkstemp(
        prefix="all_results_cleaned_",
        suffix=".csv",
        dir=RESULTS_FILE.parent,
    )

    os.close(
        temp_fd
    )

    temp_path = Path(
        temp_name
    )

    try:
        df.to_csv(
            temp_path,
            index=False,
            lineterminator="\n",
        )

        # ====================================================
        # RELOAD TEMP FILE FOR FINAL VALIDATION
        # ====================================================

        check_df = pd.read_csv(
            temp_path,
            dtype=str,
            keep_default_na=False,
            low_memory=False,
        )

        # ----------------------------------------------------
        # Same columns
        # ----------------------------------------------------

        if list(check_df.columns) != original_columns:
            raise RuntimeError(
                "Column order changed during save."
            )

        # ----------------------------------------------------
        # Same number of rows
        # ----------------------------------------------------

        if len(check_df) != len(df):
            raise RuntimeError(
                "Row count changed during save."
            )

        # ----------------------------------------------------
        # Verify non-target values after actual CSV write
        # ----------------------------------------------------

        if not check_df[
            non_target_columns
        ].equals(
            original_non_target
        ):
            raise RuntimeError(
                "A non-prize column changed during CSV write."
            )

        # ====================================================
        # ATOMIC REPLACEMENT
        #
        # This is the ONLY point where the existing
        # all_results_cleaned.csv is replaced.
        # ====================================================

        os.replace(
            temp_path,
            RESULTS_FILE,
        )

    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()

        print()
        print(
            f"ERROR saving file: {exc}",
            file=sys.stderr,
        )

        return 1

    # ========================================================
    # FINISHED
    # ========================================================

    file_size_mb = (
        RESULTS_FILE.stat().st_size
        /
        (1024 * 1024)
    )

    print()
    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Updated file: {RESULTS_FILE}"
    )

    print(
        f"File size: {file_size_mb:.2f} MB"
    )

    print()
    print(
        "Only these columns were allowed to change:"
    )

    for column in TARGET_COLUMNS:
        print(
            f"  - {column}"
        )

    print()
    print(
        "No new CSV was created."
    )

    print(
        "all_results_cleaned.csv was updated in place."
    )

    print()

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
