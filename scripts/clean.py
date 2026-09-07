#!/usr/bin/env python3

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    REPO_ROOT
    / "results"
    / "races"
    / "all_results_cleaned.csv"
)


# ============================================================
# PAYOUT RULE
#
# Races BEFORE 10 September 2023 use:
#
# 1st = 57%
# 2nd = 22%
# 3rd = 11.5%
# 4th = 6%
# 5th = 3.5%
#
# 10 September 2023 onwards is NOT changed by this script.
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
# THESE ARE THE ONLY FOUR COLUMNS THIS SCRIPT MAY MODIFY
# ============================================================

TARGET_COLUMNS = {
    "prize_payout_percentage",
    "prize_money_won_this_race",
    "career_prize_money_before",
    "career_prize_money_after",
}


REQUIRED_COLUMNS = {
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
}


MONEY_TOLERANCE = 0.01


# ============================================================
# HASH NON-TARGET COLUMNS
#
# This lets us prove that nothing outside the four prize
# columns was changed in memory before saving.
# ============================================================

def hash_columns(df, columns):
    digest = hashlib.sha256()

    for column in columns:
        digest.update(
            column.encode("utf-8")
        )

        hashed = pd.util.hash_pandas_object(
            df[column],
            index=True,
        )

        digest.update(
            hashed.values.tobytes()
        )

    return digest.hexdigest()


# ============================================================
# DEAD-HEAT PAYOUT CALCULATION
#
# This preserves the logic from your original scraper:
#
# If one horse has the placing:
#     use that placing's normal percentage.
#
# If multiple horses share a placing:
#     combine the percentages for the positions occupied
#     by the dead heat, then divide equally.
#
# Examples under the corrected historical schedule:
#
# 2-way dead heat 1st:
#     (57% + 22%) / 2 = 39.5% each
#
# 2-way dead heat 2nd:
#     (22% + 11.5%) / 2 = 16.75% each
#
# 2-way dead heat 4th:
#     (6% + 3.5%) / 2 = 4.75% each
#
# 2-way dead heat 5th:
#     (3.5% + 0%) / 2 = 1.75% each
# ============================================================

def calculate_historical_dead_heat_percentages(df):
    output = pd.Series(
        0.0,
        index=df.index,
        dtype="float64",
    )

    dead_heat_races = set()

    historical = df[
        df["_is_historical"]
    ]

    for race_id, race_group in historical.groupby(
        "race_id",
        sort=False,
        dropna=False,
    ):
        if race_group.empty:
            continue

        valid_finishers = race_group[
            race_group["_finish_numeric"].notna()
        ]

        for finishing_position, position_group in (
            valid_finishers.groupby(
                "_finish_numeric",
                sort=True,
            )
        ):
            position = int(
                finishing_position
            )

            count = len(
                position_group
            )

            if count == 1:
                percentage = (
                    HISTORICAL_PAYOUT_SCHEDULE.get(
                        position,
                        0.0,
                    )
                )

            else:
                dead_heat_races.add(
                    race_id
                )

                combined_percentage = sum(
                    HISTORICAL_PAYOUT_SCHEDULE.get(
                        position + offset,
                        0.0,
                    )
                    for offset in range(count)
                )

                percentage = (
                    combined_percentage
                    / count
                )

            output.loc[
                position_group.index
            ] = percentage

    return output, dead_heat_races


# ============================================================
# CAREER PRIZE ADJUSTMENT
#
# IMPORTANT:
#
# We do NOT rebuild career prize money blindly from zero.
#
# Instead, we calculate the difference introduced by the
# corrected race prize and carry that difference forward
# through the horse's EXISTING continuous career-money series.
#
# If the original data has a reset/discontinuity, for example:
#
# previous career_after = 5,000,000
# next career_before     = 0
#
# then the correction accumulator resets too.
#
# This protects the known 2019 -> 2025 dataset gap and avoids
# inventing a false continuous career history.
# ============================================================

def adjust_career_prize_money(
    df,
    original_prize_won,
    corrected_prize_won,
):
    original_before = pd.to_numeric(
        df["career_prize_money_before"],
        errors="coerce",
    )

    original_after = pd.to_numeric(
        df["career_prize_money_after"],
        errors="coerce",
    )

    corrected_before = original_before.copy()
    corrected_after = original_after.copy()

    sort_columns = [
        "horse_id",
        "_race_date_sort",
        "_race_number_sort",
    ]

    if "_race_index_sort" in df.columns:
        sort_columns.append(
            "_race_index_sort"
        )

    working = df.sort_values(
        sort_columns,
        kind="stable",
    )

    reset_count = 0

    for horse_id, horse_group in working.groupby(
        "horse_id",
        sort=False,
        dropna=False,
    ):
        cumulative_adjustment = 0.0
        previous_original_after = None
        first_row = True

        for index in horse_group.index:
            before = original_before.loc[
                index
            ]

            after = original_after.loc[
                index
            ]

            old_race_prize = original_prize_won.loc[
                index
            ]

            new_race_prize = corrected_prize_won.loc[
                index
            ]

            # ------------------------------------------------
            # Detect a reset / discontinuity in the existing
            # career prize-money sequence.
            # ------------------------------------------------

            if first_row:
                cumulative_adjustment = 0.0

            elif (
                pd.isna(before)
                or
                pd.isna(previous_original_after)
                or
                abs(
                    float(before)
                    -
                    float(previous_original_after)
                )
                >
                MONEY_TOLERANCE
            ):
                cumulative_adjustment = 0.0
                reset_count += 1

            # ------------------------------------------------
            # Add previous historical corrections to the
            # career money BEFORE this race.
            # ------------------------------------------------

            if pd.notna(before):
                corrected_before.loc[
                    index
                ] = round(
                    float(before)
                    +
                    cumulative_adjustment,
                    2,
                )

            # ------------------------------------------------
            # Difference caused by correcting THIS race.
            # ------------------------------------------------

            race_adjustment = round(
                float(new_race_prize)
                -
                float(old_race_prize),
                2,
            )

            # ------------------------------------------------
            # Career after receives:
            #
            # previous accumulated corrections
            # +
            # this race's correction
            # ------------------------------------------------

            if pd.notna(after):
                corrected_after.loc[
                    index
                ] = round(
                    float(after)
                    +
                    cumulative_adjustment
                    +
                    race_adjustment,
                    2,
                )

            cumulative_adjustment = round(
                cumulative_adjustment
                +
                race_adjustment,
                2,
            )

            previous_original_after = after
            first_row = False

    return (
        corrected_before,
        corrected_after,
        reset_count,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not RESULTS_FILE.exists():
        print(
            f"ERROR: File does not exist:\n"
            f"{RESULTS_FILE}",
            file=sys.stderr,
        )

        return 1

    print()
    print("=" * 70)
    print("HISTORICAL PRIZE MONEY CORRECTION")
    print("=" * 70)
    print()
    print(f"File: {RESULTS_FILE}")
    print(
        "Historical cutoff: "
        "race_date < 2023-09-10"
    )
    print()

    # --------------------------------------------------------
    # Read all values as objects/strings.
    #
    # keep_default_na=False preserves blank fields as blanks.
    # --------------------------------------------------------

    df = pd.read_csv(
        RESULTS_FILE,
        dtype=object,
        keep_default_na=False,
    )

    if df.empty:
        print("ERROR: CSV is empty.")
        return 1

    missing_columns = (
        REQUIRED_COLUMNS
        -
        set(df.columns)
    )

    if missing_columns:
        print(
            "ERROR: Missing required columns:",
            sorted(missing_columns),
            file=sys.stderr,
        )

        return 1

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    # --------------------------------------------------------
    # Verify that only our four allowed columns can change.
    # --------------------------------------------------------

    non_target_columns = [
        column
        for column in df.columns
        if column not in TARGET_COLUMNS
    ]

    non_target_hash_before = hash_columns(
        df,
        non_target_columns,
    )

    # --------------------------------------------------------
    # Helper numeric/date columns.
    # These are temporary and never written to the CSV.
    # --------------------------------------------------------

    df["_race_date_sort"] = pd.to_datetime(
        df["race_date"],
        errors="coerce",
    )

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
    ).fillna(0)

    df["_is_historical"] = (
        df["_race_date_sort"].notna()
        &
        (
            df["_race_date_sort"]
            <
            PAYOUT_CUTOFF
        )
    )

    historical_mask = df[
        "_is_historical"
    ]

    historical_rows = int(
        historical_mask.sum()
    )

    historical_races = df.loc[
        historical_mask,
        "race_id",
    ].nunique(
        dropna=False
    )

    print(
        f"Historical rows: {historical_rows:,}"
    )

    print(
        f"Historical races: {historical_races:,}"
    )

    # --------------------------------------------------------
    # Preserve ORIGINAL prize values for calculating deltas.
    # --------------------------------------------------------

    original_payout = pd.to_numeric(
        df["prize_payout_percentage"],
        errors="coerce",
    ).fillna(0.0)

    original_prize_won = pd.to_numeric(
        df["prize_money_won_this_race"],
        errors="coerce",
    ).fillna(0.0)

    # --------------------------------------------------------
    # Calculate corrected historical percentages using
    # the existing dead-heat allocation approach.
    # --------------------------------------------------------

    (
        historical_percentages,
        dead_heat_races,
    ) = calculate_historical_dead_heat_percentages(
        df
    )

    # --------------------------------------------------------
    # Corrected race prize starts as the existing value.
    #
    # Therefore races on/after 2023-09-10 remain untouched.
    # --------------------------------------------------------

    corrected_prize_won = (
        original_prize_won.copy()
    )

    # --------------------------------------------------------
    # Change ONLY historical prize_payout_percentage.
    # --------------------------------------------------------

    df.loc[
        historical_mask,
        "prize_payout_percentage",
    ] = (
        historical_percentages.loc[
            historical_mask
        ]
        .round(8)
        .values
    )

    # --------------------------------------------------------
    # Calculate:
    #
    # race purse * corrected percentage
    #
    # ONLY for historical races.
    # --------------------------------------------------------

    corrected_prize_won.loc[
        historical_mask
    ] = (
        df.loc[
            historical_mask,
            "_race_prize_numeric",
        ]
        *
        historical_percentages.loc[
            historical_mask
        ]
    ).round(2)

    df.loc[
        historical_mask,
        "prize_money_won_this_race",
    ] = (
        corrected_prize_won.loc[
            historical_mask
        ].values
    )

    # --------------------------------------------------------
    # Adjust career prize totals.
    #
    # This carries the correction forward through continuous
    # career-money history without crossing detected resets.
    # --------------------------------------------------------

    (
        corrected_career_before,
        corrected_career_after,
        reset_count,
    ) = adjust_career_prize_money(
        df,
        original_prize_won,
        corrected_prize_won,
    )

    df[
        "career_prize_money_before"
    ] = (
        corrected_career_before
        .round(2)
    )

    df[
        "career_prize_money_after"
    ] = (
        corrected_career_after
        .round(2)
    )

    # --------------------------------------------------------
    # Report changes.
    # --------------------------------------------------------

    new_payout_numeric = pd.to_numeric(
        df["prize_payout_percentage"],
        errors="coerce",
    ).fillna(0.0)

    new_prize_numeric = pd.to_numeric(
        df["prize_money_won_this_race"],
        errors="coerce",
    ).fillna(0.0)

    payout_changed = (
        (
            new_payout_numeric
            -
            original_payout
        )
        .abs()
        >
        0.00000001
    )

    prize_changed = (
        (
            new_prize_numeric
            -
            original_prize_won
        )
        .abs()
        >
        MONEY_TOLERANCE
    )

    print()
    print("Correction summary")
    print("------------------")

    print(
        "Payout percentage rows changed: "
        f"{int(payout_changed.sum()):,}"
    )

    print(
        "Prize-money rows changed:       "
        f"{int(prize_changed.sum()):,}"
    )

    print(
        "Dead-heat races encountered:    "
        f"{len(dead_heat_races):,}"
    )

    print(
        "Career sequence resets found:   "
        f"{reset_count:,}"
    )

    # --------------------------------------------------------
    # Drop temporary helper columns BEFORE validation/save.
    # --------------------------------------------------------

    temporary_columns = [
        "_race_date_sort",
        "_race_number_sort",
        "_race_index_sort",
        "_finish_numeric",
        "_race_prize_numeric",
        "_is_historical",
    ]

    df.drop(
        columns=temporary_columns,
        errors="ignore",
        inplace=True,
    )

    # --------------------------------------------------------
    # CRITICAL VALIDATION
    #
    # Every column other than the four explicitly permitted
    # prize columns must be identical to the values loaded.
    # --------------------------------------------------------

    non_target_hash_after = hash_columns(
        df,
        non_target_columns,
    )

    if (
        non_target_hash_before
        !=
        non_target_hash_after
    ):
        print(
            "ERROR: A non-prize column changed. "
            "File will NOT be saved.",
            file=sys.stderr,
        )

        return 1

    print()
    print(
        "Validation passed: no non-prize "
        "column values changed."
    )

    # --------------------------------------------------------
    # Atomic overwrite.
    #
    # We first write a temporary CSV beside the real file.
    # Only after the complete write succeeds do we replace
    # all_results_cleaned.csv.
    #
    # This prevents a failed Action from leaving a partial CSV.
    # --------------------------------------------------------

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

        os.replace(
            temp_path,
            RESULTS_FILE,
        )

    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        raise

    output_size_mb = (
        RESULTS_FILE.stat().st_size
        /
        (1024 * 1024)
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        f"Saved in place: {RESULTS_FILE}"
    )

    print(
        f"File size: {output_size_mb:.2f} MB"
    )

    print()
    print(
        "Only these columns were permitted to change:"
    )

    for column in sorted(
        TARGET_COLUMNS
    ):
        print(
            f"  - {column}"
        )

    print()

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
