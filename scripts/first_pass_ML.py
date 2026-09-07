from pathlib import Path

import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================

INPUT_FILE = Path("results/races/all_results_date_range.csv")
OUTPUT_FILE = Path("results/races/all_results_with_races_removed.csv")

# Recommended for the first ML model:
# Remove races where more than one horse is recorded as the winner.
# This guarantees exactly one winner per race.
REMOVE_DEAD_HEAT_RACES = True


# =====================================================================
# LOAD DATA
# =====================================================================

print("=" * 70)
print("FIRST PASS ML DATA CLEANING")
print("=" * 70)

print()
print(f"Reading input file:")
print(f"  {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file does not exist: {INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)

original_rows = len(df)
original_races = df["race_id"].nunique()
original_horses = df["horse_id"].nunique()

print()
print(f"Original rows:   {original_rows:,}")
print(f"Original races:  {original_races:,}")
print(f"Original horses: {original_horses:,}")


# =====================================================================
# VALIDATE REQUIRED COLUMNS
# =====================================================================

required_columns = {
    "race_id",
    "race_date",
    "racecourse_code",
    "race_number",
    "horse_id",
    "result_status",
    "is_winner",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "Input file is missing required columns: "
        f"{sorted(missing_columns)}"
    )


# =====================================================================
# CHECK FOR DUPLICATE HORSE-RACE ROWS BEFORE CLEANING
# =====================================================================

duplicate_race_horse_rows = df.duplicated(
    subset=["race_id", "horse_id"]
).sum()

if duplicate_race_horse_rows:
    raise ValueError(
        f"Found {duplicate_race_horse_rows:,} duplicate "
        "race_id + horse_id rows in source data."
    )


# =====================================================================
# NORMALISE RESULT STATUS FOR FILTERING
#
# We do NOT overwrite the original result_status field.
# =====================================================================

status = (
    df["result_status"]
    .astype("string")
    .str.strip()
    .str.upper()
)


# =====================================================================
# STEP 1
# REMOVE ENTIRE RACES CONTAINING:
#
#   DISQUALIFIED
#   NO_RESULT
#
# This removes the WHOLE race, not just the affected horse.
# =====================================================================

bad_statuses = {
    "DISQUALIFIED",
    "NO_RESULT",
}

bad_status_mask = status.isin(bad_statuses)

bad_race_ids = (
    df.loc[bad_status_mask, "race_id"]
    .dropna()
    .unique()
)

rows_before = len(df)
races_before = df["race_id"].nunique()

df = df.loc[
    ~df["race_id"].isin(bad_race_ids)
].copy()

rows_removed_bad_races = rows_before - len(df)
races_removed_bad_races = races_before - df["race_id"].nunique()

print()
print("-" * 70)
print("STEP 1 - REMOVE BAD RACES")
print("-" * 70)

print(
    f"Races containing DISQUALIFIED or NO_RESULT: "
    f"{len(bad_race_ids):,}"
)
print(
    f"Rows removed with those races: "
    f"{rows_removed_bad_races:,}"
)
print(
    f"Races removed: "
    f"{races_removed_bad_races:,}"
)


# =====================================================================
# STEP 2
# REMOVE WITHDRAWN / NON-FINISHED HORSES
#
# After the bad races have been removed, retain only FINISHED rows.
#
# This removes withdrawn horses individually without removing otherwise
# valid races.
# =====================================================================

status = (
    df["result_status"]
    .astype("string")
    .str.strip()
    .str.upper()
)

status_counts_before = status.value_counts(dropna=False)

rows_before = len(df)

df = df.loc[
    status.eq("FINISHED")
].copy()

non_finished_rows_removed = rows_before - len(df)

print()
print("-" * 70)
print("STEP 2 - REMOVE NON-FINISHED ENTRIES")
print("-" * 70)

print("Statuses before filtering:")

for status_name, count in status_counts_before.items():
    print(f"  {status_name}: {count:,}")

print()
print(
    f"Non-FINISHED rows removed: "
    f"{non_finished_rows_removed:,}"
)


# =====================================================================
# STEP 3
# REMOVE WINNER DEAD-HEAT RACES
#
# For the first ML dataset we want exactly ONE winner per race.
#
# We use is_winner to identify these races.
#
# IMPORTANT:
# is_winner is the target and must NOT later be used as an input feature.
# =====================================================================

if REMOVE_DEAD_HEAT_RACES:

    winners_per_race = (
        df.groupby("race_id", sort=False)["is_winner"]
        .sum()
    )

    dead_heat_race_ids = winners_per_race[
        winners_per_race > 1
    ].index

    rows_before = len(df)
    races_before = df["race_id"].nunique()

    df = df.loc[
        ~df["race_id"].isin(dead_heat_race_ids)
    ].copy()

    rows_removed_dead_heats = rows_before - len(df)
    races_removed_dead_heats = (
        races_before - df["race_id"].nunique()
    )

    print()
    print("-" * 70)
    print("STEP 3 - REMOVE WINNER DEAD HEATS")
    print("-" * 70)

    print(
        f"Dead-heat races removed: "
        f"{races_removed_dead_heats:,}"
    )
    print(
        f"Rows removed with dead-heat races: "
        f"{rows_removed_dead_heats:,}"
    )

else:

    print()
    print("-" * 70)
    print("STEP 3 - DEAD HEATS")
    print("-" * 70)
    print("Dead-heat race removal disabled.")


# =====================================================================
# STEP 4
# REMOVE HORSE AGE
#
# We previously decided not to use horse_age_at_race because of its very
# high missingness.
# =====================================================================

if "horse_age_at_race" in df.columns:

    df = df.drop(
        columns=["horse_age_at_race"]
    )

    print()
    print("-" * 70)
    print("STEP 4 - REMOVE HORSE AGE")
    print("-" * 70)
    print("Removed column: horse_age_at_race")

else:

    print()
    print("-" * 70)
    print("STEP 4 - REMOVE HORSE AGE")
    print("-" * 70)
    print("horse_age_at_race was not present.")


# =====================================================================
# STEP 5
# ADD ACTUAL FIELD SIZE
#
# field_size:
#   original declared field size
#
# actual_field_size:
#   number of remaining FINISHED runners in the cleaned race
#
# We retain BOTH.
# =====================================================================

df["actual_field_size"] = (
    df.groupby("race_id", sort=False)["horse_id"]
    .transform("size")
    .astype("int16")
)

print()
print("-" * 70)
print("STEP 5 - ACTUAL FIELD SIZE")
print("-" * 70)

print("Added column: actual_field_size")


# =====================================================================
# STEP 6
# PARSE RACE DATE
# =====================================================================

df["race_date"] = pd.to_datetime(
    df["race_date"],
    errors="raise",
)

print()
print("-" * 70)
print("STEP 6 - DATE VALIDATION")
print("-" * 70)

print(
    "Race date range: "
    f"{df['race_date'].min().date()} "
    "to "
    f"{df['race_date'].max().date()}"
)


# =====================================================================
# STEP 7
# SORT CHRONOLOGICALLY
#
# The original source data is not guaranteed to be physically sorted
# completely chronologically.
#
# Future rolling/history features must always use chronological ordering
# and must exclude the current race.
# =====================================================================

sort_columns = [
    "race_date",
    "racecourse_code",
    "race_number",
    "horse_id",
]

df = (
    df.sort_values(
        by=sort_columns,
        kind="stable",
    )
    .reset_index(drop=True)
)

print()
print("-" * 70)
print("STEP 7 - CHRONOLOGICAL SORT")
print("-" * 70)

print(
    "Sorted by: "
    + ", ".join(sort_columns)
)


# =====================================================================
# STEP 8
# INTEGRITY CHECKS
# =====================================================================

print()
print("-" * 70)
print("STEP 8 - INTEGRITY CHECKS")
print("-" * 70)


# ---------------------------------------------------------------------
# Every remaining row must be FINISHED
# ---------------------------------------------------------------------

final_status = (
    df["result_status"]
    .astype("string")
    .str.strip()
    .str.upper()
)

assert final_status.eq("FINISHED").all(), (
    "ERROR: Non-FINISHED rows remain in cleaned dataset."
)

print("PASS: Every remaining row is FINISHED.")


# ---------------------------------------------------------------------
# race_id + horse_id must be unique
# ---------------------------------------------------------------------

assert not df.duplicated(
    subset=["race_id", "horse_id"]
).any(), (
    "ERROR: Duplicate race_id + horse_id rows detected."
)

print("PASS: race_id + horse_id is unique.")


# ---------------------------------------------------------------------
# result_id should also be unique if it exists
# ---------------------------------------------------------------------

if "result_id" in df.columns:

    assert df["result_id"].is_unique, (
        "ERROR: Duplicate result_id values detected."
    )

    print("PASS: result_id is unique.")


# ---------------------------------------------------------------------
# Every race should contain at least two horses
# ---------------------------------------------------------------------

runners_per_race = (
    df.groupby("race_id", sort=False)
    .size()
)

assert runners_per_race.ge(2).all(), (
    "ERROR: A race with fewer than two runners remains."
)

print("PASS: Every race has at least two runners.")


# ---------------------------------------------------------------------
# actual_field_size must be constant within each race
# ---------------------------------------------------------------------

actual_field_size_unique_counts = (
    df.groupby("race_id", sort=False)[
        "actual_field_size"
    ]
    .nunique()
)

assert actual_field_size_unique_counts.eq(1).all(), (
    "ERROR: actual_field_size is inconsistent within a race."
)

print(
    "PASS: actual_field_size is constant within every race."
)


# ---------------------------------------------------------------------
# actual_field_size must equal the number of rows in the race
#
# IMPORTANT:
# Use .eq(...).all() rather than Series.equals().
#
# Series.equals() also compares dtype, which previously caused the
# workflow to fail because groupby.size() returned int64 while
# actual_field_size was stored as int16.
# ---------------------------------------------------------------------

actual_counts = (
    df.groupby("race_id", sort=False)
    .size()
)

stored_actual_counts = (
    df.groupby("race_id", sort=False)[
        "actual_field_size"
    ]
    .first()
)

assert actual_counts.eq(stored_actual_counts).all(), (
    "ERROR: actual_field_size does not match race row count."
)

print(
    "PASS: actual_field_size matches the number of runners."
)


# ---------------------------------------------------------------------
# Every race must have the correct number of winners
# ---------------------------------------------------------------------

winners_per_race = (
    df.groupby("race_id", sort=False)["is_winner"]
    .sum()
)

if REMOVE_DEAD_HEAT_RACES:

    assert winners_per_race.eq(1).all(), (
        "ERROR: Not every race has exactly one winner."
    )

    print(
        "PASS: Every race has exactly one winner."
    )

else:

    assert winners_per_race.ge(1).all(), (
        "ERROR: At least one race has no winner."
    )

    print(
        "PASS: Every race has at least one winner."
    )


# ---------------------------------------------------------------------
# No bad statuses should remain anywhere
# ---------------------------------------------------------------------

assert not final_status.isin(
    {
        "DISQUALIFIED",
        "NO_RESULT",
        "WITHDRAWN",
    }
).any(), (
    "ERROR: Invalid result statuses remain."
)

print(
    "PASS: No DISQUALIFIED, NO_RESULT or WITHDRAWN rows remain."
)


# =====================================================================
# STEP 9
# INFORMATIONAL MISSING-VALUE CHECKS
#
# These are deliberately NOT treated as errors.
#
# Do not remove rows just because rating information is unavailable.
# These missing values can be structural and CatBoost can handle
# numeric NaN values.
# =====================================================================

print()
print("-" * 70)
print("STEP 9 - IMPORTANT MISSING VALUES")
print("-" * 70)

columns_to_report = [
    "horse_rating_before",
    "rating_band",
    "race_name",
]

for column in columns_to_report:

    if column in df.columns:

        missing_count = df[column].isna().sum()
        missing_pct = (
            missing_count / len(df) * 100
            if len(df)
            else 0
        )

        print(
            f"{column}: "
            f"{missing_count:,} missing "
            f"({missing_pct:.3f}%)"
        )


# =====================================================================
# STEP 10
# CONVERT DATE BACK TO YYYY-MM-DD FOR CSV
# =====================================================================

df["race_date"] = (
    df["race_date"]
    .dt.strftime("%Y-%m-%d")
)


# =====================================================================
# STEP 11
# WRITE OUTPUT CSV
# =====================================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

print()
print("-" * 70)
print("STEP 10 - WRITE OUTPUT")
print("-" * 70)

print(f"Writing:")
print(f"  {OUTPUT_FILE}")

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# =====================================================================
# FINAL SUMMARY
# =====================================================================

final_rows = len(df)
final_races = df["race_id"].nunique()
final_horses = df["horse_id"].nunique()

rows_removed_total = (
    original_rows - final_rows
)

races_removed_total = (
    original_races - final_races
)

print()
print("=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print()
print(f"Input file:")
print(f"  {INPUT_FILE}")

print()
print(f"Output file:")
print(f"  {OUTPUT_FILE}")

print()
print("ROWS")
print(f"  Original: {original_rows:,}")
print(f"  Final:    {final_rows:,}")
print(f"  Removed:  {rows_removed_total:,}")

print()
print("RACES")
print(f"  Original: {original_races:,}")
print(f"  Final:    {final_races:,}")
print(f"  Removed:  {races_removed_total:,}")

print()
print("HORSES")
print(f"  Original: {original_horses:,}")
print(f"  Final:    {final_horses:,}")

print()
print("FINAL RESULT STATUSES")

for value, count in (
    df["result_status"]
    .value_counts(dropna=False)
    .items()
):
    print(f"  {value}: {count:,}")

print()
print("WINNERS PER RACE")

final_winner_counts = (
    df.groupby("race_id")["is_winner"]
    .sum()
    .value_counts()
    .sort_index()
)

for winner_count, race_count in final_winner_counts.items():
    print(
        f"  {winner_count} winner(s): "
        f"{race_count:,} races"
    )

print()
print("ACTUAL FIELD SIZE")

print(
    f"  Minimum: "
    f"{df['actual_field_size'].min()}"
)

print(
    f"  Maximum: "
    f"{df['actual_field_size'].max()}"
)

print()
print("=" * 70)
print("DATASET READY FOR FEATURE ENGINEERING")
print("=" * 70)
