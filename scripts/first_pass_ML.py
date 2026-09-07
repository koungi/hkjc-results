from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path("results/races/all_results_date_range.csv")
OUTPUT_FILE = Path("results/races/all_results_with_races_removed.csv")

# For the first ML version, I recommend removing winner dead-heats so that
# every race has exactly one winner.
REMOVE_DEAD_HEAT_RACES = True


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

print(f"Reading: {INPUT_FILE}")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False,
)

original_rows = len(df)
original_races = df["race_id"].nunique()

print(f"Original rows:  {original_rows:,}")
print(f"Original races: {original_races:,}")


# ---------------------------------------------------------------------
# Basic required-column validation
# ---------------------------------------------------------------------

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
        f"Input file is missing required columns: {sorted(missing_columns)}"
    )


# ---------------------------------------------------------------------
# Standardise status for filtering
#
# We use a temporary helper rather than modifying the source column.
# ---------------------------------------------------------------------

status = (
    df["result_status"]
    .astype("string")
    .str.strip()
    .str.upper()
)


# ---------------------------------------------------------------------
# STEP 1
# Remove ENTIRE races containing a DISQUALIFIED or NO_RESULT row.
#
# NO_RESULT represents races for which there is no valid race result.
# A disqualification can make the finishing/result structure unsuitable
# for the clean first-version ML universe, so the whole race is removed.
# ---------------------------------------------------------------------

bad_status_mask = status.isin(
    {
        "DISQUALIFIED",
        "NO_RESULT",
    }
)

bad_race_ids = df.loc[bad_status_mask, "race_id"].unique()

print()
print(
    f"Races containing DISQUALIFIED or NO_RESULT: "
    f"{len(bad_race_ids):,}"
)

df = df.loc[
    ~df["race_id"].isin(bad_race_ids)
].copy()


# ---------------------------------------------------------------------
# STEP 2
# Remove individual withdrawn/non-starting entries.
#
# After removing bad races, the clean training universe should contain
# only horses with an actual completed result.
# ---------------------------------------------------------------------

status = (
    df["result_status"]
    .astype("string")
    .str.strip()
    .str.upper()
)

withdrawn_rows = status.eq("WITHDRAWN").sum()

print(f"WITHDRAWN rows removed: {withdrawn_rows:,}")

df = df.loc[
    status.eq("FINISHED")
].copy()


# ---------------------------------------------------------------------
# STEP 3
# Remove winner dead-heat races.
#
# We determine this using the TARGET itself rather than trusting only
# the is_dead_heat column:
#
#     number of winners in race > 1
#
# This gives the first ML dataset the useful invariant:
#
#     exactly one winner per race
#
# IMPORTANT:
# is_winner is only being used here to define/validate the training
# universe. It must NOT later be supplied as an input feature.
# ---------------------------------------------------------------------

if REMOVE_DEAD_HEAT_RACES:
    winners_per_race = (
        df.groupby("race_id")["is_winner"]
        .sum()
    )

    dead_heat_race_ids = winners_per_race[
        winners_per_race > 1
    ].index

    print(
        f"Winner dead-heat races removed: "
        f"{len(dead_heat_race_ids):,}"
    )

    df = df.loc[
        ~df["race_id"].isin(dead_heat_race_ids)
    ].copy()


# ---------------------------------------------------------------------
# STEP 4
# Remove horse_age_at_race.
#
# We previously decided not to use this field because of its very high
# missingness. Dropping it here prevents it accidentally entering the
# future feature set.
# ---------------------------------------------------------------------

if "horse_age_at_race" in df.columns:
    df = df.drop(columns=["horse_age_at_race"])


# ---------------------------------------------------------------------
# STEP 5
# Calculate ACTUAL field size.
#
# Original field_size represents the declared field and may include
# horses that subsequently withdrew.
#
# Since withdrawals have now been removed, this gives the number of
# runners that actually completed the race.
#
# Keep BOTH fields:
#
# field_size        = originally declared field size
# actual_field_size = cleaned number of starters/results
#
# The distinction may be useful depending on prediction time.
# ---------------------------------------------------------------------

df["actual_field_size"] = (
    df.groupby("race_id")["horse_id"]
    .transform("count")
    .astype("int16")
)


# ---------------------------------------------------------------------
# STEP 6
# Parse race date.
#
# Keep ISO YYYY-MM-DD output when writing the CSV.
# ---------------------------------------------------------------------

df["race_date"] = pd.to_datetime(
    df["race_date"],
    errors="raise",
)


# ---------------------------------------------------------------------
# STEP 7
# Sort chronologically.
#
# VERY IMPORTANT:
# The original source file was not completely chronological.
#
# Any future rolling/expanding features should be calculated only after
# chronological sorting and should exclude the current race.
# ---------------------------------------------------------------------

sort_columns = [
    "race_date",
    "racecourse_code",
    "race_number",
    "horse_id",
]

df = df.sort_values(
    sort_columns,
    kind="stable",
).reset_index(drop=True)


# ---------------------------------------------------------------------
# STEP 8
# Integrity checks
# ---------------------------------------------------------------------

print()
print("Running integrity checks...")


# Every row should now be FINISHED
assert (
    df["result_status"]
    .astype("string")
    .str.strip()
    .str.upper()
    .eq("FINISHED")
    .all()
), "Non-FINISHED rows remain in cleaned dataset."


# One horse should appear only once per race
assert not df.duplicated(
    subset=["race_id", "horse_id"]
).any(), "Duplicate race_id + horse_id rows detected."


# result_id should remain unique if present
if "result_id" in df.columns:
    assert df["result_id"].is_unique, (
        "Duplicate result_id values detected."
    )


# Every race should contain at least two runners
runners_per_race = df.groupby("race_id").size()

assert (runners_per_race >= 2).all(), (
    "A race with fewer than two runners remains."
)


# actual_field_size must agree with number of remaining rows
field_size_check = (
    df.groupby("race_id")["actual_field_size"]
    .nunique()
)

assert field_size_check.eq(1).all(), (
    "actual_field_size is inconsistent within some races."
)

actual_counts = df.groupby("race_id").size()

stored_actual_counts = (
    df.groupby("race_id")["actual_field_size"]
    .first()
)

assert actual_counts.equals(stored_actual_counts), (
    "actual_field_size does not match race row count."
)


# For the recommended first model, exactly one winner per race
winners_per_race = (
    df.groupby("race_id")["is_winner"]
    .sum()
)

if REMOVE_DEAD_HEAT_RACES:
    assert winners_per_race.eq(1).all(), (
        "Not every race has exactly one winner."
    )
else:
    assert winners_per_race.ge(1).all(), (
        "At least one race has no winner."
    )


# ---------------------------------------------------------------------
# STEP 9
# Important checks for fields we intentionally leave alone
# ---------------------------------------------------------------------

# Do NOT remove rows because horse_rating_before or rating_band is
# missing. Those missing values are often structural (e.g. Griffin or
# international races) and CatBoost can handle numeric NaN values.

print("Integrity checks passed.")


# ---------------------------------------------------------------------
# STEP 10
# Convert date back to YYYY-MM-DD before saving
# ---------------------------------------------------------------------

df["race_date"] = df["race_date"].dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------
# STEP 11
# Save
# ---------------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)

final_rows = len(df)
final_races = df["race_id"].nunique()
final_horses = df["horse_id"].nunique()


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")

print()

print(f"Original rows:   {original_rows:,}")
print(f"Final rows:      {final_rows:,}")
print(f"Rows removed:    {original_rows - final_rows:,}")

print()

print(f"Original races:  {original_races:,}")
print(f"Final races:     {final_races:,}")
print(f"Races removed:   {original_races - final_races:,}")

print()

print(f"Unique horses:   {final_horses:,}")

print()

print("Remaining result statuses:")
print(df["result_status"].value_counts(dropna=False))

print()

print("Winners per race:")
print(
    df.groupby("race_id")["is_winner"]
    .sum()
    .value_counts()
    .sort_index()
)

print()
print("Ready for feature engineering.")
