import numpy as np
import pandas as pd


def add_result_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add runner-level result_status.

    Rules, in priority order:

    1. Explicitly abandoned race
       -> ABANDONED

    2. Entire race has no finishing positions
       -> NO_RESULT

    3. Runner has finishing_position
       -> FINISHED

    4. Runner has blank finishing_position and finish_time is blank / ---
       -> WITHDRAWN

    5. Runner has blank finishing_position and a real finish_time
       -> DISQUALIFIED

    6. Anything else
       -> UNKNOWN
    """

    df = df.copy()

    # --------------------------------------------------
    # Normalise finishing position
    # --------------------------------------------------

    df["finishing_position"] = pd.to_numeric(
        df["finishing_position"],
        errors="coerce"
    )

    has_position = df["finishing_position"].notna()


    # --------------------------------------------------
    # Normalise finish time
    # --------------------------------------------------

    finish_time = (
        df["finish_time"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    no_finish_time = finish_time.isin([
        "",
        "---",
        "-",
        "NAN",
        "NONE",
    ])

    has_real_finish_time = ~no_finish_time


    # --------------------------------------------------
    # Determine whether the race has any official results
    # --------------------------------------------------

    race_has_result = (
        df.groupby("race_id")["finishing_position"]
        .transform(lambda x: x.notna().any())
    )


    # --------------------------------------------------
    # Determine explicitly abandoned races
    #
    # This works if you add a race_status or
    # race_status_raw column to your scraper.
    # --------------------------------------------------

    is_abandoned = pd.Series(
        False,
        index=df.index,
        dtype=bool
    )

    for status_column in ["race_status", "race_status_raw"]:

        if status_column in df.columns:

            status_text = (
                df[status_column]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
            )

            is_abandoned |= status_text.str.contains(
                r"ABANDONED|NOT OFFERED",
                regex=True,
                na=False
            )


    # --------------------------------------------------
    # Assign runner-level result_status
    #
    # IMPORTANT:
    # Order matters.
    # --------------------------------------------------

    conditions = [

        # 1. Race explicitly marked as abandoned
        is_abandoned,

        # 2. Whole race has no finishing positions
        ~race_has_result,

        # 3. Runner has an official placing
        has_position,

        # 4. Runner has no placing and no finish time
        (
            ~has_position
            & no_finish_time
        ),

        # 5. Runner has no placing but does have a finish time
        (
            ~has_position
            & has_real_finish_time
        ),
    ]

    choices = [
        "ABANDONED",
        "NO_RESULT",
        "FINISHED",
        "WITHDRAWN",
        "DISQUALIFIED",
    ]

    df["result_status"] = np.select(
        conditions,
        choices,
        default="UNKNOWN"
    )


    # --------------------------------------------------
    # Optional useful derived fields
    # --------------------------------------------------

    df["actually_started"] = df["result_status"].isin([
        "FINISHED",
        "DISQUALIFIED",
    ])

    df["is_completed"] = (
        df["result_status"] == "FINISHED"
    )

    # Dead heat remains a separate characteristic
    df["is_dead_heat"] = (
        df["margin"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("DH")
    )


    return df


# ======================================================
# Example usage
# ======================================================

# If loading your CSV-style TXT file:
df = pd.read_csv("all_results_date_range.txt")

df = add_result_status(df)


# ======================================================
# Quick checks
# ======================================================

print("\nResult status counts:")
print(
    df["result_status"]
    .value_counts(dropna=False)
)


print("\nResult status percentages:")
print(
    (
        df["result_status"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(3)
    )
)


# Show all unusual runner results
unusual = df[
    df["result_status"].isin([
        "DISQUALIFIED",
        "WITHDRAWN",
        "ABANDONED",
        "NO_RESULT",
        "UNKNOWN",
    ])
][
    [
        "race_id",
        "race_date",
        "racecourse_code",
        "race_number",
        "horse_id",
        "horse_name",
        "finishing_position",
        "finish_time",
        "odds",
        "result_status",
    ]
]

print("\nUnusual results:")
print(unusual.to_string(index=False))


# ======================================================
# Optional validation checks
# ======================================================

# FINISHED must always have a position
assert not (
    (df["result_status"] == "FINISHED")
    & df["finishing_position"].isna()
).any()


# DISQUALIFIED should have no numeric position
assert not (
    (df["result_status"] == "DISQUALIFIED")
    & df["finishing_position"].notna()
).any()


# DISQUALIFIED should have a real finish time
assert not (
    (df["result_status"] == "DISQUALIFIED")
    & (
        df["finish_time"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["", "---", "-", "NAN", "NONE"])
    )
).any()


# Dead heats should still be FINISHED
dead_heat_non_finished = df[
    df["is_dead_heat"]
    & (df["result_status"] != "FINISHED")
]

if len(dead_heat_non_finished) > 0:
    print(
        "\nWARNING: Dead-heat rows found "
        "without FINISHED result status:"
    )

    print(
        dead_heat_non_finished[
            [
                "race_id",
                "horse_name",
                "finishing_position",
                "margin",
                "result_status",
            ]
        ].to_string(index=False)
    )
