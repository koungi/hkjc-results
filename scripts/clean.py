from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================
# FILE
#
# This script ONLY reads and overwrites:
#
# results/races/all_results_cleaned.csv
# ======================================================

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT
    / "results"
    / "races"
    / "all_results_cleaned.csv"
)


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

    # ======================================================
    # NORMALISE FINISHING POSITION
    # ======================================================

    df["finishing_position"] = pd.to_numeric(
        df["finishing_position"],
        errors="coerce",
    )

    has_position = df["finishing_position"].notna()

    # ======================================================
    # NORMALISE FINISH TIME
    # ======================================================

    finish_time = (
        df["finish_time"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    no_finish_time = finish_time.isin(
        [
            "",
            "---",
            "-",
            "NAN",
            "NONE",
        ]
    )

    has_real_finish_time = ~no_finish_time

    # ======================================================
    # DETERMINE WHETHER EACH RACE HAS AN OFFICIAL RESULT
    # ======================================================

    race_has_result = (
        df.groupby("race_id")["finishing_position"]
        .transform(lambda x: x.notna().any())
    )

    # ======================================================
    # DETERMINE EXPLICITLY ABANDONED RACES
    #
    # If race_status or race_status_raw exists,
    # inspect those columns for abandonment markers.
    # ======================================================

    is_abandoned = pd.Series(
        False,
        index=df.index,
        dtype=bool,
    )

    for status_column in [
        "race_status",
        "race_status_raw",
    ]:
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
                na=False,
            )

    # ======================================================
    # ASSIGN RUNNER-LEVEL RESULT STATUS
    #
    # Order matters.
    # ======================================================

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

        # 5. Runner has no placing but has a finish time
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
        default="UNKNOWN",
    )

    # ======================================================
    # DERIVED FIELDS
    # ======================================================

    df["actually_started"] = df["result_status"].isin(
        [
            "FINISHED",
            "DISQUALIFIED",
        ]
    )

    df["is_completed"] = (
        df["result_status"] == "FINISHED"
    )

    # Dead heat remains a separate characteristic.
    df["is_dead_heat"] = (
        df["margin"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("DH")
    )

    return df


def validate_results(df: pd.DataFrame) -> None:
    """
    Validate the generated result-status values.
    """

    # FINISHED must always have a position.
    assert not (
        (df["result_status"] == "FINISHED")
        & df["finishing_position"].isna()
    ).any(), (
        "Validation failed: "
        "FINISHED runner found without finishing_position."
    )

    # DISQUALIFIED must not have a numeric position.
    assert not (
        (df["result_status"] == "DISQUALIFIED")
        & df["finishing_position"].notna()
    ).any(), (
        "Validation failed: "
        "DISQUALIFIED runner found with finishing_position."
    )

    # DISQUALIFIED should have a real finish time.
    invalid_disqualified_time = (
        df["finish_time"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(
            [
                "",
                "---",
                "-",
                "NAN",
                "NONE",
            ]
        )
    )

    assert not (
        (df["result_status"] == "DISQUALIFIED")
        & invalid_disqualified_time
    ).any(), (
        "Validation failed: "
        "DISQUALIFIED runner found without a real finish_time."
    )

    # Dead heats should normally be FINISHED.
    dead_heat_non_finished = df[
        df["is_dead_heat"]
        & (df["result_status"] != "FINISHED")
    ]

    if not dead_heat_non_finished.empty:
        print(
            "\nWARNING: Dead-heat rows found "
            "without FINISHED result status:"
        )

        columns = [
            column
            for column in [
                "race_id",
                "horse_name",
                "finishing_position",
                "margin",
                "result_status",
            ]
            if column in dead_heat_non_finished.columns
        ]

        print(
            dead_heat_non_finished[
                columns
            ].to_string(index=False)
        )


def main() -> None:
    # ======================================================
    # VERIFY FILE EXISTS
    # ======================================================

    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Input CSV does not exist: {CSV_PATH}"
        )

    print("Opening:")
    print(CSV_PATH)

    # ======================================================
    # OPEN THE CLEANED RESULTS CSV
    # ======================================================

    df = pd.read_csv(
        CSV_PATH,
        low_memory=False,
    )

    print(f"\nRows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    # ======================================================
    # REQUIRED COLUMNS
    # ======================================================

    required_columns = [
        "race_id",
        "finishing_position",
        "finish_time",
        "margin",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing from "
            "all_results_cleaned.csv: "
            + ", ".join(missing_columns)
        )

    # ======================================================
    # ADD RESULT STATUS
    # ======================================================

    df = add_result_status(df)

    # ======================================================
    # QUICK CHECKS
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
            .value_counts(
                normalize=True,
                dropna=False,
            )
            .mul(100)
            .round(3)
        )
    )

    # ======================================================
    # SHOW UNUSUAL RESULTS
    # ======================================================

    unusual = df[
        df["result_status"].isin(
            [
                "DISQUALIFIED",
                "WITHDRAWN",
                "ABANDONED",
                "NO_RESULT",
                "UNKNOWN",
            ]
        )
    ]

    unusual_columns = [
        column
        for column in [
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
        if column in unusual.columns
    ]

    print("\nUnusual results:")

    if unusual.empty:
        print("None.")
    else:
        print(
            unusual[
                unusual_columns
            ].to_string(index=False)
        )

    # ======================================================
    # VALIDATE RESULTS
    # ======================================================

    validate_results(df)

    print("\nValidation passed.")

    # ======================================================
    # SAVE BACK TO THE EXACT SAME FILE
    # ======================================================

    df.to_csv(
        CSV_PATH,
        index=False,
    )

    print("\nSaved:")
    print(CSV_PATH)

    print(f"Rows saved: {len(df):,}")


if __name__ == "__main__":
    main()
