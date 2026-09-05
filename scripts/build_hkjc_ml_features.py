#!/usr/bin/env python3
"""
Build point-in-time-safe ML features for HKJC horse-racing data.

Key rule:
For each race, engineered ML predictors only use information that was
available BEFORE that race.

Current-race results are retained as targets, but they are only used to
create features for future races.

Example:
python scripts/build_hkjc_ml_features.py \
    --input results/races/all_results.csv \
    --output results/races/all_results_ML.csv

Dependencies:
    pandas
    numpy
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "race_id",
    "race_date",
    "racecourse_code",
    "race_number",
    "race_index",
    "distance_m",
    "field_size",
    "horse_id",
    "horse_name",
    "horse_rating_before",
    "finishing_position",
    "is_winner",
    "is_top_three",
    "jockey",
    "trainer",
    "actual_weight",
    "declared_horse_weight",
    "draw",
    "margin",
    "finish_time",
    "odds",
    "race_class",
    "going",
    "surface",
    "course",
]


CURRENT_RACE_OUTCOME_COLUMNS = [
    "finishing_position",
    "is_winner",
    "is_top_three",
    "margin",
    "finish_time",
    "prize_payout_percentage",
    "prize_money_won_this_race",
    "career_prize_money_after",
    "horse_rating_after",
]


INTERNAL_COLUMNS = [
    "_row_id",
    "_event_order",
    "_started",
    "_finish_num",
    "_relative_finish",
    "_margin_lengths",
    "_finish_time_seconds",
    "_speed_mps",
    "_race_class_num",
    "_winner_time_seconds",
    "_time_behind_winner_seconds",
    "_speed_context_z",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create leakage-safe HKJC ML features."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV path",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output enriched CSV path",
    )

    parser.add_argument(
        "--include-market-features",
        action="store_true",
        help=(
            "Include same-race odds / market features. "
            "Only use when live odds are available at prediction time."
        ),
    )

    parser.add_argument(
        "--keep-helper-columns",
        action="store_true",
        help="Keep internal helper columns.",
    )

    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON feature manifest path.",
    )

    return parser.parse_args()


def require_columns(df, required):
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Input is missing required columns: "
            + ", ".join(missing)
        )


def to_num(df, cols: Iterable[str]):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )


def parse_margin_lengths(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    if not value:
        return np.nan

    if value in {
        "---",
        "ML",
        "TO",
    }:
        return np.nan

    approximations = {
        "DH": 0.0,
        "N": 0.05,
        "NOSE": 0.05,
        "+NOSE": 0.05,
        "SH": 0.10,
        "+SH": 0.10,
        "HD": 0.20,
    }

    if value in approximations:
        return approximations[value]

    if value.startswith("+"):
        value = value[1:]

    try:

        if "-" in value:
            whole, fraction = value.split("-", 1)
            numerator, denominator = fraction.split("/", 1)

            return (
                float(whole)
                + float(numerator) / float(denominator)
            )

        if "/" in value:
            numerator, denominator = value.split("/", 1)

            return (
                float(numerator)
                / float(denominator)
            )

        return float(value)

    except (ValueError, ZeroDivisionError):
        return np.nan


def parse_finish_time_seconds(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if not value or value == "---":
        return np.nan

    match = re.fullmatch(
        r"(?:(\d+):)?(\d{1,2}(?:\.\d+)?)",
        value,
    )

    if not match:
        return np.nan

    minutes = int(match.group(1) or 0)
    seconds = float(match.group(2))

    return minutes * 60.0 + seconds


def parse_class_number(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if re.fullmatch(r"[1-5]", value):
        return float(value)

    return np.nan


def add_base_helpers(df):

    df = df.copy()

    df["race_date"] = pd.to_datetime(
        df["race_date"],
        errors="coerce",
    )

    if df["race_date"].isna().any():

        bad = int(
            df["race_date"].isna().sum()
        )

        raise ValueError(
            f"{bad} rows have invalid race_date"
        )

    to_num(
        df,
        [
            "race_number",
            "race_index",
            "distance_m",
            "field_size",
            "horse_rating_before",
            "horse_rating_after",
            "finishing_position",
            "is_winner",
            "is_top_three",
            "actual_weight",
            "declared_horse_weight",
            "draw",
            "odds",
            "career_starts_before",
            "career_wins_before",
            "career_seconds_before",
            "career_thirds_before",
            "career_top3_before",
            "career_win_rate_before",
            "career_top3_rate_before",
            "career_prize_money_before",
            "career_prize_money_after",
            "prize_money_hkd",
            "prize_money_won_this_race",
        ],
    )

    df["_row_id"] = np.arange(
        len(df),
        dtype=np.int64,
    )

    sort_columns = [
        "race_date",
        "race_index",
        "race_number",
        "race_id",
    ]

    if "horse_number" in df.columns:
        sort_columns.append(
            "horse_number"
        )

    sort_columns.append(
        "_row_id"
    )

    df = df.sort_values(
        sort_columns,
        kind="mergesort",
        na_position="last",
    ).reset_index(
        drop=True
    )

    df["_event_order"] = np.arange(
        len(df),
        dtype=np.int64,
    )

    df["_finish_num"] = pd.to_numeric(
        df["finishing_position"],
        errors="coerce",
    )

    df["_started"] = (
        df["_finish_num"]
        .notna()
        .astype(np.int8)
    )

    denominator = (
        pd.to_numeric(
            df["field_size"],
            errors="coerce",
        )
        - 1.0
    )

    df["_relative_finish"] = np.where(
        df["_finish_num"].notna()
        & (denominator > 0),

        (
            df["_finish_num"] - 1.0
        ) / denominator,

        np.nan,
    )

    df["_margin_lengths"] = (
        df["margin"]
        .map(parse_margin_lengths)
    )

    df.loc[
        (df["_finish_num"] == 1)
        & df["_margin_lengths"].isna(),

        "_margin_lengths",

    ] = 0.0

    df["_finish_time_seconds"] = (
        df["finish_time"]
        .map(parse_finish_time_seconds)
    )

    df["_speed_mps"] = np.where(

        df["_finish_time_seconds"] > 0,

        pd.to_numeric(
            df["distance_m"],
            errors="coerce",
        )
        / df["_finish_time_seconds"],

        np.nan,
    )

    df["_race_class_num"] = (
        df["race_class"]
        .map(parse_class_number)
    )

    df["distance_bucket"] = pd.cut(

        pd.to_numeric(
            df["distance_m"],
            errors="coerce",
        ),

        bins=[
            -np.inf,
            1200,
            1400,
            1650,
            2000,
            np.inf,
        ],

        labels=[
            "sprint_1200_or_less",
            "1400",
            "mile_1600_1650",
            "middle_1800_2000",
            "staying_over_2000",
        ],

    ).astype("string")

    going = (
        df["going"]
        .astype("string")
        .str.upper()
    )

    conditions = [
        going.str.contains(
            "FIRM",
            na=False,
        ),

        going.eq("GOOD").fillna(False),

        going.str.contains(
            "YIELD",
            na=False,
        ),

        going.str.contains(
            "SOFT",
            na=False,
        ),

        going.str.contains(
            "WET",
            na=False,
        ),

        going.str.contains(
            "FAST",
            na=False,
        ),
    ]

    conditions = [
        c.to_numpy(
            dtype=bool,
            na_value=False,
        )
        for c in conditions
    ]

    df["going_bucket"] = np.select(

        conditions,

        [
            "firm",
            "good",
            "yielding",
            "soft",
            "wet",
            "fast",
        ],

        default="other",
    )

    return df


def add_race_outcome_helpers(df):

    winner_times = (

        df.loc[
            df["_finish_num"] == 1
        ]

        .groupby(
            "race_id",
            dropna=False,
        )["_finish_time_seconds"]

        .min()
    )

    df["_winner_time_seconds"] = (
        df["race_id"]
        .map(winner_times)
    )

    df["_time_behind_winner_seconds"] = (
        np.where(

            df["_finish_time_seconds"].notna()
            & df["_winner_time_seconds"].notna(),

            df["_finish_time_seconds"]
            - df["_winner_time_seconds"],

            np.nan,
        )
    )

    return df


def add_context_speed_z(
    df,
    min_prior_observations=20,
):

    context = [
        "racecourse_code",
        "surface",
        "course",
        "distance_m",
        "going",
    ]

    valid = (
        df["_speed_mps"]
        .notna()
    )

    temp = df.loc[
        valid,
        [
            "race_id",
            "race_date",
            "_event_order",
            *context,
            "_speed_mps",
        ],
    ].copy()

    if temp.empty:

        df["_speed_context_z"] = np.nan

        return df

    temp["_sq"] = (
        temp["_speed_mps"] ** 2
    )

    race_stats = (

        temp.groupby(
            "race_id",
            as_index=False,
            dropna=False,
        )

        .agg(
            race_date=(
                "race_date",
                "first",
            ),

            event_order=(
                "_event_order",
                "min",
            ),

            speed_sum=(
                "_speed_mps",
                "sum",
            ),

            speed_sumsq=(
                "_sq",
                "sum",
            ),

            speed_count=(
                "_speed_mps",
                "count",
            ),

            **{
                f"ctx_{c}": (
                    c,
                    "first",
                )
                for c in context
            },
        )

        .sort_values(
            [
                "race_date",
                "event_order",
            ],
            kind="mergesort",
        )
    )

    context_columns = [
        f"ctx_{c}"
        for c in context
    ]

    grouped = race_stats.groupby(
        context_columns,
        dropna=False,
        sort=False,
    )

    race_stats["prior_sum"] = (
        grouped["speed_sum"].cumsum()
        - race_stats["speed_sum"]
    )

    race_stats["prior_sumsq"] = (
        grouped["speed_sumsq"].cumsum()
        - race_stats["speed_sumsq"]
    )

    race_stats["prior_count"] = (
        grouped["speed_count"].cumsum()
        - race_stats["speed_count"]
    )

    race_stats["prior_mean"] = (

        race_stats["prior_sum"]

        / race_stats[
            "prior_count"
        ].replace(
            0,
            np.nan,
        )
    )

    variance = (

        race_stats["prior_sumsq"]

        / race_stats[
            "prior_count"
        ].replace(
            0,
            np.nan,
        )

        - race_stats[
            "prior_mean"
        ] ** 2
    )

    race_stats["prior_std"] = np.sqrt(
        variance.clip(
            lower=0
        )
    )

    benchmark = (

        race_stats

        .set_index(
            "race_id"
        )

        [
            [
                "prior_count",
                "prior_mean",
                "prior_std",
            ]
        ]
    )

    df = df.join(
        benchmark,
        on="race_id",
    )

    valid_benchmark = (

        (
            df["prior_count"]
            >= min_prior_observations
        )

        & (
            df["prior_std"]
            > 0
        )
    )

    df["_speed_context_z"] = np.where(

        valid_benchmark
        & df["_speed_mps"].notna(),

        (
            df["_speed_mps"]
            - df["prior_mean"]
        )
        / df["prior_std"],

        np.nan,
    )

    return df.drop(
        columns=[
            "prior_count",
            "prior_mean",
            "prior_std",
        ]
    )


def add_horse_elo(
    df,
    base=1500.0,
    k=16.0,
):

    ratings = {}

    elo_before = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    for _, race in df.groupby(
        "race_id",
        sort=False,
    ):

        indices = race.index.to_numpy()

        horses = (
            race["horse_id"]
            .astype(str)
            .to_numpy()
        )

        pre_ratings = np.array(
            [
                ratings.get(
                    horse,
                    base,
                )
                for horse in horses
            ],
            dtype=float,
        )

        elo_before[
            indices
        ] = pre_ratings

        finish = (
            race["_finish_num"]
            .to_numpy(
                dtype=float
            )
        )

        valid_positions = np.where(
            ~np.isnan(finish)
        )[0]

        if len(
            valid_positions
        ) < 2:
            continue

        delta = np.zeros(
            len(race),
            dtype=float,
        )

        for a in range(
            len(valid_positions)
        ):

            i = valid_positions[a]

            for b in range(
                a + 1,
                len(valid_positions),
            ):

                j = valid_positions[b]

                rating_i = (
                    pre_ratings[i]
                )

                rating_j = (
                    pre_ratings[j]
                )

                expected_i = (
                    1.0
                    /
                    (
                        1.0
                        + 10.0
                        ** (
                            (
                                rating_j
                                - rating_i
                            )
                            / 400.0
                        )
                    )
                )

                if finish[i] < finish[j]:
                    actual_i = 1.0

                elif finish[i] > finish[j]:
                    actual_i = 0.0

                else:
                    actual_i = 0.5

                difference = (
                    k
                    * (
                        actual_i
                        - expected_i
                    )
                )

                delta[i] += difference
                delta[j] -= difference

        opponents = max(
            len(valid_positions) - 1,
            1,
        )

        for local_index in valid_positions:

            horse = horses[
                local_index
            ]

            ratings[horse] = (

                pre_ratings[
                    local_index
                ]

                + delta[
                    local_index
                ]
                / opponents
            )

    df["horse_elo_before"] = (
        elo_before
    )

    return df


def add_basic_field_features(df):

    grouped = df.groupby(
        "race_id",
        dropna=False,
    )

    rating = pd.to_numeric(
        df["horse_rating_before"],
        errors="coerce",
    )

    df["field_avg_rating"] = (
        grouped[
            "horse_rating_before"
        ].transform("mean")
    )

    df["field_max_rating"] = (
        grouped[
            "horse_rating_before"
        ].transform("max")
    )

    df["field_min_rating"] = (
        grouped[
            "horse_rating_before"
        ].transform("min")
    )

    df["field_rating_std"] = (
        grouped[
            "horse_rating_before"
        ].transform("std")
    )

    df["rating_vs_field_mean"] = (
        rating
        - df["field_avg_rating"]
    )

    df["rating_vs_field_max"] = (
        rating
        - df["field_max_rating"]
    )

    df["rating_rank_in_field"] = (
        grouped[
            "horse_rating_before"
        ].rank(
            method="min",
            ascending=False,
        )
    )

    df["rating_percentile_in_field"] = (
        grouped[
            "horse_rating_before"
        ].rank(
            pct=True,
            ascending=True,
        )
    )

    field_size = pd.to_numeric(
        df["field_size"],
        errors="coerce",
    )

    draw = pd.to_numeric(
        df["draw"],
        errors="coerce",
    )

    df["draw_normalized"] = np.where(

        field_size > 0,

        draw / field_size,

        np.nan,
    )

    df["draw_percentile"] = np.where(

        field_size > 1,

        (
            draw - 1.0
        )
        / (
            field_size - 1.0
        ),

        np.nan,
    )

    df["inside_draw_flag"] = np.where(

        df["draw_percentile"].notna(),

        (
            df["draw_percentile"]
            <= 0.33
        ).astype(int),

        np.nan,
    )

    df["outside_draw_flag"] = np.where(

        df["draw_percentile"].notna(),

        (
            df["draw_percentile"]
            >= 0.67
        ).astype(int),

        np.nan,
    )

    df["horse_elo_rank_in_field"] = (

        grouped[
            "horse_elo_before"
        ]

        .rank(
            method="min",
            ascending=False,
        )
    )

    df["horse_elo_vs_field_mean"] = (

        df["horse_elo_before"]

        - grouped[
            "horse_elo_before"
        ].transform(
            "mean"
        )
    )

    return df


def rowwise_slope(values):

    y = values.to_numpy(
        dtype=float
    )

    x = np.arange(
        y.shape[1],
        dtype=float,
    )[None, :]

    mask = ~np.isnan(y)

    n = mask.sum(
        axis=1
    ).astype(float)

    sx = (
        mask * x
    ).sum(
        axis=1
    )

    sy = np.nansum(
        y,
        axis=1,
    )

    sxx = (
        mask
        * (
            x ** 2
        )
    ).sum(
        axis=1
    )

    sxy = np.nansum(

        np.where(
            mask,
            y * x,
            np.nan,
        ),

        axis=1,
    )

    denominator = (
        n * sxx
        - sx * sx
    )

    output = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    valid = (
        (n >= 2)
        & (denominator != 0)
    )

    output[valid] = (

        (
            n[valid]
            * sxy[valid]
            - sx[valid]
            * sy[valid]
        )

        / denominator[valid]
    )

    return output


def add_horse_history(df):

    starts = df.loc[
        df["_started"] == 1
    ].copy()

    if starts.empty:
        return df

    history_sources = {

        "date":
            "race_date",

        "race_id":
            "race_id",

        "finish":
            "_finish_num",

        "relative_finish":
            "_relative_finish",

        "margin_lengths":
            "_margin_lengths",

        "finish_time_seconds":
            "_finish_time_seconds",

        "speed_mps":
            "_speed_mps",

        "speed_context_z":
            "_speed_context_z",

        "time_behind_winner_seconds":
            "_time_behind_winner_seconds",

        "distance_m":
            "distance_m",

        "race_class":
            "race_class",

        "race_class_num":
            "_race_class_num",

        "horse_rating_before":
            "horse_rating_before",

        "horse_rating_after":
            "horse_rating_after",

        "actual_weight":
            "actual_weight",

        "declared_horse_weight":
            "declared_horse_weight",

        "draw":
            "draw",

        "odds":
            "odds",

        "track":
            "racecourse_code",

        "surface":
            "surface",

        "course":
            "course",

        "going":
            "going",

        "jockey":
            "jockey",

        "trainer":
            "trainer",

        "field_size":
            "field_size",

        "field_avg_rating":
            "field_avg_rating",

        "field_max_rating":
            "field_max_rating",

        "rating_vs_field_mean":
            "rating_vs_field_mean",
    }

    source_columns = list(
        history_sources.values()
    )

    name_for_source = {
        value: key
        for key, value
        in history_sources.items()
    }

    grouped = starts.groupby(
        "horse_id",
        sort=False,
    )[source_columns]

    lag_frames = []

    for lag in range(5):

        if lag == 0:

            block = (
                starts[
                    source_columns
                ].copy()
            )

        else:

            block = (
                grouped.shift(
                    lag
                )
            )

        block = block.rename(
            columns={
                source:
                    f"last{lag + 1}_{name_for_source[source]}"
                for source in source_columns
            }
        )

        lag_frames.append(
            block
        )

    history = pd.concat(
        lag_frames,
        axis=1,
    )

    summary = pd.DataFrame(
        index=starts.index
    )

    for n in (
        3,
        5,
    ):

        finish_columns = [
            f"last{i}_finish"
            for i in range(
                1,
                n + 1,
            )
        ]

        relative_columns = [
            f"last{i}_relative_finish"
            for i in range(
                1,
                n + 1,
            )
        ]

        margin_columns = [
            f"last{i}_margin_lengths"
            for i in range(
                1,
                n + 1,
            )
        ]

        speed_columns = [
            f"last{i}_speed_mps"
            for i in range(
                1,
                n + 1,
            )
        ]

        speed_z_columns = [
            f"last{i}_speed_context_z"
            for i in range(
                1,
                n + 1,
            )
        ]

        field_columns = [
            f"last{i}_field_avg_rating"
            for i in range(
                1,
                n + 1,
            )
        ]

        finish_values = (
            history[
                finish_columns
            ].astype(float)
        )

        summary[
            f"avg_finish_last{n}"
        ] = finish_values.mean(
            axis=1
        )

        summary[
            f"median_finish_last{n}"
        ] = finish_values.median(
            axis=1
        )

        summary[
            f"best_finish_last{n}"
        ] = finish_values.min(
            axis=1
        )

        summary[
            f"worst_finish_last{n}"
        ] = finish_values.max(
            axis=1
        )

        summary[
            f"finish_std_last{n}"
        ] = finish_values.std(
            axis=1,
            ddof=1,
        )

        summary[
            f"avg_relative_finish_last{n}"
        ] = (

            history[
                relative_columns
            ]

            .astype(float)

            .mean(
                axis=1
            )
        )

        summary[
            f"avg_margin_lengths_last{n}"
        ] = (

            history[
                margin_columns
            ]

            .astype(float)

            .mean(
                axis=1
            )
        )

        speed_values = (

            history[
                speed_columns
            ]

            .astype(float)
        )

        summary[
            f"avg_speed_mps_last{n}"
        ] = speed_values.mean(
            axis=1
        )

        summary[
            f"best_speed_mps_last{n}"
        ] = speed_values.max(
            axis=1
        )

        summary[
            f"speed_std_last{n}"
        ] = speed_values.std(
            axis=1,
            ddof=1,
        )

        summary[
            f"avg_speed_context_z_last{n}"
        ] = (

            history[
                speed_z_columns
            ]

            .astype(float)

            .mean(
                axis=1
            )
        )

        summary[
            f"avg_field_rating_last{n}"
        ] = (

            history[
                field_columns
            ]

            .astype(float)

            .mean(
                axis=1
            )
        )

    finish10 = history[
        [
            f"last{i}_finish"
            for i in range(
                1,
                6,
            )
        ]
    ].copy()

    finish_group = starts.groupby(
        "horse_id",
        sort=False,
    )["_finish_num"]

    for lag in range(
        5,
        10,
    ):

        finish10[
            f"last{lag + 1}_finish"
        ] = finish_group.shift(
            lag
        )

    for n in (
        3,
        5,
        10,
    ):

        values = (

            finish10[
                [
                    f"last{i}_finish"
                    for i in range(
                        1,
                        n + 1,
                    )
                ]
            ]

            .astype(float)
        )

        wins = (
            values.eq(1)
            .sum(axis=1)
        )

        top3 = (
            values.le(3)
            .sum(axis=1)
        )

        count = (
            values.notna()
            .sum(axis=1)
            .replace(
                0,
                np.nan,
            )
        )

        summary[
            f"wins_last{n}"
        ] = wins.astype(float)

        summary[
            f"top3_last{n}"
        ] = top3.astype(float)

        summary[
            f"win_rate_last{n}"
        ] = (
            wins / count
        )

        summary[
            f"top3_rate_last{n}"
        ] = (
            top3 / count
        )

    summary[
        "prior_completed_starts_derived"
    ] = (

        starts.groupby(
            "horse_id",
            sort=False,
        )

        .cumcount()

        + 1
    )

    summary[
        "finish_trend_last5"
    ] = rowwise_slope(

        history[
            [
                f"last{i}_finish"
                for i in range(
                    5,
                    0,
                    -1,
                )
            ]
        ].astype(float)
    )

    summary[
        "relative_finish_trend_last5"
    ] = rowwise_slope(

        history[
            [
                f"last{i}_relative_finish"
                for i in range(
                    5,
                    0,
                    -1,
                )
            ]
        ].astype(float)
    )

    summary[
        "speed_trend_last5"
    ] = rowwise_slope(

        history[
            [
                f"last{i}_speed_mps"
                for i in range(
                    5,
                    0,
                    -1,
                )
            ]
        ].astype(float)
    )

    starts_features = pd.concat(
        [
            starts[
                [
                    "horse_id",
                    "_event_order",
                ]
            ],
            history,
            summary,
        ],
        axis=1,
    )

    carry_columns = [
        c
        for c in starts_features.columns
        if c not in {
            "horse_id",
            "_event_order",
        }
    ]

    right = starts_features.rename(
        columns={
            "_event_order":
                "_prior_event_order"
        }
    )

    left = df[
        [
            "_row_id",
            "horse_id",
            "_event_order",
        ]
    ].copy()

    left = left.sort_values(
        [
            "_event_order",
            "horse_id",
        ],
        kind="mergesort",
    )

    right = right.sort_values(
        [
            "_prior_event_order",
            "horse_id",
        ],
        kind="mergesort",
    )

    merged = pd.merge_asof(

        left,

        right,

        left_on="_event_order",

        right_on="_prior_event_order",

        by="horse_id",

        direction="backward",

        allow_exact_matches=False,

    ).set_index(
        "_row_id"
    )

    carry = (
        merged[
            carry_columns
        ]

        .reindex(
            df["_row_id"]
            .to_numpy()
        )

        .copy()
    )

    carry.index = df.index

    df = pd.concat(
        [
            df,
            carry,
        ],
        axis=1,
    )

    df["days_since_last_start"] = (

        df["race_date"]

        - pd.to_datetime(
            df["last1_date"],
            errors="coerce",
        )
    ).dt.days

    start_dates_by_horse = {

        horse:
            group[
                "race_date"
            ].to_numpy(
                dtype="datetime64[ns]"
            )

        for horse, group
        in starts.groupby(
            "horse_id",
            sort=False,
        )
    }

    target_groups = (
        df.groupby(
            "horse_id",
            sort=False,
        )
        .groups
    )

    workload = {}

    for window in (
        30,
        90,
        365,
    ):

        output = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        for horse, target_indices in target_groups.items():

            target_indices = np.asarray(
                list(target_indices),
                dtype=int,
            )

            target_dates = (

                df.loc[
                    target_indices,
                    "race_date",
                ]

                .to_numpy(
                    dtype="datetime64[ns]"
                )
            )

            start_dates = (
                start_dates_by_horse
                .get(horse)
            )

            if (
                start_dates is None
                or len(start_dates) == 0
            ):

                output[
                    target_indices
                ] = 0

                continue

            end = np.searchsorted(
                start_dates,
                target_dates,
                side="left",
            )

            begin = np.searchsorted(

                start_dates,

                target_dates
                - np.timedelta64(
                    window,
                    "D",
                ),

                side="left",
            )

            output[
                target_indices
            ] = (
                end - begin
            )

        workload[
            f"starts_last{window}d"
        ] = output

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                workload,
                index=df.index,
            ),
        ],
        axis=1,
    )

    changes = pd.DataFrame(
        index=df.index
    )

    changes[
        "distance_change_from_last"
    ] = (

        pd.to_numeric(
            df["distance_m"],
            errors="coerce",
        )

        - pd.to_numeric(
            df["last1_distance_m"],
            errors="coerce",
        )
    )

    changes[
        "rating_change_from_last_start_pre"
    ] = (

        pd.to_numeric(
            df["horse_rating_before"],
            errors="coerce",
        )

        - pd.to_numeric(
            df["last1_horse_rating_before"],
            errors="coerce",
        )
    )

    changes[
        "rating_change_since_last_official"
    ] = (

        pd.to_numeric(
            df["horse_rating_before"],
            errors="coerce",
        )

        - pd.to_numeric(
            df["last1_horse_rating_after"],
            errors="coerce",
        )
    )

    changes[
        "actual_weight_change_from_last"
    ] = (

        pd.to_numeric(
            df["actual_weight"],
            errors="coerce",
        )

        - pd.to_numeric(
            df["last1_actual_weight"],
            errors="coerce",
        )
    )

    changes[
        "bodyweight_change_from_last"
    ] = (

        pd.to_numeric(
            df["declared_horse_weight"],
            errors="coerce",
        )

        - pd.to_numeric(
            df["last1_declared_horse_weight"],
            errors="coerce",
        )
    )

    changes[
        "bodyweight_change_pct_from_last"
    ] = (

        changes[
            "bodyweight_change_from_last"
        ]

        / pd.to_numeric(
            df["last1_declared_horse_weight"],
            errors="coerce",
        ).replace(
            0,
            np.nan,
        )
    )

    changes[
        "class_number_change_from_last"
    ] = (

        df["_race_class_num"]

        - pd.to_numeric(
            df["last1_race_class_num"],
            errors="coerce",
        )
    )

    changes[
        "class_drop_flag"
    ] = np.where(

        changes[
            "class_number_change_from_last"
        ].notna(),

        (
            changes[
                "class_number_change_from_last"
            ]
            > 0
        ).astype(int),

        np.nan,
    )

    changes[
        "class_rise_flag"
    ] = np.where(

        changes[
            "class_number_change_from_last"
        ].notna(),

        (
            changes[
                "class_number_change_from_last"
            ]
            < 0
        ).astype(int),

        np.nan,
    )

    comparisons = [

        (
            "racecourse_code",
            "last1_track",
            "track_change_flag",
        ),

        (
            "surface",
            "last1_surface",
            "surface_change_flag",
        ),

        (
            "course",
            "last1_course",
            "course_change_flag",
        ),

        (
            "going",
            "last1_going",
            "going_change_flag",
        ),

        (
            "jockey",
            "last1_jockey",
            "jockey_change_flag",
        ),

        (
            "trainer",
            "last1_trainer",
            "trainer_change_flag",
        ),
    ]

    for current, prior, name in comparisons:

        both = (
            df[current].notna()
            & df[prior].notna()
        )

        changes[name] = np.where(

            both,

            (
                df[current].astype(str)
                != df[prior].astype(str)
            ).astype(int),

            np.nan,
        )

    df = pd.concat(
        [
            df,
            changes,
        ],
        axis=1,
    )

    def format_finish(value):

        if pd.isna(value):
            return ""

        value = float(value)

        if value.is_integer():
            return str(
                int(value)
            )

        return str(value)

    finish_columns = [
        f"last{i}_finish"
        for i in range(
            1,
            6,
        )
    ]

    df[
        "form_last5_latest_first"
    ] = (

        df[
            finish_columns
        ]

        .apply(
            lambda row:
                "-".join(
                    [
                        x
                        for x in (
                            format_finish(value)
                            for value in row
                        )
                        if x
                    ]
                ),
            axis=1,
        )
    )

    return df


def add_horse_context_stats(df):

    contexts = {

        "distance":
            ["distance_m"],

        "track":
            ["racecourse_code"],

        "track_distance":
            [
                "racecourse_code",
                "distance_m",
            ],

        "surface":
            ["surface"],

        "going":
            ["going"],

        "class":
            ["race_class"],
    }

    wins = (

        (
            df["_finish_num"] == 1
        )

        & (
            df["_started"] == 1
        )
    ).astype(float)

    top3 = (

        (
            df["_finish_num"] <= 3
        )

        & (
            df["_started"] == 1
        )
    ).astype(float)

    for label, context_columns in contexts.items():

        keys = [
            "horse_id",
            *context_columns,
        ]

        grouped = df.groupby(
            keys,
            dropna=False,
            sort=False,
        )

        prior_starts = (

            grouped[
                "_started"
            ].cumsum()

            - df["_started"]
        )

        cumulative_wins = (

            wins.groupby(
                [
                    df[k]
                    for k in keys
                ],
                dropna=False,
            )

            .cumsum()

            - wins
        )

        cumulative_top3 = (

            top3.groupby(
                [
                    df[k]
                    for k in keys
                ],
                dropna=False,
            )

            .cumsum()

            - top3
        )

        relative_valid = (
            df["_relative_finish"]
            .notna()
            .astype(float)
        )

        relative_values = (
            df["_relative_finish"]
            .fillna(0.0)
        )

        relative_sum = (

            relative_values.groupby(
                [
                    df[k]
                    for k in keys
                ],
                dropna=False,
            )

            .cumsum()

            - relative_values
        )

        relative_count = (

            relative_valid.groupby(
                [
                    df[k]
                    for k in keys
                ],
                dropna=False,
            )

            .cumsum()

            - relative_valid
        )

        speed_valid = (
            df["_speed_mps"]
            .notna()
            .astype(float)
        )

        speed_values = (
            df["_speed_mps"]
            .fillna(0.0)
        )

        speed_sum = (

            speed_values.groupby(
                [
                    df[k]
                    for k in keys
                ],
                dropna=False,
            )

            .cumsum()

            - speed_values
        )

        speed_count = (

            speed_valid.groupby(
                [
                    df[k]
                    for k in keys
                ],
                dropna=False,
            )

            .cumsum()

            - speed_valid
        )

        df[
            f"horse_{label}_starts_before"
        ] = prior_starts.astype(float)

        df[
            f"horse_{label}_wins_before"
        ] = cumulative_wins.astype(float)

        df[
            f"horse_{label}_top3_before"
        ] = cumulative_top3.astype(float)

        df[
            f"horse_{label}_win_rate_before"
        ] = (

            cumulative_wins

            / prior_starts.replace(
                0,
                np.nan,
            )
        )

        df[
            f"horse_{label}_top3_rate_before"
        ] = (

            cumulative_top3

            / prior_starts.replace(
                0,
                np.nan,
            )
        )

        df[
            f"horse_{label}_avg_relative_finish_before"
        ] = (

            relative_sum

            / relative_count.replace(
                0,
                np.nan,
            )
        )

        df[
            f"horse_{label}_avg_speed_mps_before"
        ] = (

            speed_sum

            / speed_count.replace(
                0,
                np.nan,
            )
        )

    return df


def add_entity_daily_form(
    df,
    entity,
    prefix,
):

    base = df[
        [
            entity,
            "race_date",
            "_started",
            "_finish_num",
        ]
    ].copy()

    base["wins"] = (

        (
            base["_finish_num"] == 1
        )

        & (
            base["_started"] == 1
        )
    ).astype(int)

    base["top3"] = (

        (
            base["_finish_num"] <= 3
        )

        & (
            base["_started"] == 1
        )
    ).astype(int)

    daily = (

        base.groupby(
            [
                entity,
                "race_date",
            ],
            as_index=False,
            dropna=False,
        )

        .agg(
            starts=(
                "_started",
                "sum",
            ),

            wins=(
                "wins",
                "sum",
            ),

            top3=(
                "top3",
                "sum",
            ),
        )

        .sort_values(
            [
                entity,
                "race_date",
            ],
            kind="mergesort",
        )
    )

    pieces = []

    for _, group in daily.groupby(
        entity,
        dropna=False,
        sort=False,
    ):

        group = (
            group.sort_values(
                "race_date"
            )
            .copy()
        )

        group[
            "career_starts_before"
        ] = (

            group["starts"]
            .cumsum()
            .shift(1)
            .fillna(0)
        )

        group[
            "career_wins_before"
        ] = (

            group["wins"]
            .cumsum()
            .shift(1)
            .fillna(0)
        )

        group[
            "career_top3_before"
        ] = (

            group["top3"]
            .cumsum()
            .shift(1)
            .fillna(0)
        )

        indexed = group.set_index(
            "race_date"
        )

        for days in (
            30,
            90,
            365,
        ):

            group[
                f"starts_{days}d"
            ] = (

                indexed["starts"]

                .rolling(
                    f"{days}D",
                    closed="left",
                )

                .sum()

                .to_numpy()
            )

            group[
                f"wins_{days}d"
            ] = (

                indexed["wins"]

                .rolling(
                    f"{days}D",
                    closed="left",
                )

                .sum()

                .to_numpy()
            )

            group[
                f"top3_{days}d"
            ] = (

                indexed["top3"]

                .rolling(
                    f"{days}D",
                    closed="left",
                )

                .sum()

                .to_numpy()
            )

        pieces.append(
            group
        )

    if pieces:

        stats = pd.concat(
            pieces,
            ignore_index=True,
        )

    else:

        stats = daily

    stats[
        f"{prefix}_career_win_rate_before"
    ] = (

        stats[
            "career_wins_before"
        ]

        / stats[
            "career_starts_before"
        ].replace(
            0,
            np.nan,
        )
    )

    stats[
        f"{prefix}_career_top3_rate_before"
    ] = (

        stats[
            "career_top3_before"
        ]

        / stats[
            "career_starts_before"
        ].replace(
            0,
            np.nan,
        )
    )

    keep = [
        entity,
        "race_date",
    ]

    for days in (
        30,
        90,
        365,
    ):

        stats[
            f"{prefix}_starts_{days}d"
        ] = (
            stats[
                f"starts_{days}d"
            ]
        )

        stats[
            f"{prefix}_win_rate_{days}d"
        ] = (

            stats[
                f"wins_{days}d"
            ]

            / stats[
                f"starts_{days}d"
            ].replace(
                0,
                np.nan,
            )
        )

        stats[
            f"{prefix}_top3_rate_{days}d"
        ] = (

            stats[
                f"top3_{days}d"
            ]

            / stats[
                f"starts_{days}d"
            ].replace(
                0,
                np.nan,
            )
        )

        keep += [

            f"{prefix}_starts_{days}d",

            f"{prefix}_win_rate_{days}d",

            f"{prefix}_top3_rate_{days}d",
        ]

    keep += [

        f"{prefix}_career_win_rate_before",

        f"{prefix}_career_top3_rate_before",
    ]

    return df.merge(

        stats[
            keep
        ],

        on=[
            entity,
            "race_date",
        ],

        how="left",

        sort=False,
    )


def add_daily_combo_stats(
    df,
    keys: Sequence[str],
    prefix,
):

    columns = [
        *keys,
        "race_date",
        "_started",
        "_finish_num",
    ]

    base = df[
        columns
    ].copy()

    base["wins"] = (

        (
            base["_finish_num"] == 1
        )

        & (
            base["_started"] == 1
        )
    ).astype(int)

    base["top3"] = (

        (
            base["_finish_num"] <= 3
        )

        & (
            base["_started"] == 1
        )
    ).astype(int)

    daily = (

        base.groupby(
            [
                *keys,
                "race_date",
            ],
            as_index=False,
            dropna=False,
        )

        .agg(

            starts=(
                "_started",
                "sum",
            ),

            wins=(
                "wins",
                "sum",
            ),

            top3=(
                "top3",
                "sum",
            ),
        )

        .sort_values(
            [
                *keys,
                "race_date",
            ],
            kind="mergesort",
        )
    )

    grouped = daily.groupby(
        list(keys),
        dropna=False,
        sort=False,
    )

    daily[
        f"{prefix}_starts_before"
    ] = (

        grouped["starts"]
        .cumsum()

        - daily["starts"]
    )

    daily[
        f"{prefix}_wins_before"
    ] = (

        grouped["wins"]
        .cumsum()

        - daily["wins"]
    )

    daily[
        f"{prefix}_top3_before"
    ] = (

        grouped["top3"]
        .cumsum()

        - daily["top3"]
    )

    daily[
        f"{prefix}_win_rate_before"
    ] = (

        daily[
            f"{prefix}_wins_before"
        ]

        / daily[
            f"{prefix}_starts_before"
        ].replace(
            0,
            np.nan,
        )
    )

    daily[
        f"{prefix}_top3_rate_before"
    ] = (

        daily[
            f"{prefix}_top3_before"
        ]

        / daily[
            f"{prefix}_starts_before"
        ].replace(
            0,
            np.nan,
        )
    )

    keep = [

        *keys,

        "race_date",

        f"{prefix}_starts_before",

        f"{prefix}_wins_before",

        f"{prefix}_top3_before",

        f"{prefix}_win_rate_before",

        f"{prefix}_top3_rate_before",
    ]

    return df.merge(

        daily[
            keep
        ],

        on=[
            *keys,
            "race_date",
        ],

        how="left",

        sort=False,
    )


def add_entity_context_stats(
    df,
    entity,
    context,
    prefix,
):

    return add_daily_combo_stats(
        df,
        [
            entity,
            context,
        ],
        prefix,
    )


def add_post_history_field_features(df):

    grouped = df.groupby(
        "race_id",
        dropna=False,
    )

    if (
        "avg_speed_mps_last5"
        in df.columns
    ):

        df[
            "field_avg_recent_speed_last5"
        ] = (

            grouped[
                "avg_speed_mps_last5"
            ]

            .transform(
                "mean"
            )
        )

        df[
            "recent_speed_vs_field_mean"
        ] = (

            df[
                "avg_speed_mps_last5"
            ]

            - df[
                "field_avg_recent_speed_last5"
            ]
        )

        df[
            "recent_speed_rank_in_field"
        ] = (

            grouped[
                "avg_speed_mps_last5"
            ]

            .rank(
                method="min",
                ascending=False,
            )
        )

    if (
        "avg_relative_finish_last5"
        in df.columns
    ):

        df[
            "field_avg_recent_relative_finish_last5"
        ] = (

            grouped[
                "avg_relative_finish_last5"
            ]

            .transform(
                "mean"
            )
        )

        df[
            "recent_form_vs_field_mean"
        ] = (

            df[
                "avg_relative_finish_last5"
            ]

            - df[
                "field_avg_recent_relative_finish_last5"
            ]
        )

        df[
            "recent_form_rank_in_field"
        ] = (

            grouped[
                "avg_relative_finish_last5"
            ]

            .rank(
                method="min",
                ascending=True,
            )
        )

    comparisons = [

        (
            "career_win_rate_before",
            True,
            "career_win_rate",
        ),

        (
            "trainer_win_rate_90d",
            True,
            "trainer_form",
        ),

        (
            "jockey_win_rate_90d",
            True,
            "jockey_form",
        ),
    ]

    for (
        column,
        better_high,
        short_name,
    ) in comparisons:

        if column in df.columns:

            df[
                f"{short_name}_rank_in_field"
            ] = (

                grouped[
                    column
                ]

                .rank(
                    method="min",
                    ascending=not better_high,
                )
            )

            df[
                f"{short_name}_vs_field_mean"
            ] = (

                df[column]

                - grouped[
                    column
                ].transform(
                    "mean"
                )
            )

    weight_columns = [

        (
            "actual_weight",
            "carried_weight",
        ),

        (
            "declared_horse_weight",
            "bodyweight",
        ),
    ]

    for column, name in weight_columns:

        if column in df.columns:

            df[
                f"field_avg_{name}"
            ] = (

                grouped[
                    column
                ]

                .transform(
                    "mean"
                )
            )

            df[
                f"{name}_vs_field_mean"
            ] = (

                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

                - df[
                    f"field_avg_{name}"
                ]
            )

            df[
                f"{name}_rank_in_field"
            ] = (

                grouped[
                    column
                ]

                .rank(
                    method="min",
                    ascending=False,
                )
            )

    return df


def add_market_features(df):

    odds = pd.to_numeric(
        df["odds"],
        errors="coerce",
    )

    df[
        "market_implied_probability_raw"
    ] = np.where(

        odds > 0,

        1.0 / odds,

        np.nan,
    )

    grouped = df.groupby(
        "race_id",
        dropna=False,
    )

    df[
        "market_overround"
    ] = (

        grouped[
            "market_implied_probability_raw"
        ]

        .transform(
            "sum"
        )
    )

    df[
        "market_implied_probability_normalized"
    ] = (

        df[
            "market_implied_probability_raw"
        ]

        / df[
            "market_overround"
        ].replace(
            0,
            np.nan,
        )
    )

    df[
        "market_rank"
    ] = (

        grouped[
            "odds"
        ]

        .rank(
            method="min",
            ascending=True,
        )
    )

    df[
        "market_favourite_flag"
    ] = np.where(

        df[
            "market_rank"
        ].notna(),

        (
            df[
                "market_rank"
            ]
            == 1
        ).astype(int),

        np.nan,
    )

    return df


def reorder_output(
    df,
    original_columns,
):

    original = [
        column
        for column in original_columns
        if column in df.columns
    ]

    engineered = [
        column
        for column in df.columns
        if column not in original
    ]

    return df[
        original
        + engineered
    ]


def build_features(
    df,
    include_market_features=False,
    keep_helpers=False,
):

    original_columns = (
        df.columns.tolist()
    )

    require_columns(
        df,
        REQUIRED_COLUMNS,
    )

    print(
        "1/11 Base helpers..."
    )
    df = add_base_helpers(
        df
    )

    print(
        "2/11 Race performance helpers..."
    )
    df = add_race_outcome_helpers(
        df
    )

    print(
        "3/11 Context speed ratings..."
    )
    df = add_context_speed_z(
        df
    )

    print(
        "4/11 Horse Elo..."
    )
    df = add_horse_elo(
        df
    )

    print(
        "5/11 Field features..."
    )
    df = add_basic_field_features(
        df
    )

    print(
        "6/11 Last-5 and rolling horse history..."
    )
    df = add_horse_history(
        df
    )

    print(
        "7/11 Distance/course/surface/going/class history..."
    )
    df = add_horse_context_stats(
        df
    )

    print(
        "8/11 Trainer form..."
    )
    df = add_entity_daily_form(
        df,
        "trainer",
        "trainer",
    )

    print(
        "9/11 Jockey form..."
    )
    df = add_entity_daily_form(
        df,
        "jockey",
        "jockey",
    )

    print(
        "10/11 Combinations and context..."
    )

    df = add_daily_combo_stats(
        df,
        [
            "horse_id",
            "jockey",
        ],
        "horse_jockey",
    )

    df = add_daily_combo_stats(
        df,
        [
            "trainer",
            "jockey",
        ],
        "trainer_jockey",
    )

    for entity in (
        "trainer",
        "jockey",
    ):

        contexts = [

            (
                "racecourse_code",
                "track",
            ),

            (
                "distance_m",
                "distance",
            ),

            (
                "race_class",
                "class",
            ),
        ]

        for context, short_name in contexts:

            df = add_entity_context_stats(

                df,

                entity,

                context,

                f"{entity}_{short_name}",
            )

    print(
        "11/11 Relative-to-field features..."
    )

    df = add_post_history_field_features(
        df
    )

    if include_market_features:

        print(
            "Adding current-race market features..."
        )

        df = add_market_features(
            df
        )

    df = (

        df.sort_values(
            "_event_order",
            kind="mergesort",
        )

        .reset_index(
            drop=True
        )
    )

    if not keep_helpers:

        df = df.drop(

            columns=[
                column
                for column in INTERNAL_COLUMNS
                if column in df.columns
            ]
        )

    df["race_date"] = (

        pd.to_datetime(
            df["race_date"]
        )

        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    for i in range(
        1,
        6,
    ):

        column = (
            f"last{i}_date"
        )

        if column in df.columns:

            df[column] = (

                pd.to_datetime(
                    df[column],
                    errors="coerce",
                )

                .dt.strftime(
                    "%Y-%m-%d"
                )
            )

    df = reorder_output(
        df,
        original_columns,
    )

    generated = [
        column
        for column in df.columns
        if column not in original_columns
    ]

    return (
        df,
        generated,
    )


def write_manifest(
    path,
    generated,
    include_market,
):

    manifest = {

        "description":
            "HKJC point-in-time engineered feature manifest",

        "generated_feature_count":
            len(generated),

        "generated_columns":
            list(generated),

        "current_race_columns_never_use_as_predictors":
            CURRENT_RACE_OUTCOME_COLUMNS,

        "market_features_enabled":
            include_market,

        "history_convention":
            (
                "last1_* is the most recent PRIOR "
                "completed start. last5_* is the fifth "
                "most recent PRIOR completed start."
            ),

        "trainer_jockey_timing":
            (
                "Rolling trainer/jockey statistics "
                "exclude all outcomes from the "
                "current race date."
            ),

        "notes": [

            (
                "Upcoming rows with blank "
                "finishing_position are not counted "
                "as completed starts."
            ),

            (
                "horse_age_at_race is not imputed."
            ),

            (
                "Ambiguous margin codes are not "
                "given invented numeric values."
            ),
        ],
    }

    path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():

    args = parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    if args.manifest:

        manifest_path = Path(
            args.manifest
        )

    else:

        manifest_path = (
            output_path.with_suffix(
                output_path.suffix
                + ".features.json"
            )
        )

    print(
        f"Reading: {input_path}"
    )

    df = pd.read_csv(
        input_path,
        low_memory=False,
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Input columns: {len(df.columns):,}"
    )

    enriched, generated = build_features(

        df,

        include_market_features=(
            args.include_market_features
        ),

        keep_helpers=(
            args.keep_helper_columns
        ),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Writing: {output_path}"
    )

    enriched.to_csv(
        output_path,
        index=False,
    )

    write_manifest(
        manifest_path,
        generated,
        args.include_market_features,
    )

    print()
    print(
        "COMPLETE"
    )

    print(
        f"Output rows: {len(enriched):,}"
    )

    print(
        f"Output columns: {len(enriched.columns):,}"
    )

    print(
        f"Engineered columns added: {len(generated):,}"
    )

    print(
        f"Output file: {output_path}"
    )

    print(
        f"Feature manifest: {manifest_path}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT train using current-race outcome "
        "columns as predictors."
    )

    print(
        "Examples: finishing_position, is_winner, "
        "is_top_three, finish_time, margin, "
        "horse_rating_after."
    )


if __name__ == "__main__":
    main()
