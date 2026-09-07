#!/usr/bin/env python3
"""
Build a point-in-time-safe CatBoost-ready HKJC dataset.

Default GitHub-repo paths:
    history: results/races/all_results.csv
    training rows: results/races/all_results_with_races_removed.csv
    output: results/races/all_results_ML_ready.csv

Core design
-----------
1. Use all_results.csv as the COMPLETE historical event stream for feature creation.
2. Use all_results_with_races_removed.csv only to decide which race/horse rows are emitted.
3. Every engineered predictor is based only on information from PRIOR race dates/events.
4. Current-race outcome columns are retained for target/audit purposes, but are explicitly
   marked as NEVER-use predictors.
5. The first intended target is is_winner.

This means a race removed from the training dataset can still contribute valid historical
information to a later race, provided it occurred earlier in time.

Dependencies:
    pandas
    numpy

Example:
    python scripts/build_hkjc_ml_ready.py

Optional:
    python scripts/build_hkjc_ml_ready.py \
        --history results/races/all_results.csv \
        --rows results/races/all_results_with_races_removed.csv \
        --output results/races/all_results_ML_ready.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_HISTORY = "results/races/all_results.csv"
DEFAULT_ROWS = "results/races/all_results_with_races_removed.csv"
DEFAULT_OUTPUT = "results/races/all_results_ML_ready.csv"

TARGET_COLUMN = "is_winner"

REQUIRED_COLUMNS = [
    "result_id",
    "race_id",
    "race_date",
    "racecourse_code",
    "race_number",
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
    "race_class",
    "going",
    "surface",
    "course",
]

# Columns that may legitimately remain in the exported CSV for target/audit/reconstruction,
# but MUST NOT be used as predictors for the current race.
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
    "result_status",
    "is_completed",
    "is_dead_heat",
]

IDENTIFIER_OR_DISPLAY_COLUMNS = [
    "result_id",
    "race_id",
    "race_url",
    "horse_profile_url",
    "horse_profile_scraped_at",
    "horse_name",
    "race_name",
    "brand_number",
    # Timing / integrity caution: retain for auditing, exclude from first-pass predictors.
    "actual_field_size",
    "actually_started",
    "career_prize_money_before",
    "race_date",
]

# These are useful in the master export, but generally should not be fed to CatBoost directly.
NEVER_PREDICTOR_COLUMNS = sorted(
    set(CURRENT_RACE_OUTCOME_COLUMNS + IDENTIFIER_OR_DISPLAY_COLUMNS)
)

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
    "_emit_row",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build point-in-time-safe HKJC CatBoost-ready features."
    )
    parser.add_argument(
        "--history",
        default=DEFAULT_HISTORY,
        help=f"Complete historical results CSV. Default: {DEFAULT_HISTORY}",
    )
    parser.add_argument(
        "--rows",
        default=DEFAULT_ROWS,
        help=(
            "Rows to emit in final ML dataset. Usually the cleaned/filtered results file. "
            f"Default: {DEFAULT_ROWS}"
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON feature manifest path.",
    )
    parser.add_argument(
        "--include-market-features",
        action="store_true",
        help=(
            "Include current-race odds/market features. Only enable if the odds in your source "
            "represent information available at the exact prediction cutoff."
        ),
    )
    parser.add_argument(
        "--keep-helper-columns",
        action="store_true",
        help="Keep internal helper columns for debugging.",
    )
    parser.add_argument(
        "--history-lags",
        type=int,
        default=10,
        help="Number of prior horse starts to expose as lastN_* columns. Default: 10",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def to_num(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int64")
    text = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": 1,
        "1": 1,
        "yes": 1,
        "y": 1,
        "false": 0,
        "0": 0,
        "no": 0,
        "n": 0,
    }
    return text.map(mapping).astype("Int64")


def parse_margin_lengths(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().upper()
    if not value:
        return np.nan
    if value in {"---", "ML", "TO"}:
        return np.nan

    approximations = {
        "DH": 0.0,
        "N": 0.05,
        "NOSE": 0.05,
        "+NOSE": 0.05,
        "SH": 0.10,
        "+SH": 0.10,
        "HD": 0.20,
        "+HD": 0.20,
        "NK": 0.25,
        "NECK": 0.25,
        "+NK": 0.25,
    }
    if value in approximations:
        return approximations[value]
    if value.startswith("+"):
        value = value[1:]
    try:
        if "-" in value:
            whole, fraction = value.split("-", 1)
            numerator, denominator = fraction.split("/", 1)
            return float(whole) + float(numerator) / float(denominator)
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return float(numerator) / float(denominator)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return np.nan


def parse_finish_time_seconds(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if not value or value == "---":
        return np.nan
    match = re.fullmatch(r"(?:(\d+):)?(\d{1,2}(?:\.\d+)?)", value)
    if not match:
        return np.nan
    minutes = int(match.group(1) or 0)
    seconds = float(match.group(2))
    return minutes * 60.0 + seconds


def parse_class_number(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().upper()
    if re.fullmatch(r"[1-5]", value):
        return float(value)
    return np.nan


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    date = df["race_date"]
    df["race_year"] = date.dt.year.astype("Int64")
    df["race_month"] = date.dt.month.astype("Int64")
    df["race_day"] = date.dt.day.astype("Int64")
    df["race_dayofweek"] = date.dt.dayofweek.astype("Int64")
    df["race_is_weekend"] = date.dt.dayofweek.isin([5, 6]).astype(np.int8)
    df["race_season_month"] = date.dt.month.astype("Int64")
    # HK racing season roughly begins Sep. This is a stable calendar descriptor, not outcome data.
    df["hk_season_start_year"] = np.where(date.dt.month >= 9, date.dt.year, date.dt.year - 1)
    return df


def add_base_helpers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    if df["race_date"].isna().any():
        raise ValueError(f"{int(df['race_date'].isna().sum())} rows have invalid race_date")

    numeric_columns = [
        "race_number", "race_index", "distance_m", "field_size", "actual_field_size",
        "horse_number", "horse_rating_before", "horse_rating_after", "finishing_position",
        "actual_weight", "declared_horse_weight", "draw", "odds", "career_starts_before",
        "career_wins_before", "career_seconds_before", "career_thirds_before",
        "career_top3_before", "career_win_rate_before", "career_top3_rate_before",
        "career_prize_money_before", "career_prize_money_after", "prize_money_hkd",
        "prize_money_won_this_race", "prize_payout_percentage",
    ]
    to_num(df, numeric_columns)

    for col in ["is_winner", "is_top_three", "actually_started", "is_completed", "is_dead_heat"]:
        if col in df.columns:
            df[col] = normalize_bool(df[col])

    df["_row_id"] = np.arange(len(df), dtype=np.int64)

    sort_columns = ["race_date"]
    if "race_index" in df.columns:
        sort_columns.append("race_index")
    sort_columns += ["race_number", "race_id"]
    if "horse_number" in df.columns:
        sort_columns.append("horse_number")
    sort_columns.append("_row_id")

    df = df.sort_values(sort_columns, kind="mergesort", na_position="last").reset_index(drop=True)
    df["_event_order"] = np.arange(len(df), dtype=np.int64)

    df["_finish_num"] = pd.to_numeric(df["finishing_position"], errors="coerce")
    if "actually_started" in df.columns:
        df["_started"] = df["actually_started"].fillna(0).astype(int)
    else:
        df["_started"] = df["_finish_num"].notna().astype(np.int8)

    # Prefer actual_field_size when it is already known and represents the final starter count.
    # For generic pre-race use, keep field_size as the primary field-size predictor.
    denom = pd.to_numeric(df["field_size"], errors="coerce") - 1.0
    df["_relative_finish"] = np.where(
        df["_finish_num"].notna() & (denom > 0),
        (df["_finish_num"] - 1.0) / denom,
        np.nan,
    )

    df["_margin_lengths"] = df["margin"].map(parse_margin_lengths)
    df.loc[(df["_finish_num"] == 1) & df["_margin_lengths"].isna(), "_margin_lengths"] = 0.0
    df["_finish_time_seconds"] = df["finish_time"].map(parse_finish_time_seconds)
    df["_speed_mps"] = np.where(
        df["_finish_time_seconds"] > 0,
        pd.to_numeric(df["distance_m"], errors="coerce") / df["_finish_time_seconds"],
        np.nan,
    )
    df["_race_class_num"] = df["race_class"].map(parse_class_number)

    df["distance_bucket"] = pd.cut(
        pd.to_numeric(df["distance_m"], errors="coerce"),
        bins=[-np.inf, 1200, 1400, 1650, 2000, np.inf],
        labels=[
            "sprint_1200_or_less", "1400", "mile_1600_1650",
            "middle_1800_2000", "staying_over_2000",
        ],
    ).astype("string")

    going = df["going"].astype("string").str.upper()
    conditions = [
        going.str.contains("FIRM", na=False),
        going.eq("GOOD").fillna(False),
        going.str.contains("YIELD", na=False),
        going.str.contains("SOFT", na=False),
        going.str.contains("WET", na=False),
        going.str.contains("FAST", na=False),
    ]
    df["going_bucket"] = np.select(
        [c.to_numpy(dtype=bool, na_value=False) for c in conditions],
        ["firm", "good", "yielding", "soft", "wet", "fast"],
        default="other",
    )

    if "country_of_origin" in df.columns:
        country = df["country_of_origin"].astype("string").str.upper().str.strip()
        northern = {"GB", "IRE", "FR", "USA", "CAN", "JPN", "GER", "ITY", "SPA", "GR"}
        southern = {"AUS", "NZ", "SAF", "ARG", "BRZ", "CHI", "URU"}
        derived = pd.Series(pd.NA, index=df.index, dtype="string")
        derived.loc[country.isin(northern)] = "Northern"
        derived.loc[country.isin(southern)] = "Southern"
        df["hemisphere_of_origin_derived"] = derived
        if "hemisphere_of_origin" in df.columns:
            df["hemisphere_mapping_conflict"] = np.where(
                derived.notna() & df["hemisphere_of_origin"].notna(),
                (derived.astype(str) != df["hemisphere_of_origin"].astype(str)).astype(int),
                0,
            )

    df = add_date_features(df)
    return df


def add_race_outcome_helpers(df: pd.DataFrame) -> pd.DataFrame:
    winner_times = (
        df.loc[df["_finish_num"] == 1]
        .groupby("race_id", dropna=False)["_finish_time_seconds"]
        .min()
    )
    df["_winner_time_seconds"] = df["race_id"].map(winner_times)
    df["_time_behind_winner_seconds"] = np.where(
        df["_finish_time_seconds"].notna() & df["_winner_time_seconds"].notna(),
        df["_finish_time_seconds"] - df["_winner_time_seconds"],
        np.nan,
    )
    return df


def add_context_speed_z(df: pd.DataFrame, min_prior_observations: int = 20) -> pd.DataFrame:
    context = ["racecourse_code", "surface", "course", "distance_m", "going"]
    valid = df["_speed_mps"].notna()
    temp = df.loc[valid, ["race_id", "race_date", "_event_order", *context, "_speed_mps"]].copy()
    if temp.empty:
        df["_speed_context_z"] = np.nan
        return df

    temp["_sq"] = temp["_speed_mps"] ** 2
    race_stats = (
        temp.groupby("race_id", as_index=False, dropna=False)
        .agg(
            race_date=("race_date", "first"),
            event_order=("_event_order", "min"),
            speed_sum=("_speed_mps", "sum"),
            speed_sumsq=("_sq", "sum"),
            speed_count=("_speed_mps", "count"),
            **{f"ctx_{c}": (c, "first") for c in context},
        )
        .sort_values(["race_date", "event_order"], kind="mergesort")
    )
    context_columns = [f"ctx_{c}" for c in context]
    grouped = race_stats.groupby(context_columns, dropna=False, sort=False)
    race_stats["prior_sum"] = grouped["speed_sum"].cumsum() - race_stats["speed_sum"]
    race_stats["prior_sumsq"] = grouped["speed_sumsq"].cumsum() - race_stats["speed_sumsq"]
    race_stats["prior_count"] = grouped["speed_count"].cumsum() - race_stats["speed_count"]
    race_stats["prior_mean"] = race_stats["prior_sum"] / race_stats["prior_count"].replace(0, np.nan)
    variance = (
        race_stats["prior_sumsq"] / race_stats["prior_count"].replace(0, np.nan)
        - race_stats["prior_mean"] ** 2
    )
    race_stats["prior_std"] = np.sqrt(variance.clip(lower=0))
    benchmark = race_stats.set_index("race_id")[["prior_count", "prior_mean", "prior_std"]]
    df = df.join(benchmark, on="race_id")
    valid_benchmark = (df["prior_count"] >= min_prior_observations) & (df["prior_std"] > 0)
    df["_speed_context_z"] = np.where(
        valid_benchmark & df["_speed_mps"].notna(),
        (df["_speed_mps"] - df["prior_mean"]) / df["prior_std"],
        np.nan,
    )
    return df.drop(columns=["prior_count", "prior_mean", "prior_std"])


def add_horse_elo(df: pd.DataFrame, base: float = 1500.0, k: float = 16.0) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    elo_before = np.full(len(df), np.nan, dtype=float)

    for _, race in df.groupby("race_id", sort=False):
        indices = race.index.to_numpy()
        horses = race["horse_id"].astype(str).to_numpy()
        pre = np.array([ratings.get(h, base) for h in horses], dtype=float)
        elo_before[indices] = pre
        finish = race["_finish_num"].to_numpy(dtype=float)
        valid_positions = np.where(~np.isnan(finish))[0]
        if len(valid_positions) < 2:
            continue

        delta = np.zeros(len(race), dtype=float)
        for a in range(len(valid_positions)):
            i = valid_positions[a]
            for b in range(a + 1, len(valid_positions)):
                j = valid_positions[b]
                expected_i = 1.0 / (1.0 + 10.0 ** ((pre[j] - pre[i]) / 400.0))
                actual_i = 1.0 if finish[i] < finish[j] else 0.0 if finish[i] > finish[j] else 0.5
                difference = k * (actual_i - expected_i)
                delta[i] += difference
                delta[j] -= difference

        opponents = max(len(valid_positions) - 1, 1)
        for local_index in valid_positions:
            horse = horses[local_index]
            ratings[horse] = pre[local_index] + delta[local_index] / opponents

    df["horse_elo_before"] = elo_before
    return df


def add_basic_field_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("race_id", dropna=False)
    rating = pd.to_numeric(df["horse_rating_before"], errors="coerce")

    df["field_avg_rating"] = grouped["horse_rating_before"].transform("mean")
    df["field_median_rating"] = grouped["horse_rating_before"].transform("median")
    df["field_max_rating"] = grouped["horse_rating_before"].transform("max")
    df["field_min_rating"] = grouped["horse_rating_before"].transform("min")
    df["field_rating_std"] = grouped["horse_rating_before"].transform("std")
    df["field_rating_range"] = df["field_max_rating"] - df["field_min_rating"]
    df["rating_vs_field_mean"] = rating - df["field_avg_rating"]
    df["rating_vs_field_median"] = rating - df["field_median_rating"]
    df["rating_vs_field_max"] = rating - df["field_max_rating"]
    df["rating_rank_in_field"] = grouped["horse_rating_before"].rank(method="min", ascending=False)
    df["rating_percentile_in_field"] = grouped["horse_rating_before"].rank(pct=True, ascending=True)

    field_size = pd.to_numeric(df["field_size"], errors="coerce")
    draw = pd.to_numeric(df["draw"], errors="coerce")
    df["draw_normalized"] = np.where(field_size > 0, draw / field_size, np.nan)
    df["draw_percentile"] = np.where(field_size > 1, (draw - 1.0) / (field_size - 1.0), np.nan)
    df["inside_draw_flag"] = np.where(df["draw_percentile"].notna(), (df["draw_percentile"] <= 0.33).astype(int), np.nan)
    df["middle_draw_flag"] = np.where(df["draw_percentile"].notna(), ((df["draw_percentile"] > 0.33) & (df["draw_percentile"] < 0.67)).astype(int), np.nan)
    df["outside_draw_flag"] = np.where(df["draw_percentile"].notna(), (df["draw_percentile"] >= 0.67).astype(int), np.nan)

    df["horse_elo_rank_in_field"] = grouped["horse_elo_before"].rank(method="min", ascending=False)
    df["horse_elo_vs_field_mean"] = df["horse_elo_before"] - grouped["horse_elo_before"].transform("mean")

    if "actual_weight" in df.columns:
        df["field_avg_carried_weight"] = grouped["actual_weight"].transform("mean")
        df["carried_weight_vs_field_mean"] = pd.to_numeric(df["actual_weight"], errors="coerce") - df["field_avg_carried_weight"]
        df["carried_weight_rank_in_field"] = grouped["actual_weight"].rank(method="min", ascending=False)

    if "declared_horse_weight" in df.columns:
        df["field_avg_bodyweight"] = grouped["declared_horse_weight"].transform("mean")
        df["bodyweight_vs_field_mean"] = pd.to_numeric(df["declared_horse_weight"], errors="coerce") - df["field_avg_bodyweight"]
        df["bodyweight_rank_in_field"] = grouped["declared_horse_weight"].rank(method="min", ascending=False)

    return df


def rowwise_slope(values: pd.DataFrame) -> np.ndarray:
    y = values.to_numpy(dtype=float)
    x = np.arange(y.shape[1], dtype=float)[None, :]
    mask = ~np.isnan(y)
    n = mask.sum(axis=1).astype(float)
    sx = (mask * x).sum(axis=1)
    sy = np.nansum(y, axis=1)
    sxx = (mask * (x ** 2)).sum(axis=1)
    sxy = np.nansum(np.where(mask, y * x, np.nan), axis=1)
    denominator = n * sxx - sx * sx
    output = np.full(len(values), np.nan, dtype=float)
    valid = (n >= 2) & (denominator != 0)
    output[valid] = (n[valid] * sxy[valid] - sx[valid] * sy[valid]) / denominator[valid]
    return output


def add_horse_history(df: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    starts = df.loc[df["_started"] == 1].copy()
    if starts.empty:
        return df

    history_sources = {
        "date": "race_date",
        "race_id": "race_id",
        "finish": "_finish_num",
        "relative_finish": "_relative_finish",
        "margin_lengths": "_margin_lengths",
        "finish_time_seconds": "_finish_time_seconds",
        "speed_mps": "_speed_mps",
        "speed_context_z": "_speed_context_z",
        "time_behind_winner_seconds": "_time_behind_winner_seconds",
        "distance_m": "distance_m",
        "race_class": "race_class",
        "race_class_num": "_race_class_num",
        "horse_rating_before": "horse_rating_before",
        "horse_rating_after": "horse_rating_after",
        "actual_weight": "actual_weight",
        "declared_horse_weight": "declared_horse_weight",
        "draw": "draw",
        "odds": "odds",
        "track": "racecourse_code",
        "surface": "surface",
        "course": "course",
        "going": "going",
        "jockey": "jockey",
        "trainer": "trainer",
        "field_size": "field_size",
        "field_avg_rating": "field_avg_rating",
        "field_max_rating": "field_max_rating",
        "rating_vs_field_mean": "rating_vs_field_mean",
        "is_winner": "is_winner",
        "is_top_three": "is_top_three",
    }
    source_columns = [c for c in history_sources.values() if c in starts.columns]
    name_for_source = {value: key for key, value in history_sources.items() if value in source_columns}
    grouped = starts.groupby("horse_id", sort=False)[source_columns]

    lag_frames = []
    for lag in range(max_lag):
        block = starts[source_columns].copy() if lag == 0 else grouped.shift(lag)
        block = block.rename(columns={source: f"last{lag + 1}_{name_for_source[source]}" for source in source_columns})
        lag_frames.append(block)
    history = pd.concat(lag_frames, axis=1)

    summary = pd.DataFrame(index=starts.index)
    for n in (3, 5, 10):
        if n > max_lag:
            continue
        finish_cols = [f"last{i}_finish" for i in range(1, n + 1)]
        rel_cols = [f"last{i}_relative_finish" for i in range(1, n + 1)]
        speed_cols = [f"last{i}_speed_mps" for i in range(1, n + 1)]
        speed_z_cols = [f"last{i}_speed_context_z" for i in range(1, n + 1)]
        margin_cols = [f"last{i}_margin_lengths" for i in range(1, n + 1)]

        finish_values = history[finish_cols].astype(float)
        summary[f"avg_finish_last{n}"] = finish_values.mean(axis=1)
        summary[f"median_finish_last{n}"] = finish_values.median(axis=1)
        summary[f"best_finish_last{n}"] = finish_values.min(axis=1)
        summary[f"worst_finish_last{n}"] = finish_values.max(axis=1)
        summary[f"finish_std_last{n}"] = finish_values.std(axis=1, ddof=1)
        summary[f"avg_relative_finish_last{n}"] = history[rel_cols].astype(float).mean(axis=1)
        summary[f"avg_margin_lengths_last{n}"] = history[margin_cols].astype(float).mean(axis=1)
        speed_values = history[speed_cols].astype(float)
        summary[f"avg_speed_mps_last{n}"] = speed_values.mean(axis=1)
        summary[f"best_speed_mps_last{n}"] = speed_values.max(axis=1)
        summary[f"speed_std_last{n}"] = speed_values.std(axis=1, ddof=1)
        summary[f"avg_speed_context_z_last{n}"] = history[speed_z_cols].astype(float).mean(axis=1)
        wins = finish_values.eq(1).sum(axis=1)
        top3 = finish_values.le(3).sum(axis=1)
        count = finish_values.notna().sum(axis=1).replace(0, np.nan)
        summary[f"wins_last{n}"] = wins.astype(float)
        summary[f"top3_last{n}"] = top3.astype(float)
        summary[f"win_rate_last{n}"] = wins / count
        summary[f"top3_rate_last{n}"] = top3 / count

    summary["prior_completed_starts_derived"] = starts.groupby("horse_id", sort=False).cumcount() + 1
    if max_lag >= 5:
        chronological_finish = history[[f"last{i}_finish" for i in range(5, 0, -1)]].astype(float)
        chronological_rel = history[[f"last{i}_relative_finish" for i in range(5, 0, -1)]].astype(float)
        chronological_speed = history[[f"last{i}_speed_mps" for i in range(5, 0, -1)]].astype(float)
        summary["finish_trend_last5"] = rowwise_slope(chronological_finish)
        summary["relative_finish_trend_last5"] = rowwise_slope(chronological_rel)
        summary["speed_trend_last5"] = rowwise_slope(chronological_speed)

    starts_features = pd.concat([starts[["horse_id", "_event_order"]], history, summary], axis=1)
    carry_columns = [c for c in starts_features.columns if c not in {"horse_id", "_event_order"}]
    right = starts_features.rename(columns={"_event_order": "_prior_event_order"})
    left = df[["_row_id", "horse_id", "_event_order"]].copy()
    left = left.sort_values(["_event_order", "horse_id"], kind="mergesort")
    right = right.sort_values(["_prior_event_order", "horse_id"], kind="mergesort")

    merged = pd.merge_asof(
        left,
        right,
        left_on="_event_order",
        right_on="_prior_event_order",
        by="horse_id",
        direction="backward",
        allow_exact_matches=False,
    ).set_index("_row_id")

    carry = merged[carry_columns].reindex(df["_row_id"].to_numpy()).copy()
    carry.index = df.index
    df = pd.concat([df, carry], axis=1)

    if "last1_date" in df.columns:
        df["days_since_last_start"] = (
            df["race_date"] - pd.to_datetime(df["last1_date"], errors="coerce")
        ).dt.days

    # Observed-history coverage indicators. These help CatBoost distinguish a genuinely low-history
    # horse from one for which the dataset starts after its career had already begun.
    stored_career = pd.to_numeric(df.get("career_starts_before"), errors="coerce")
    derived_career = pd.to_numeric(df.get("prior_completed_starts_derived"), errors="coerce")
    if stored_career is not None and derived_career is not None:
        df["history_starts_missing_before"] = (stored_career - derived_career).clip(lower=0)
        df["history_coverage_ratio"] = np.where(
            stored_career > 0,
            derived_career / stored_career,
            np.where(derived_career == 0, 1.0, np.nan),
        )
        df["history_appears_complete_flag"] = np.where(
            stored_career.notna() & derived_career.notna(),
            (stored_career == derived_career).astype(int),
            np.nan,
        )

    # Workload counts from complete history.
    start_dates_by_horse = {
        horse: group["race_date"].to_numpy(dtype="datetime64[ns]")
        for horse, group in starts.groupby("horse_id", sort=False)
    }
    target_groups = df.groupby("horse_id", sort=False).groups
    workload = {}
    for window in (14, 30, 60, 90, 180, 365):
        output = np.full(len(df), np.nan, dtype=float)
        for horse, target_indices in target_groups.items():
            target_indices = np.asarray(list(target_indices), dtype=int)
            target_dates = df.loc[target_indices, "race_date"].to_numpy(dtype="datetime64[ns]")
            start_dates = start_dates_by_horse.get(horse)
            if start_dates is None or len(start_dates) == 0:
                output[target_indices] = 0
                continue
            end = np.searchsorted(start_dates, target_dates, side="left")
            begin = np.searchsorted(start_dates, target_dates - np.timedelta64(window, "D"), side="left")
            output[target_indices] = end - begin
        workload[f"starts_last{window}d"] = output
    df = pd.concat([df, pd.DataFrame(workload, index=df.index)], axis=1)

    changes = pd.DataFrame(index=df.index)
    if "last1_distance_m" in df.columns:
        changes["distance_change_from_last"] = pd.to_numeric(df["distance_m"], errors="coerce") - pd.to_numeric(df["last1_distance_m"], errors="coerce")
    if "last1_horse_rating_before" in df.columns:
        changes["rating_change_from_last_start_pre"] = pd.to_numeric(df["horse_rating_before"], errors="coerce") - pd.to_numeric(df["last1_horse_rating_before"], errors="coerce")
    if "last1_horse_rating_after" in df.columns:
        changes["rating_change_since_last_official"] = pd.to_numeric(df["horse_rating_before"], errors="coerce") - pd.to_numeric(df["last1_horse_rating_after"], errors="coerce")
    if "last1_actual_weight" in df.columns:
        changes["actual_weight_change_from_last"] = pd.to_numeric(df["actual_weight"], errors="coerce") - pd.to_numeric(df["last1_actual_weight"], errors="coerce")
    if "last1_declared_horse_weight" in df.columns:
        body_change = pd.to_numeric(df["declared_horse_weight"], errors="coerce") - pd.to_numeric(df["last1_declared_horse_weight"], errors="coerce")
        changes["bodyweight_change_from_last"] = body_change
        changes["bodyweight_change_pct_from_last"] = body_change / pd.to_numeric(df["last1_declared_horse_weight"], errors="coerce").replace(0, np.nan)
    if "last1_race_class_num" in df.columns:
        class_change = df["_race_class_num"] - pd.to_numeric(df["last1_race_class_num"], errors="coerce")
        changes["class_number_change_from_last"] = class_change
        changes["class_drop_flag"] = np.where(class_change.notna(), (class_change > 0).astype(int), np.nan)
        changes["class_rise_flag"] = np.where(class_change.notna(), (class_change < 0).astype(int), np.nan)

    comparisons = [
        ("racecourse_code", "last1_track", "track_change_flag"),
        ("surface", "last1_surface", "surface_change_flag"),
        ("course", "last1_course", "course_change_flag"),
        ("going", "last1_going", "going_change_flag"),
        ("jockey", "last1_jockey", "jockey_change_flag"),
        ("trainer", "last1_trainer", "trainer_change_flag"),
    ]
    for current, prior, name in comparisons:
        if current in df.columns and prior in df.columns:
            both = df[current].notna() & df[prior].notna()
            changes[name] = np.where(both, (df[current].astype(str) != df[prior].astype(str)).astype(int), np.nan)
    df = pd.concat([df, changes], axis=1)

    if max_lag >= 5:
        finish_columns = [f"last{i}_finish" for i in range(1, 6)]
        def format_finish(value):
            if pd.isna(value):
                return ""
            value = float(value)
            return str(int(value)) if value.is_integer() else str(value)
        df["form_last5_latest_first"] = df[finish_columns].apply(
            lambda row: "-".join([x for x in (format_finish(v) for v in row) if x]),
            axis=1,
        )

    return df


def add_horse_context_stats(df: pd.DataFrame) -> pd.DataFrame:
    contexts = {
        "distance": ["distance_m"],
        "track": ["racecourse_code"],
        "track_distance": ["racecourse_code", "distance_m"],
        "surface": ["surface"],
        "going": ["going"],
        "class": ["race_class"],
        "distance_surface": ["distance_m", "surface"],
        "track_surface": ["racecourse_code", "surface"],
    }

    wins = ((df["_finish_num"] == 1) & (df["_started"] == 1)).astype(float)
    top3 = ((df["_finish_num"] <= 3) & (df["_started"] == 1)).astype(float)

    for label, context_columns in contexts.items():
        keys = ["horse_id", *context_columns]
        grouped = df.groupby(keys, dropna=False, sort=False)
        prior_starts = grouped["_started"].cumsum() - df["_started"]
        group_arrays = [df[k] for k in keys]
        cumulative_wins = wins.groupby(group_arrays, dropna=False).cumsum() - wins
        cumulative_top3 = top3.groupby(group_arrays, dropna=False).cumsum() - top3

        rel_valid = df["_relative_finish"].notna().astype(float)
        rel_values = df["_relative_finish"].fillna(0.0)
        rel_sum = rel_values.groupby(group_arrays, dropna=False).cumsum() - rel_values
        rel_count = rel_valid.groupby(group_arrays, dropna=False).cumsum() - rel_valid

        speed_valid = df["_speed_mps"].notna().astype(float)
        speed_values = df["_speed_mps"].fillna(0.0)
        speed_sum = speed_values.groupby(group_arrays, dropna=False).cumsum() - speed_values
        speed_count = speed_valid.groupby(group_arrays, dropna=False).cumsum() - speed_valid

        df[f"horse_{label}_starts_before"] = prior_starts.astype(float)
        df[f"horse_{label}_wins_before"] = cumulative_wins.astype(float)
        df[f"horse_{label}_top3_before"] = cumulative_top3.astype(float)
        df[f"horse_{label}_win_rate_before"] = cumulative_wins / prior_starts.replace(0, np.nan)
        df[f"horse_{label}_top3_rate_before"] = cumulative_top3 / prior_starts.replace(0, np.nan)
        df[f"horse_{label}_avg_relative_finish_before"] = rel_sum / rel_count.replace(0, np.nan)
        df[f"horse_{label}_avg_speed_mps_before"] = speed_sum / speed_count.replace(0, np.nan)

    return df


def add_entity_daily_form(df: pd.DataFrame, entity: str, prefix: str) -> pd.DataFrame:
    base = df[[entity, "race_date", "_started", "_finish_num"]].copy()
    base["wins"] = ((base["_finish_num"] == 1) & (base["_started"] == 1)).astype(int)
    base["top3"] = ((base["_finish_num"] <= 3) & (base["_started"] == 1)).astype(int)

    daily = (
        base.groupby([entity, "race_date"], as_index=False, dropna=False)
        .agg(starts=("_started", "sum"), wins=("wins", "sum"), top3=("top3", "sum"))
        .sort_values([entity, "race_date"], kind="mergesort")
    )

    pieces = []
    for _, group in daily.groupby(entity, dropna=False, sort=False):
        group = group.sort_values("race_date").copy()
        group["career_starts_before"] = group["starts"].cumsum().shift(1).fillna(0)
        group["career_wins_before"] = group["wins"].cumsum().shift(1).fillna(0)
        group["career_top3_before"] = group["top3"].cumsum().shift(1).fillna(0)
        indexed = group.set_index("race_date")
        for days in (30, 90, 180, 365):
            group[f"starts_{days}d"] = indexed["starts"].rolling(f"{days}D", closed="left").sum().to_numpy()
            group[f"wins_{days}d"] = indexed["wins"].rolling(f"{days}D", closed="left").sum().to_numpy()
            group[f"top3_{days}d"] = indexed["top3"].rolling(f"{days}D", closed="left").sum().to_numpy()
        pieces.append(group)

    stats = pd.concat(pieces, ignore_index=True) if pieces else daily
    stats[f"{prefix}_career_starts_before_derived"] = stats["career_starts_before"]
    stats[f"{prefix}_career_win_rate_before"] = stats["career_wins_before"] / stats["career_starts_before"].replace(0, np.nan)
    stats[f"{prefix}_career_top3_rate_before"] = stats["career_top3_before"] / stats["career_starts_before"].replace(0, np.nan)

    keep = [entity, "race_date", f"{prefix}_career_starts_before_derived", f"{prefix}_career_win_rate_before", f"{prefix}_career_top3_rate_before"]
    for days in (30, 90, 180, 365):
        stats[f"{prefix}_starts_{days}d"] = stats[f"starts_{days}d"]
        stats[f"{prefix}_win_rate_{days}d"] = stats[f"wins_{days}d"] / stats[f"starts_{days}d"].replace(0, np.nan)
        stats[f"{prefix}_top3_rate_{days}d"] = stats[f"top3_{days}d"] / stats[f"starts_{days}d"].replace(0, np.nan)
        keep += [f"{prefix}_starts_{days}d", f"{prefix}_win_rate_{days}d", f"{prefix}_top3_rate_{days}d"]

    return df.merge(stats[keep], on=[entity, "race_date"], how="left", sort=False)


def add_daily_combo_stats(df: pd.DataFrame, keys: Sequence[str], prefix: str) -> pd.DataFrame:
    base = df[[*keys, "race_date", "_started", "_finish_num"]].copy()
    base["wins"] = ((base["_finish_num"] == 1) & (base["_started"] == 1)).astype(int)
    base["top3"] = ((base["_finish_num"] <= 3) & (base["_started"] == 1)).astype(int)
    daily = (
        base.groupby([*keys, "race_date"], as_index=False, dropna=False)
        .agg(starts=("_started", "sum"), wins=("wins", "sum"), top3=("top3", "sum"))
        .sort_values([*keys, "race_date"], kind="mergesort")
    )
    grouped = daily.groupby(list(keys), dropna=False, sort=False)
    daily[f"{prefix}_starts_before"] = grouped["starts"].cumsum() - daily["starts"]
    daily[f"{prefix}_wins_before"] = grouped["wins"].cumsum() - daily["wins"]
    daily[f"{prefix}_top3_before"] = grouped["top3"].cumsum() - daily["top3"]
    daily[f"{prefix}_win_rate_before"] = daily[f"{prefix}_wins_before"] / daily[f"{prefix}_starts_before"].replace(0, np.nan)
    daily[f"{prefix}_top3_rate_before"] = daily[f"{prefix}_top3_before"] / daily[f"{prefix}_starts_before"].replace(0, np.nan)
    keep = [
        *keys, "race_date", f"{prefix}_starts_before", f"{prefix}_wins_before",
        f"{prefix}_top3_before", f"{prefix}_win_rate_before", f"{prefix}_top3_rate_before",
    ]
    return df.merge(daily[keep], on=[*keys, "race_date"], how="left", sort=False)


def add_post_history_field_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("race_id", dropna=False)

    comparisons = []
    if "avg_speed_mps_last5" in df.columns:
        df["field_avg_recent_speed_last5"] = grouped["avg_speed_mps_last5"].transform("mean")
        df["recent_speed_vs_field_mean"] = df["avg_speed_mps_last5"] - df["field_avg_recent_speed_last5"]
        df["recent_speed_rank_in_field"] = grouped["avg_speed_mps_last5"].rank(method="min", ascending=False)
    if "avg_relative_finish_last5" in df.columns:
        df["field_avg_recent_relative_finish_last5"] = grouped["avg_relative_finish_last5"].transform("mean")
        df["recent_form_vs_field_mean"] = df["avg_relative_finish_last5"] - df["field_avg_recent_relative_finish_last5"]
        df["recent_form_rank_in_field"] = grouped["avg_relative_finish_last5"].rank(method="min", ascending=True)

    comparisons += [
        ("career_win_rate_before", True, "career_win_rate"),
        ("trainer_win_rate_90d", True, "trainer_form"),
        ("jockey_win_rate_90d", True, "jockey_form"),
        ("horse_track_win_rate_before", True, "horse_track_win_rate"),
        ("horse_distance_win_rate_before", True, "horse_distance_win_rate"),
        ("horse_track_distance_win_rate_before", True, "horse_track_distance_win_rate"),
    ]
    for column, better_high, short_name in comparisons:
        if column in df.columns:
            df[f"{short_name}_rank_in_field"] = grouped[column].rank(method="min", ascending=not better_high)
            df[f"{short_name}_vs_field_mean"] = df[column] - grouped[column].transform("mean")

    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    odds = pd.to_numeric(df["odds"], errors="coerce")
    df["market_implied_probability_raw"] = np.where(odds > 0, 1.0 / odds, np.nan)
    grouped = df.groupby("race_id", dropna=False)
    df["market_overround"] = grouped["market_implied_probability_raw"].transform("sum")
    df["market_implied_probability_normalized"] = df["market_implied_probability_raw"] / df["market_overround"].replace(0, np.nan)
    df["market_rank"] = grouped["odds"].rank(method="min", ascending=True)
    df["market_favourite_flag"] = np.where(df["market_rank"].notna(), (df["market_rank"] == 1).astype(int), np.nan)
    return df


def add_training_metadata(df: pd.DataFrame) -> pd.DataFrame:
    # These columns are not predictors; they make downstream training safer and more reproducible.
    df["ml_target_is_winner"] = pd.to_numeric(df["is_winner"], errors="coerce").astype("Int64")
    df["ml_group_id"] = df["race_id"].astype("string")

    date = pd.to_datetime(df["race_date"], errors="coerce")
    # Recommended time split based on the audited data range. Adjust later as desired.
    df["ml_split"] = np.select(
        [date <= pd.Timestamp("2024-07-31"), date <= pd.Timestamp("2025-07-31")],
        ["train", "validation"],
        default="test",
    )

    df["ml_row_weight"] = 1.0
    return df


def validate_history_and_emit_rows(history: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    require_columns(history, REQUIRED_COLUMNS, "History file")
    require_columns(rows, REQUIRED_COLUMNS, "Rows file")

    if history["result_id"].duplicated().any():
        dupes = int(history["result_id"].duplicated().sum())
        raise ValueError(f"History file has {dupes} duplicate result_id values")
    if rows["result_id"].duplicated().any():
        dupes = int(rows["result_id"].duplicated().sum())
        raise ValueError(f"Rows file has {dupes} duplicate result_id values")

    history_ids = set(history["result_id"].astype(str))
    row_ids = set(rows["result_id"].astype(str))
    missing = row_ids - history_ids
    if missing:
        example = sorted(missing)[:10]
        raise ValueError(
            f"{len(missing):,} rows in the emit file are not present in all_results.csv. "
            f"Examples: {example}"
        )

    history = history.copy()
    history["_emit_row"] = history["result_id"].astype(str).isin(row_ids).astype(np.int8)
    return history


def build_features(
    history_df: pd.DataFrame,
    emit_df: pd.DataFrame,
    include_market_features: bool = False,
    keep_helpers: bool = False,
    history_lags: int = 10,
):
    original_emit_columns = emit_df.columns.tolist()
    df = validate_history_and_emit_rows(history_df, emit_df)

    print("1/12 Base helpers...")
    df = add_base_helpers(df)
    print("2/12 Historical race performance helpers...")
    df = add_race_outcome_helpers(df)
    print("3/12 Historical context speed benchmark...")
    df = add_context_speed_z(df)
    print("4/12 Horse Elo...")
    df = add_horse_elo(df)
    print("5/12 Current field context...")
    df = add_basic_field_features(df)
    print("6/12 Horse prior-start history...")
    df = add_horse_history(df, max_lag=history_lags)
    print("7/12 Horse track/distance/surface/class history...")
    df = add_horse_context_stats(df)
    print("8/12 Trainer historical form...")
    df = add_entity_daily_form(df, "trainer", "trainer")
    print("9/12 Jockey historical form...")
    df = add_entity_daily_form(df, "jockey", "jockey")
    print("10/12 Horse-jockey / trainer-jockey / contextual combinations...")
    df = add_daily_combo_stats(df, ["horse_id", "jockey"], "horse_jockey")
    df = add_daily_combo_stats(df, ["trainer", "jockey"], "trainer_jockey")
    for entity in ("trainer", "jockey"):
        for context, short_name in [
            ("racecourse_code", "track"),
            ("distance_m", "distance"),
            ("race_class", "class"),
            ("surface", "surface"),
        ]:
            df = add_daily_combo_stats(df, [entity, context], f"{entity}_{short_name}")
    print("11/12 Relative-to-field historical features...")
    df = add_post_history_field_features(df)
    if include_market_features:
        print("     Adding current-race market features...")
        df = add_market_features(df)
    print("12/12 Training metadata and output filtering...")
    df = add_training_metadata(df)

    # Critical: calculate on FULL history first; filter only at the very end.
    df = df.loc[df["_emit_row"] == 1].copy()

    # Restore the row order of the filtered source file exactly.
    emit_order = pd.Series(np.arange(len(emit_df)), index=emit_df["result_id"].astype(str))
    df["_emit_order"] = df["result_id"].astype(str).map(emit_order)
    df = df.sort_values("_emit_order", kind="mergesort").drop(columns=["_emit_order"]).reset_index(drop=True)

    if not keep_helpers:
        df = df.drop(columns=[c for c in INTERNAL_COLUMNS if c in df.columns])

    # Format dates for CSV readability.
    df["race_date"] = pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d")
    for i in range(1, history_lags + 1):
        col = f"last{i}_date"
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")

    # Keep original cleaned-file columns first, engineered columns after them.
    original = [c for c in original_emit_columns if c in df.columns]
    engineered = [c for c in df.columns if c not in original]
    df = df[original + engineered]
    generated = engineered
    return df, generated


def build_predictor_lists(df: pd.DataFrame, include_market_features: bool) -> tuple[list[str], list[str]]:
    forbidden = set(NEVER_PREDICTOR_COLUMNS)
    forbidden.update({"ml_target_is_winner", "ml_group_id", "ml_split", "ml_row_weight"})

    # Raw odds are excluded unless the user explicitly opts into market features.
    if not include_market_features:
        forbidden.add("odds")
        forbidden.update({c for c in df.columns if c.startswith("market_")})

    # lastN_* columns are prior-race facts, so they are allowed even when they contain words like
    # finish_time or horse_rating_after; the prefix guarantees they refer to a PRIOR race.
    predictors = []
    for col in df.columns:
        if col in forbidden:
            continue
        if col.startswith("_"):
            continue
        if col in {TARGET_COLUMN}:
            continue
        predictors.append(col)

    categorical_candidates = [
        "racecourse_code", "racecourse_name", "race_class", "rating_band", "going", "surface",
        "course", "country_of_origin", "hemisphere_of_origin", "hemisphere_of_origin_derived",
        "horse_colour", "horse_sex", "sire", "dam", "dam_sire", "jockey", "trainer",
        "distance_bucket", "going_bucket", "form_last5_latest_first",
    ]
    categorical = [c for c in categorical_candidates if c in predictors]
    categorical += [
        c for c in predictors
        if re.fullmatch(r"last\d+_(race_class|track|surface|course|going|jockey|trainer)", c)
    ]
    categorical = sorted(set(categorical))
    return predictors, categorical


def write_manifest(
    path: Path,
    df: pd.DataFrame,
    generated: Sequence[str],
    include_market: bool,
    history_path: Path,
    rows_path: Path,
    output_path: Path,
) -> None:
    predictors, categorical = build_predictor_lists(df, include_market)
    manifest = {
        "description": "HKJC point-in-time CatBoost-ready feature manifest",
        "target": TARGET_COLUMN,
        "preferred_target_alias": "ml_target_is_winner",
        "group_id": "race_id",
        "preferred_group_alias": "ml_group_id",
        "history_source": str(history_path),
        "emitted_rows_source": str(rows_path),
        "output": str(output_path),
        "generated_feature_count": len(generated),
        "generated_columns": list(generated),
        "recommended_predictor_count": len(predictors),
        "recommended_predictor_columns": predictors,
        "recommended_categorical_columns": categorical,
        "never_use_as_current_race_predictors": NEVER_PREDICTOR_COLUMNS,
        "market_features_enabled": include_market,
        "history_rule": (
            "Features are computed on the complete all_results.csv timeline. "
            "Only after feature creation are rows filtered to all_results_with_races_removed.csv."
        ),
        "same_day_rule": (
            "Trainer/jockey daily-form features exclude all outcomes from the current race date. "
            "Horse last-start features use strictly earlier event_order rows."
        ),
        "recommended_split": {
            "train": "race_date <= 2024-07-31",
            "validation": "2024-08-01 to 2025-07-31",
            "test": "race_date >= 2025-08-01",
        },
        "warnings": [
            "Do not use current-race finishing_position, is_top_three, margin, finish_time, horse_rating_after, prize payout fields, or other outcome columns as predictors.",
            "Do not enable market features unless the odds values are known to be available at the model prediction cutoff.",
            "Prize-money-derived history should be treated cautiously until historical payout allocation logic is fully audited/corrected.",
            "Profile-snapshot fields that can change over time must not be assumed historically valid merely because they exist in the current horse profile.",
        ],
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def validate_output(df: pd.DataFrame, emit_df: pd.DataFrame) -> None:
    if len(df) != len(emit_df):
        raise ValueError(f"Output row count {len(df):,} does not equal emit row count {len(emit_df):,}")
    if df["result_id"].duplicated().any():
        raise ValueError("Output contains duplicate result_id values")
    if not df["result_id"].astype(str).equals(emit_df["result_id"].astype(str).reset_index(drop=True)):
        raise ValueError("Output result_id ordering does not match the cleaned source file")
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    bad_target = target.dropna()[~target.dropna().isin([0, 1])]
    if len(bad_target):
        raise ValueError("is_winner contains values other than 0/1")


def main():
    args = parse_args()
    history_path = Path(args.history)
    rows_path = Path(args.rows)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(output_path.suffix + ".features.json")

    print(f"Reading complete history: {history_path}")
    history_df = pd.read_csv(history_path, low_memory=False)
    print(f"  rows={len(history_df):,}, columns={len(history_df.columns):,}")

    print(f"Reading rows to emit:      {rows_path}")
    emit_df = pd.read_csv(rows_path, low_memory=False)
    print(f"  rows={len(emit_df):,}, columns={len(emit_df.columns):,}")

    enriched, generated = build_features(
        history_df,
        emit_df,
        include_market_features=args.include_market_features,
        keep_helpers=args.keep_helper_columns,
        history_lags=max(1, args.history_lags),
    )
    validate_output(enriched, emit_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing ML-ready CSV: {output_path}")
    enriched.to_csv(output_path, index=False)

    write_manifest(
        manifest_path,
        enriched,
        generated,
        args.include_market_features,
        history_path,
        rows_path,
        output_path,
    )

    predictors, categorical = build_predictor_lists(enriched, args.include_market_features)
    print("\nCOMPLETE")
    print(f"Output rows:               {len(enriched):,}")
    print(f"Output columns:            {len(enriched.columns):,}")
    print(f"Engineered columns added:  {len(generated):,}")
    print(f"Recommended predictors:    {len(predictors):,}")
    print(f"Categorical predictors:    {len(categorical):,}")
    print(f"Target:                    {TARGET_COLUMN}")
    print(f"Output:                    {output_path}")
    print(f"Manifest:                  {manifest_path}")
    print("\nIMPORTANT:")
    print("- Features are built from COMPLETE all_results.csv history, then filtered to cleaned rows.")
    print("- Do not use current-race outcome columns as predictors.")
    print("- Keep odds disabled unless you have confirmed the odds timestamp matches prediction time.")


if __name__ == "__main__":
    main()
