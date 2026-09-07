#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ML-ready racing CSV to Parquet."
    )

    parser.add_argument(
        "--input",
        default="results/races/all_results_ML_ready.csv",
        help="Input CSV path",
    )

    parser.add_argument(
        "--output",
        default="results/races/all_results_ML_ready.parquet",
        help="Output Parquet path",
    )

    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "none"],
        help="Parquet compression method",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Reading CSV: {input_path}")

    df = pd.read_csv(
        input_path,
        low_memory=False,
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    compression = (
        None
        if args.compression == "none"
        else args.compression
    )

    print(
        f"Writing Parquet: {output_path}"
    )

    df.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression=compression,
    )

    print()
    print("Validating Parquet...")

    test = pd.read_parquet(
        output_path,
        engine="pyarrow",
    )

    if len(test) != len(df):
        raise RuntimeError(
            "Row count changed during conversion"
        )

    if list(test.columns) != list(df.columns):
        raise RuntimeError(
            "Column order changed during conversion"
        )

    csv_size = input_path.stat().st_size
    parquet_size = output_path.stat().st_size

    print()
    print("COMPLETE")
    print(f"CSV size:     {csv_size / 1024 / 1024:,.1f} MB")
    print(f"Parquet size: {parquet_size / 1024 / 1024:,.1f} MB")

    if csv_size > 0:
        reduction = (
            1 - parquet_size / csv_size
        ) * 100

        print(
            f"Size reduction: {reduction:.1f}%"
        )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()
