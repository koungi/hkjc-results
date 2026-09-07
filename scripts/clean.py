#!/usr/bin/env python3

import csv
import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = REPO_ROOT / "results" / "races" / "all_results.csv"
OUTPUT_FILE = REPO_ROOT / "results" / "races" / "all_results_cleaned.csv"

COLUMN_NAME = "dam_sire"
TRAILING_TEXT = " Same"


def clean_dam_sire(value: str) -> tuple[str, bool]:
    if value.endswith(TRAILING_TEXT):
        return value[: -len(TRAILING_TEXT)], True

    return value, False


def main() -> int:
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file does not exist: {INPUT_FILE}", file=sys.stderr)
        return 1

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    changed_rows = 0

    temp_fd, temp_name = tempfile.mkstemp(
        prefix="all_results_cleaned_",
        suffix=".csv",
        dir=OUTPUT_FILE.parent,
    )

    os.close(temp_fd)
    temp_path = Path(temp_name)

    try:
        with INPUT_FILE.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as infile, temp_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as outfile:

            reader = csv.DictReader(infile)

            if reader.fieldnames is None:
                raise ValueError("CSV has no header row.")

            if COLUMN_NAME not in reader.fieldnames:
                raise ValueError(
                    f"Required column '{COLUMN_NAME}' was not found.\n"
                    f"Available columns: {reader.fieldnames}"
                )

            writer = csv.DictWriter(
                outfile,
                fieldnames=reader.fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )

            writer.writeheader()

            for row in reader:
                total_rows += 1

                original_value = row.get(COLUMN_NAME, "")

                if original_value is None:
                    original_value = ""

                cleaned_value, changed = clean_dam_sire(original_value)

                if changed:
                    changed_rows += 1

                row[COLUMN_NAME] = cleaned_value
                writer.writerow(row)

        os.replace(temp_path, OUTPUT_FILE)

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    input_size_mb = INPUT_FILE.stat().st_size / (1024 * 1024)
    output_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

    print("")
    print("Cleaning complete")
    print("-----------------")
    print(f"Input file:       {INPUT_FILE}")
    print(f"Output file:      {OUTPUT_FILE}")
    print(f"Rows processed:   {total_rows:,}")
    print(f"Rows changed:     {changed_rows:,}")
    print(f"Rows unchanged:   {total_rows - changed_rows:,}")
    print(f"Input size:       {input_size_mb:.2f} MB")
    print(f"Output size:      {output_size_mb:.2f} MB")
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
