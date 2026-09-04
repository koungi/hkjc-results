import os
import re
import time
from datetime import datetime, timedelta
from io import StringIO
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIG
# ============================================================

BASE_URL = (
    "https://racing.hkjc.com/en-us/local/information/"
    "archive/localresults"
)

START_DATE = os.getenv("START_DATE", "2006-12-01")
END_DATE = os.getenv("END_DATE", "2006-12-31")

DELAY_SECONDS = 2

RESULTS_DIR = "results"
RACES_DIR = os.path.join(RESULTS_DIR, "races")
HORSES_DIR = os.path.join(RESULTS_DIR, "horses")

RACE_RESULTS_FILE = os.path.join(
    RACES_DIR,
    "all_results.csv"
)

HORSE_MASTER_FILE = os.path.join(
    HORSES_DIR,
    "horse_master.csv"
)


session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
})


# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_folders():
    os.makedirs(
        RACES_DIR,
        exist_ok=True
    )

    os.makedirs(
        HORSES_DIR,
        exist_ok=True
    )


def date_range(start_date, end_date):
    current = start_date

    while current <= end_date:
        yield current
        current += timedelta(days=1)


def clean_text(value):
    if value is None:
        return ""

    return (
        str(value)
        .replace("\xa0", " ")
        .strip()
    )


def parse_integer(value):
    if value is None:
        return None

    text = clean_text(value)

    match = re.search(
        r"-?\d+",
        text
    )

    if not match:
        return None

    try:
        return int(
            match.group()
        )
    except ValueError:
        return None


def parse_float(value):
    if value is None:
        return None

    text = clean_text(value)

    text = text.replace(
        ",",
        ""
    )

    try:
        return float(text)
    except ValueError:
        return None


def parse_prize_money(value):
    if not value:
        return None

    text = clean_text(value)

    text = (
        text
        .replace("HK$", "")
        .replace("$", "")
        .replace(",", "")
        .strip()
    )

    try:
        return int(
            float(text)
        )
    except ValueError:
        return None


def parse_distance(value):
    if not value:
        return None

    match = re.search(
        r"(\d+)\s*M",
        str(value),
        re.I
    )

    if not match:
        return None

    return int(
        match.group(1)
    )


def extract_horse_id(href):
    if not href:
        return ""

    match = re.search(
        r"horseid=([^&]+)",
        href,
        re.I
    )

    if match:
        return match.group(1)

    match = re.search(
        r"HorseID=([^&]+)",
        href,
        re.I
    )

    if match:
        return match.group(1)

    return ""


# ============================================================
# URL / REQUEST
# ============================================================

def build_url(
    race_date,
    racecourse=None,
    race_no=None
):
    date_string = race_date.strftime(
        "%Y/%m/%d"
    )

    url = (
        f"{BASE_URL}"
        f"?racedate={date_string}"
    )

    if racecourse:
        url += (
            f"&racecourse="
            f"{racecourse}"
        )

    if race_no:
        url += (
            f"&RaceNo="
            f"{race_no}"
        )

    return url


def request_page(
    race_date,
    racecourse=None,
    race_no=None
):
    url = build_url(
        race_date,
        racecourse,
        race_no
    )

    try:
        response = session.get(
            url,
            timeout=60
        )

        print(
            f"GET {url} "
            f"-> {response.status_code}"
        )

        response.raise_for_status()

        return response

    except requests.RequestException as exc:
        print(
            "Request failed:",
            exc
        )

        return None


# ============================================================
# MEETING DETECTION
# ============================================================

def detect_meeting(html):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    element = soup.select_one(
        ".raceMeeting_select"
    )

    if element is None:
        return None

    text = element.get_text(
        " ",
        strip=True
    )

    if "Happy Valley" in text:
        return {
            "racecourse_code": "HV",
            "racecourse_name": "Happy Valley"
        }

    if "Sha Tin" in text:
        return {
            "racecourse_code": "ST",
            "racecourse_name": "Sha Tin"
        }

    return None


def detect_race_numbers(html):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    race_numbers = set()

    # Current Race 1 may not be a link.
    text_match = soup.find(
        string=lambda text:
        text and "RACE 1" in text.upper()
    )

    if text_match:
        race_numbers.add(1)

    for link in soup.select(
        'a[href*="RaceNo="]'
    ):
        href = link.get(
            "href",
            ""
        )

        match = re.search(
            r"RaceNo=(\d+)",
            href,
            re.I
        )

        if match:
            race_numbers.add(
                int(
                    match.group(1)
                )
            )

    return sorted(
        race_numbers
    )


# ============================================================
# RACE HEADER
# ============================================================

def extract_race_metadata(
    html,
    meeting_date,
    racecourse_code,
    racecourse_name,
    race_no,
    race_url
):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    table = soup.select_one(
        ".race_tab table"
    )

    metadata = {
        "race_id": "",
        "race_date": (
            meeting_date
            .strftime("%Y-%m-%d")
        ),
        "racecourse_code": (
            racecourse_code
        ),
        "racecourse_name": (
            racecourse_name
        ),
        "race_number": race_no,
        "race_index": None,
        "race_name": "",
        "race_class": "",
        "distance_m": None,
        "rating_band": "",
        "going": "",
        "surface": "",
        "course": "",
        "prize_money_hkd": None,
        "race_url": race_url,
    }

    metadata["race_id"] = (
        f"{racecourse_code}_"
        f"{meeting_date.strftime('%Y%m%d')}_"
        f"R{race_no:02d}"
    )

    if table is None:
        return metadata

    rows = table.find_all(
        "tr"
    )

    # --------------------------------------
    # RACE 2 (212)
    # --------------------------------------

    if len(rows) >= 1:
        text = rows[0].get_text(
            " ",
            strip=True
        )

        match = re.search(
            r"RACE\s+\d+\s+\((\d+)\)",
            text,
            re.I
        )

        if match:
            metadata["race_index"] = int(
                match.group(1)
            )

    # --------------------------------------
    # Class 5 - 1650M - (40-20)
    # Going: GOOD TO FIRM
    # --------------------------------------

    if len(rows) >= 2:
        cells = rows[1].find_all(
            "td"
        )

        if cells:
            left = cells[0].get_text(
                " ",
                strip=True
            )

            parts = [
                p.strip()
                for p in left.split("-")
            ]

            if len(parts) >= 1:
                metadata["race_class"] = (
                    parts[0]
                )

            metadata["distance_m"] = (
                parse_distance(left)
            )

            match = re.search(
                r"\(([^)]+)\)",
                left
            )

            if match:
                metadata["rating_band"] = (
                    match.group(1)
                )

            if len(cells) >= 3:
                metadata["going"] = (
                    cells[-1]
                    .get_text(
                        " ",
                        strip=True
                    )
                )

    # --------------------------------------
    # PACIFIC OCEAN HANDICAP
    # TURF - "A" Course
    # --------------------------------------

    if len(rows) >= 3:
        cells = rows[2].find_all(
            "td"
        )

        if cells:
            metadata["race_name"] = (
                cells[0]
                .get_text(
                    " ",
                    strip=True
                )
            )

            if len(cells) >= 3:
                course_text = (
                    cells[-1]
                    .get_text(
                        " ",
                        strip=True
                    )
                )

                metadata["course"] = (
                    course_text
                )

                if "TURF" in course_text.upper():
                    metadata["surface"] = (
                        "TURF"
                    )

                elif "ALL WEATHER" in course_text.upper():
                    metadata["surface"] = (
                        "ALL WEATHER"
                    )

    # --------------------------------------
    # HK$ 450,000
    # --------------------------------------

    if len(rows) >= 4:
        cells = rows[3].find_all(
            "td"
        )

        if cells:
            metadata[
                "prize_money_hkd"
            ] = parse_prize_money(
                cells[0].get_text(
                    " ",
                    strip=True
                )
            )

    return metadata


# ============================================================
# RESULT TABLE
# ============================================================

def extract_results(
    html,
    race_metadata
):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    table = soup.select_one(
        ".performance table"
    )

    if table is None:
        return None

    rows = table.select(
        "tbody tr"
    )

    results = []

    field_size = len(rows)

    for row in rows:
        cells = row.find_all(
            "td"
        )

        if len(cells) < 10:
            continue

        horse_link = row.select_one(
            'a[href*="horse"]'
        )

        horse_id = ""

        if horse_link:
            horse_id = extract_horse_id(
                horse_link.get(
                    "href",
                    ""
                )
            )

        finishing_position = (
            parse_integer(
                cells[0].get_text(
                    " ",
                    strip=True
                )
            )
        )

        horse_number = (
            parse_integer(
                cells[1].get_text(
                    " ",
                    strip=True
                )
            )
        )

        horse_name = ""

        if horse_link:
            horse_name = (
                horse_link
                .get_text(
                    " ",
                    strip=True
                )
            )
        else:
            horse_name = (
                cells[2]
                .get_text(
                    " ",
                    strip=True
                )
            )

        jockey = clean_text(
            cells[3].get_text(
                " ",
                strip=True
            )
        )

        trainer = clean_text(
            cells[4].get_text(
                " ",
                strip=True
            )
        )

        actual_weight = (
            parse_integer(
                cells[5].get_text(
                    " ",
                    strip=True
                )
            )
        )

        declared_horse_weight = (
            parse_integer(
                cells[6].get_text(
                    " ",
                    strip=True
                )
            )
        )

        draw = parse_integer(
            cells[7].get_text(
                " ",
                strip=True
            )
        )

        margin = clean_text(
            cells[8].get_text(
                " ",
                strip=True
            )
        )

        finish_time = clean_text(
            cells[9].get_text(
                " ",
                strip=True
            )
        )

        odds = None

        if len(cells) >= 11:
            odds = parse_float(
                cells[10].get_text(
                    " ",
                    strip=True
                )
            )

        result_id = (
            f"{race_metadata['race_id']}_"
            f"{horse_id or horse_number}"
        )

        record = {
            "result_id": result_id,
            "race_id": race_metadata[
                "race_id"
            ],
            "race_date": race_metadata[
                "race_date"
            ],
            "racecourse_code": (
                race_metadata[
                    "racecourse_code"
                ]
            ),
            "racecourse_name": (
                race_metadata[
                    "racecourse_name"
                ]
            ),
            "race_number": (
                race_metadata[
                    "race_number"
                ]
            ),
            "race_index": (
                race_metadata[
                    "race_index"
                ]
            ),
            "race_name": (
                race_metadata[
                    "race_name"
                ]
            ),
            "race_class": (
                race_metadata[
                    "race_class"
                ]
            ),
            "distance_m": (
                race_metadata[
                    "distance_m"
                ]
            ),
            "rating_band": (
                race_metadata[
                    "rating_band"
                ]
            ),
            "going": (
                race_metadata[
                    "going"
                ]
            ),
            "surface": (
                race_metadata[
                    "surface"
                ]
            ),
            "course": (
                race_metadata[
                    "course"
                ]
            ),
            "prize_money_hkd": (
                race_metadata[
                    "prize_money_hkd"
                ]
            ),
            "field_size": field_size,

            "horse_id": horse_id,
            "horse_number": horse_number,
            "horse_name": horse_name,

            "finishing_position": (
                finishing_position
            ),
            "jockey": jockey,
            "trainer": trainer,
            "actual_weight": (
                actual_weight
            ),
            "declared_horse_weight": (
                declared_horse_weight
            ),
            "draw": draw,
            "margin": margin,
            "finish_time": finish_time,
            "odds": odds,

            "race_url": (
                race_metadata[
                    "race_url"
                ]
            ),
        }

        results.append(
            record
        )

    if not results:
        return None

    return pd.DataFrame(
        results
    )


# ============================================================
# HORSE MASTER
# ============================================================

def load_known_horses():
    if not os.path.exists(
        HORSE_MASTER_FILE
    ):
        return set()

    try:
        df = pd.read_csv(
            HORSE_MASTER_FILE,
            dtype=str
        )

        if "horse_id" not in df.columns:
            return set()

        return set(
            df["horse_id"]
            .dropna()
            .astype(str)
        )

    except Exception:
        return set()


def append_new_horses_from_results(
    results_df,
    known_horses
):
    if results_df is None:
        return

    new_rows = []

    for _, row in results_df.iterrows():
        horse_id = clean_text(
            row.get(
                "horse_id",
                ""
            )
        )

        if not horse_id:
            continue

        if horse_id in known_horses:
            continue

        new_rows.append({
            "horse_id": horse_id,
            "horse_name": clean_text(
                row.get(
                    "horse_name",
                    ""
                )
            ),

            # These can be populated later
            # from the horse profile page.
            "owner": "",
            "colour": "",
            "sex": "",
            "country_of_origin": "",
            "import_type": "",
            "sire": "",
            "dam": "",
            "dam_sire": "",
        })

        known_horses.add(
            horse_id
        )

    if not new_rows:
        return

    new_df = pd.DataFrame(
        new_rows
    )

    file_exists = os.path.exists(
        HORSE_MASTER_FILE
    )

    new_df.to_csv(
        HORSE_MASTER_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )

    print(
        f"Added {len(new_df)} "
        "new horses to horse_master.csv"
    )


# ============================================================
# SAVE RACE RESULTS
# ============================================================

def load_existing_result_ids():
    if not os.path.exists(
        RACE_RESULTS_FILE
    ):
        return set()

    try:
        df = pd.read_csv(
            RACE_RESULTS_FILE,
            usecols=[
                "result_id"
            ],
            dtype=str
        )

        return set(
            df["result_id"]
            .dropna()
            .astype(str)
        )

    except Exception:
        return set()


def append_results(
    results_df,
    existing_ids
):
    if results_df is None:
        return

    results_df = results_df[
        ~results_df[
            "result_id"
        ].isin(
            existing_ids
        )
    ]

    if results_df.empty:
        print(
            "No new result rows."
        )

        return

    file_exists = os.path.exists(
        RACE_RESULTS_FILE
    )

    results_df.to_csv(
        RACE_RESULTS_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )

    existing_ids.update(
        results_df[
            "result_id"
        ].astype(str)
    )

    print(
        f"Added {len(results_df)} "
        "race-result rows."
    )


# ============================================================
# PROCESS DATE
# ============================================================

def process_date(
    meeting_date,
    existing_result_ids,
    known_horses
):
    print()
    print(
        "=" * 60
    )
    print(
        "Checking:",
        meeting_date.strftime(
            "%Y/%m/%d"
        )
    )
    print(
        "=" * 60
    )

    response = request_page(
        meeting_date
    )

    if response is None:
        return

    meeting = detect_meeting(
        response.text
    )

    if meeting is None:
        print(
            "No HKJC meeting detected."
        )
        return

    racecourse_code = (
        meeting[
            "racecourse_code"
        ]
    )

    racecourse_name = (
        meeting[
            "racecourse_name"
        ]
    )

    print(
        "Meeting:",
        racecourse_name
    )

    race_numbers = detect_race_numbers(
        response.text
    )

    if not race_numbers:
        print(
            "No races detected."
        )
        return

    print(
        "Races:",
        race_numbers
    )

    for race_no in race_numbers:

        race_url = build_url(
            meeting_date,
            racecourse_code,
            race_no
        )

        print(
            f"Processing Race {race_no}"
        )

        race_response = request_page(
            meeting_date,
            racecourse_code,
            race_no
        )

        if race_response is None:
            continue

        metadata = extract_race_metadata(
            race_response.text,
            meeting_date,
            racecourse_code,
            racecourse_name,
            race_no,
            race_url
        )

        results_df = extract_results(
            race_response.text,
            metadata
        )

        if results_df is None:
            print(
                "No horse results found."
            )

            continue

        append_new_horses_from_results(
            results_df,
            known_horses
        )

        append_results(
            results_df,
            existing_result_ids
        )

        time.sleep(
            DELAY_SECONDS
        )


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_folders()

    try:
        start_date = datetime.strptime(
            START_DATE,
            "%Y-%m-%d"
        ).date()

        end_date = datetime.strptime(
            END_DATE,
            "%Y-%m-%d"
        ).date()

    except ValueError:
        print(
            "Dates must use "
            "YYYY-MM-DD format."
        )
        return

    if start_date > end_date:
        print(
            "START_DATE cannot be "
            "after END_DATE."
        )
        return

    existing_result_ids = (
        load_existing_result_ids()
    )

    known_horses = (
        load_known_horses()
    )

    print()
    print(
        "HKJC Historical Results Collector"
    )

    print(
        "Start:",
        start_date
    )

    print(
        "End:",
        end_date
    )

    print(
        "Existing race results:",
        len(
            existing_result_ids
        )
    )

    print(
        "Known horses:",
        len(
            known_horses
        )
    )

    for meeting_date in date_range(
        start_date,
        end_date
    ):

        process_date(
            meeting_date,
            existing_result_ids,
            known_horses
        )

        time.sleep(
            DELAY_SECONDS
        )


if __name__ == "__main__":
    main()
