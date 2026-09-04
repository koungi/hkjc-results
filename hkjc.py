import os
import re
import time
from datetime import datetime, timedelta

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

START_DATE = os.getenv("START_DATE", "2006-01-01")
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


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})


# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_folders():
    os.makedirs(RACES_DIR, exist_ok=True)
    os.makedirs(HORSES_DIR, exist_ok=True)


def date_range(start_date, end_date):
    current = start_date

    while current <= end_date:
        yield current
        current += timedelta(days=1)


def clean_text(value):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


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
        return int(match.group())
    except ValueError:
        return None


def parse_float(value):
    if value is None:
        return None

    text = clean_text(value)
    text = text.replace(",", "")

    try:
        return float(text)
    except ValueError:
        return None


def parse_prize_money(value):
    if not value:
        return None

    match = re.search(
        r"HK\$\s*([\d,]+)",
        clean_text(value),
        re.I
    )

    if not match:
        return None

    try:
        return int(
            match.group(1).replace(",", "")
        )
    except ValueError:
        return None


def parse_distance(value):
    if not value:
        return None

    match = re.search(
        r"(\d{3,4})\s*M\b",
        clean_text(value),
        re.I
    )

    if not match:
        return None

    return int(match.group(1))


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

    return ""


# ============================================================
# URL BUILDING
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
            f"&racecourse={racecourse}"
        )

    if race_no:
        url += (
            f"&RaceNo={race_no}"
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

    text = clean_text(
        element.get_text(
            " ",
            strip=True
        )
    )

    if "Happy Valley" in text:
        return {
            "racecourse_code": "HV",
            "racecourse_name": "Happy Valley",
        }

    if "Sha Tin" in text:
        return {
            "racecourse_code": "ST",
            "racecourse_name": "Sha Tin",
        }

    return None


# ============================================================
# DETECT RACES
# ============================================================

def detect_race_numbers(html):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    race_numbers = set()

    # Detect current race from text
    for text in soup.stripped_strings:

        match = re.search(
            r"\bRACE\s+(\d+)\b",
            text,
            re.I
        )

        if match:
            race_numbers.add(
                int(match.group(1))
            )

    # Detect other races from navigation links
    for link in soup.find_all(
        "a",
        href=True
    ):
        href = link.get(
            "href",
            ""
        )

        match = re.search(
            r"[?&]RaceNo=(\d+)",
            href,
            re.I
        )

        if match:
            race_numbers.add(
                int(match.group(1))
            )

    return sorted(race_numbers)


# ============================================================
# HEADER HELPERS
# ============================================================

def get_cell_texts(row):
    return [
        clean_text(
            cell.get_text(
                " ",
                strip=True
            )
        )
        for cell in row.find_all(
            ["td", "th"]
        )
    ]


def extract_label_value(
    cells,
    label
):
    """
    Handles either:

        Going :
        GOOD TO FIRM

    or:

        Going : GOOD TO FIRM
    """

    pattern = re.compile(
        rf"\b{re.escape(label)}\s*:",
        re.I
    )

    for index, cell in enumerate(cells):

        if not pattern.search(cell):
            continue

        # Value may be in same cell
        same_cell = pattern.sub(
            "",
            cell
        ).strip()

        if same_cell:
            return same_cell

        # Or in the next non-empty cell
        for next_index in range(
            index + 1,
            len(cells)
        ):
            if cells[next_index]:
                return cells[next_index]

    return ""


# ============================================================
# RACE HEADER PARSER
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

    metadata = {
        "race_id": (
            f"{racecourse_code}_"
            f"{meeting_date.strftime('%Y%m%d')}_"
            f"R{race_no:02d}"
        ),
        "race_date": (
            meeting_date.strftime(
                "%Y-%m-%d"
            )
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

    # IMPORTANT:
    # Historical pages can have race_tab directly
    # on the table OR around the table.
    header = soup.select_one(
        "table.race_tab"
    )

    if header is None:
        header = soup.select_one(
            ".race_tab"
        )

    if header is None:
        print(
            "WARNING: Race header "
            "could not be located."
        )
        return metadata

    rows = header.find_all("tr")

    full_text = clean_text(
        header.get_text(
            " ",
            strip=True
        )
    )

    print()
    print(
        "========== RACE HEADER =========="
    )

    print(
        "HEADER:",
        full_text
    )

    # ========================================================
    # RACE NUMBER / RACE INDEX
    #
    # RACE 1 (211)
    # ========================================================

    race_match = re.search(
        r"\bRACE\s+(\d+)\s*"
        r"\(\s*(\d+)\s*\)",
        full_text,
        re.I
    )

    if race_match:

        metadata["race_number"] = int(
            race_match.group(1)
        )

        metadata["race_index"] = int(
            race_match.group(2)
        )

    # ========================================================
    # PROCESS ROWS
    # ========================================================

    for row_number, row in enumerate(rows):

        cells = get_cell_texts(row)

        if not cells:
            continue

        row_text = clean_text(
            " | ".join(cells)
        )

        print(
            f"HEADER ROW {row_number}:",
            cells
        )

        # ----------------------------------------------------
        # CLASS + DISTANCE + RATING BAND
        #
        # Example:
        # Class 4 - 1200M - (60-40)
        # ----------------------------------------------------

        for cell in cells:

            if not re.search(
                r"\bClass\s+\d+",
                cell,
                re.I
            ):
                continue

            class_match = re.search(
                r"\b(Class\s+\d+)\b",
                cell,
                re.I
            )

            if class_match:
                metadata[
                    "race_class"
                ] = class_match.group(
                    1
                ).title()

            distance_match = re.search(
                r"\b(\d{3,4})\s*M\b",
                cell,
                re.I
            )

            if distance_match:
                metadata[
                    "distance_m"
                ] = int(
                    distance_match.group(1)
                )

            rating_match = re.search(
                r"\(\s*(\d+)"
                r"\s*-\s*"
                r"(\d+)\s*\)",
                cell
            )

            if rating_match:

                metadata[
                    "rating_band"
                ] = (
                    f"{rating_match.group(1)}-"
                    f"{rating_match.group(2)}"
                )

        # ----------------------------------------------------
        # GOING
        # ----------------------------------------------------

        going = extract_label_value(
            cells,
            "Going"
        )

        if going:
            metadata["going"] = going

        # ----------------------------------------------------
        # COURSE
        # ----------------------------------------------------

        course = extract_label_value(
            cells,
            "Course"
        )

        if course:
            metadata["course"] = course

        # ----------------------------------------------------
        # RACE NAME
        #
        # Normally the first cell on the row
        # containing Course :
        # ----------------------------------------------------

        contains_course = any(
            re.search(
                r"\bCourse\s*:",
                cell,
                re.I
            )
            for cell in cells
        )

        if contains_course:

            course_position = None

            for index, cell in enumerate(
                cells
            ):

                if re.search(
                    r"\bCourse\s*:",
                    cell,
                    re.I
                ):
                    course_position = index
                    break

            if course_position is not None:

                candidates = cells[
                    :course_position
                ]

                for candidate in candidates:

                    candidate = clean_text(
                        candidate
                    )

                    if not candidate:
                        continue

                    # Don't accidentally use
                    # class/distance as race name
                    if re.search(
                        r"\bClass\s+\d+",
                        candidate,
                        re.I
                    ):
                        continue

                    if re.search(
                        r"\bRACE\s+\d+",
                        candidate,
                        re.I
                    ):
                        continue

                    if re.search(
                        r"\bGoing\s*:",
                        candidate,
                        re.I
                    ):
                        continue

                    metadata[
                        "race_name"
                    ] = candidate

                    break

        # ----------------------------------------------------
        # PRIZE MONEY
        # ----------------------------------------------------

        for cell in cells:

            prize = parse_prize_money(
                cell
            )

            if prize is not None:
                metadata[
                    "prize_money_hkd"
                ] = prize

    # ========================================================
    # FALLBACK CLASS / DISTANCE / RATING
    # ========================================================

    if not metadata["race_class"]:

        match = re.search(
            r"\b(Class\s+\d+)\b",
            full_text,
            re.I
        )

        if match:
            metadata[
                "race_class"
            ] = match.group(1).title()

    if metadata["distance_m"] is None:

        match = re.search(
            r"\b(\d{3,4})\s*M\b",
            full_text,
            re.I
        )

        if match:
            metadata[
                "distance_m"
            ] = int(
                match.group(1)
            )

    if not metadata["rating_band"]:

        match = re.search(
            r"\(\s*(\d+)"
            r"\s*-\s*"
            r"(\d+)\s*\)",
            full_text
        )

        if match:
            metadata[
                "rating_band"
            ] = (
                f"{match.group(1)}-"
                f"{match.group(2)}"
            )

    # ========================================================
    # FALLBACK PRIZE MONEY
    # ========================================================

    if metadata[
        "prize_money_hkd"
    ] is None:

        metadata[
            "prize_money_hkd"
        ] = parse_prize_money(
            full_text
        )

    # ========================================================
    # FALLBACK GOING
    # ========================================================

    if not metadata["going"]:

        going_match = re.search(
            r"Going\s*:\s*"
            r"([A-Z][A-Z ]+?)"
            r"(?=\s+(?:"
            r"Course\s*:|"
            r"HK\$|"
            r"Time\s*:|"
            r"[A-Z][A-Z '&.-]+"
            r"\s+(?:HANDICAP|CUP|STAKES)"
            r"))",
            full_text,
            re.I
        )

        if going_match:
            metadata["going"] = clean_text(
                going_match.group(1)
            )

    # ========================================================
    # SURFACE
    # ========================================================

    course_upper = metadata[
        "course"
    ].upper()

    if "TURF" in course_upper:

        metadata["surface"] = (
            "TURF"
        )

    elif (
        "ALL WEATHER" in course_upper
        or "AWT" in course_upper
    ):

        metadata["surface"] = (
            "ALL WEATHER TRACK"
        )

    # ========================================================
    # CLEAN VALUES
    # ========================================================

    metadata["going"] = re.sub(
        r"^Going\s*:\s*",
        "",
        metadata["going"],
        flags=re.I
    ).strip()

    metadata["course"] = re.sub(
        r"^Course\s*:\s*",
        "",
        metadata["course"],
        flags=re.I
    ).strip()

    # ========================================================
    # DEBUG OUTPUT
    # ========================================================

    print()
    print(
        "PARSED RACE METADATA:"
    )

    print(
        " race_number:",
        metadata["race_number"]
    )

    print(
        " race_index:",
        metadata["race_index"]
    )

    print(
        " race_name:",
        metadata["race_name"]
    )

    print(
        " race_class:",
        metadata["race_class"]
    )

    print(
        " distance_m:",
        metadata["distance_m"]
    )

    print(
        " rating_band:",
        metadata["rating_band"]
    )

    print(
        " going:",
        metadata["going"]
    )

    print(
        " surface:",
        metadata["surface"]
    )

    print(
        " course:",
        metadata["course"]
    )

    print(
        " prize_money_hkd:",
        metadata[
            "prize_money_hkd"
        ]
    )

    print(
        "================================="
    )
    print()

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

    # Historical pages may use either structure
    table = soup.select_one(
        "table.performance"
    )

    if table is None:
        table = soup.select_one(
            ".performance table"
        )

    if table is None:
        print(
            "WARNING: Performance "
            "table not found."
        )
        return None

    rows = table.select(
        "tbody tr"
    )

    # Some historical pages may not
    # explicitly contain tbody
    if not rows:

        rows = table.find_all(
            "tr"
        )

    results = []

    valid_rows = []

    for row in rows:

        cells = row.find_all(
            "td"
        )

        if len(cells) >= 10:
            valid_rows.append(row)

    field_size = len(valid_rows)

    for row in valid_rows:

        cells = row.find_all(
            "td"
        )

        horse_link = row.find(
            "a",
            href=re.compile(
                r"horse",
                re.I
            )
        )

        horse_id = ""

        if horse_link:

            horse_id = (
                extract_horse_id(
                    horse_link.get(
                        "href",
                        ""
                    )
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

        if horse_link:

            horse_name = clean_text(
                horse_link.get_text(
                    " ",
                    strip=True
                )
            )

        else:

            horse_name = clean_text(
                cells[2].get_text(
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

        # ----------------------------------------------------
        # UNIQUE RESULT ID
        # ----------------------------------------------------

        result_id = (
            f"{race_metadata['race_id']}_"
            f"{horse_id or horse_number}"
        )

        # ----------------------------------------------------
        # DERIVED COLUMNS
        # ----------------------------------------------------

        is_winner = (
            finishing_position == 1
        )

        is_top_three = (
            finishing_position is not None
            and finishing_position <= 3
        )

        # ----------------------------------------------------
        # OUTPUT ROW
        # ----------------------------------------------------

        results.append({

            "result_id": result_id,

            "race_id": (
                race_metadata[
                    "race_id"
                ]
            ),

            "race_date": (
                race_metadata[
                    "race_date"
                ]
            ),

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

            "field_size": (
                field_size
            ),

            "horse_id": (
                horse_id
            ),

            "horse_number": (
                horse_number
            ),

            "horse_name": (
                horse_name
            ),

            "finishing_position": (
                finishing_position
            ),

            "is_winner": (
                is_winner
            ),

            "is_top_three": (
                is_top_three
            ),

            "jockey": (
                jockey
            ),

            "trainer": (
                trainer
            ),

            "actual_weight": (
                actual_weight
            ),

            "declared_horse_weight": (
                declared_horse_weight
            ),

            "draw": (
                draw
            ),

            "margin": (
                margin
            ),

            "finish_time": (
                finish_time
            ),

            "odds": (
                odds
            ),

            "race_url": (
                race_metadata[
                    "race_url"
                ]
            ),
        })

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

    except Exception as exc:

        print(
            "Could not read horse master:",
            exc
        )

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
# EXISTING RESULT IDS
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

    except Exception as exc:

        print(
            "Could not read existing "
            "result IDs:",
            exc
        )

        return set()


# ============================================================
# APPEND RESULTS
# ============================================================

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
# PROCESS ONE DATE
# ============================================================

def process_date(
    meeting_date,
    existing_result_ids,
    known_horses
):

    print()
    print(
        "=" * 70
    )

    print(
        "Checking:",
        meeting_date.strftime(
            "%Y/%m/%d"
        )
    )

    print(
        "=" * 70
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

    race_numbers = (
        detect_race_numbers(
            response.text
        )
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

    # ========================================================
    # PROCESS EVERY RACE
    # ========================================================

    for race_no in race_numbers:

        print()
        print(
            f"Processing Race {race_no}"
        )

        race_url = build_url(
            meeting_date,
            racecourse_code,
            race_no
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

        print(
            f"Found {len(results_df)} "
            "horse results."
        )

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
        len(existing_result_ids)
    )

    print(
        "Known horses:",
        len(known_horses)
    )

    # ========================================================
    # PROCESS DATE RANGE
    # ========================================================

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

    print()
    print(
        "Collection complete."
    )

    print(
        "Race results:",
        RACE_RESULTS_FILE
    )

    print(
        "Horse master:",
        HORSE_MASTER_FILE
    )


if __name__ == "__main__":
    main()
