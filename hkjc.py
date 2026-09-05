import os
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qsl, urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://racing.hkjc.com/en-us/local/information/archive/localresults"
HORSE_BASE_URL = "https://racing.hkjc.com/en-us/local/information/horse"

START_DATE = os.getenv("START_DATE", "2006-01-01")
END_DATE = os.getenv("END_DATE", "2006-12-31")

# One horse HTTP request supplies BOTH:
#   1. horse profile information
#   2. complete form/rating history
#
# You can override this, e.g.
# DELAY_SECONDS=1 python script.py
DELAY_SECONDS = float(os.getenv("DELAY_SECONDS", "2"))

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

HORSE_RATINGS_CACHE_FILE = os.path.join(
    HORSES_DIR,
    "horse_ratings_cache.csv"
)

PAYOUT_MODEL_CUTOFF = pd.Timestamp("2023-09-10")


# ============================================================
# COLUMNS
# ============================================================

HORSE_COLUMNS = [
    "horse_id",
    "horse_name",
    "brand_number",
    "country_of_origin",
    "hemisphere_of_origin",
    "horse_age",
    "foaled_date",
    "horse_colour",
    "horse_sex",
    "sire",
    "dam",
    "dam_sire",
    "profile_url",
    "profile_scraped",
    "profile_scraped_at",
]


HORSE_RATING_CACHE_COLUMNS = [
    "horse_id",
    "race_date",
    "race_index",
    "race_class",
    "horse_rating_before",
    "horse_rating_after",
    "rating_source_url",
    "rating_scraped_at",
]


RACE_COLUMNS = [
    "result_id",
    "race_id",

    "race_date",
    "racecourse_code",
    "racecourse_name",
    "race_number",
    "race_index",
    "race_name",
    "race_class",
    "distance_m",
    "rating_band",
    "going",
    "surface",
    "course",
    "prize_money_hkd",
    "field_size",

    "horse_id",
    "horse_number",
    "horse_name",

    "brand_number",
    "country_of_origin",
    "hemisphere_of_origin",
    "horse_age_at_race",
    "horse_rating_before",
    "horse_rating_after",
    "horse_colour",
    "horse_sex",
    "sire",
    "dam",
    "dam_sire",

    "career_starts_before",
    "career_wins_before",
    "career_seconds_before",
    "career_thirds_before",
    "career_top3_before",
    "career_win_rate_before",
    "career_top3_rate_before",

    "prize_payout_percentage",
    "prize_money_won_this_race",
    "career_prize_money_before",
    "career_prize_money_after",

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

    "horse_profile_url",
    "horse_profile_scraped_at",
    "race_url",
]


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
    os.makedirs(
        RACES_DIR,
        exist_ok=True
    )

    os.makedirs(
        HORSES_DIR,
        exist_ok=True
    )


def date_range(
    start_date,
    end_date
):
    current = start_date

    while current <= end_date:
        yield current
        current += timedelta(
            days=1
        )


def clean_text(value):
    if value is None:
        return ""

    text = str(value)

    text = text.replace(
        "\xa0",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def utc_now_string():
    return datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def parse_integer(value):
    if value is None:
        return None

    match = re.search(
        r"-?\d+",
        clean_text(value)
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

    text = clean_text(
        value
    ).replace(
        ",",
        ""
    )

    try:
        return float(
            text
        )
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
            match.group(1).replace(
                ",",
                ""
            )
        )
    except ValueError:
        return None


def extract_horse_id(href):
    if not href:
        return ""

    match = re.search(
        r"[?&]horseid=([^&]+)",
        href,
        re.I
    )

    if match:
        return clean_text(
            match.group(1)
        )

    return ""


def clean_finish_time(value):
    text = clean_text(
        value
    )

    if not text:
        return None

    normalised = text.replace(
        " ",
        ""
    )

    if re.fullmatch(
        r"0+(?:(?::|\.)0+)+",
        normalised
    ):
        return None

    return text


# ============================================================
# HORSE HEMISPHERE + AGE
# ============================================================

SOUTHERN_HEMISPHERE_ORIGINS = {
    "AUS",
    "NZ",
    "SAF",
    "ARG",
    "ZIM",
    "BRZ",
    "CHI",
}


NORTHERN_HEMISPHERE_ORIGINS = {
    "IRE",
    "GB",
    "USA",
    "FR",
    "JPN",
    "CAN",
    "GER",
}


def get_hemisphere_of_origin(
    country_of_origin
):
    country = clean_text(
        country_of_origin
    ).upper()

    if country in SOUTHERN_HEMISPHERE_ORIGINS:
        return "Southern"

    if country in NORTHERN_HEMISPHERE_ORIGINS:
        return "Northern"

    return "Other"


def is_southern_hemisphere_horse(
    country_of_origin
):
    return (
        get_hemisphere_of_origin(
            country_of_origin
        )
        == "Southern"
    )


def get_official_horse_birthday(
    country_of_origin
):
    if is_southern_hemisphere_horse(
        country_of_origin
    ):
        return 8, 1

    return 1, 1


def parse_date_only(value):
    if value is None:
        return None

    text = clean_text(
        value
    )

    if not text:
        return None

    parsed = pd.to_datetime(
        text,
        errors="coerce",
        utc=True
    )

    if pd.isna(
        parsed
    ):
        return None

    return parsed.date()


def infer_horse_birth_year(
    current_age,
    country_of_origin,
    profile_scraped_at
):
    age = parse_integer(
        current_age
    )

    if age is None:
        return None

    scraped_date = parse_date_only(
        profile_scraped_at
    )

    if scraped_date is None:
        return None

    (
        birthday_month,
        birthday_day
    ) = get_official_horse_birthday(
        country_of_origin
    )

    official_birthday_this_year = (
        scraped_date.replace(
            month=birthday_month,
            day=birthday_day
        )
    )

    if (
        scraped_date
        <
        official_birthday_this_year
    ):
        reference_birthday_year = (
            scraped_date.year - 1
        )
    else:
        reference_birthday_year = (
            scraped_date.year
        )

    return (
        reference_birthday_year
        -
        age
    )


def calculate_horse_age_at_race(
    current_age,
    country_of_origin,
    profile_scraped_at,
    race_date
):
    birth_year = infer_horse_birth_year(
        current_age,
        country_of_origin,
        profile_scraped_at
    )

    if birth_year is None:
        return None

    race_date_parsed = parse_date_only(
        race_date
    )

    if race_date_parsed is None:
        return None

    (
        birthday_month,
        birthday_day
    ) = get_official_horse_birthday(
        country_of_origin
    )

    age_at_race = (
        race_date_parsed.year
        -
        birth_year
    )

    if (
        race_date_parsed.month,
        race_date_parsed.day
    ) < (
        birthday_month,
        birthday_day
    ):
        age_at_race -= 1

    if age_at_race < 0:
        return None

    return age_at_race


# ============================================================
# URL BUILDERS
# ============================================================

def ensure_horse_option_1(
    url
):
    url = clean_text(
        url
    )

    if not url:
        return ""

    parts = urlsplit(
        url
    )

    query_pairs = [
        (
            key,
            value
        )
        for (
            key,
            value
        ) in parse_qsl(
            parts.query,
            keep_blank_values=True
        )
        if key.lower() != "option"
    ]

    query_pairs.append(
        (
            "Option",
            "1"
        )
    )

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(
                query_pairs
            ),
            parts.fragment,
        )
    )


def build_url(
    race_date,
    racecourse=None,
    race_no=None
):
    url = (
        f"{BASE_URL}"
        f"?racedate="
        f"{race_date.strftime('%Y/%m/%d')}"
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


def build_horse_url(
    horse_id
):
    return ensure_horse_option_1(
        f"{HORSE_BASE_URL}"
        f"?horseid={horse_id}"
    )


# ============================================================
# REQUESTS
# ============================================================

def request_race_page(
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
            "Race request failed:",
            exc
        )

        return None


def request_horse_page(
    horse_id
):
    url = build_horse_url(
        horse_id
    )

    try:
        response = session.get(
            url,
            timeout=60,
            allow_redirects=True
        )

        print(
            f"HORSE {horse_id} "
            f"{url} "
            f"-> {response.status_code}"
        )

        response.raise_for_status()

        return response

    except requests.RequestException as exc:
        print(
            f"Horse request failed "
            f"{horse_id}:",
            exc
        )

        return None


# ============================================================
# MEETING DETECTION
# ============================================================

def detect_meeting(
    html
):
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
            "racecourse_code":
                "HV",

            "racecourse_name":
                "Happy Valley",
        }

    if "Sha Tin" in text:
        return {
            "racecourse_code":
                "ST",

            "racecourse_name":
                "Sha Tin",
        }

    return None


def detect_race_numbers(
    html
):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    race_numbers = set()

    for text in soup.stripped_strings:

        match = re.search(
            r"\bRACE\s+(\d+)\b",
            text,
            re.I
        )

        if match:
            race_numbers.add(
                int(
                    match.group(1)
                )
            )

    for link in soup.find_all(
        "a",
        href=True
    ):

        match = re.search(
            r"[?&]RaceNo=(\d+)",
            link.get(
                "href",
                ""
            ),
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
# RACE HEADER PARSER
# ============================================================

def get_cell_texts(
    row
):
    return [
        clean_text(
            cell.get_text(
                " ",
                strip=True
            )
        )
        for cell in row.find_all(
            [
                "td",
                "th"
            ]
        )
    ]


def extract_label_value(
    cells,
    label
):
    pattern = re.compile(
        rf"\b{re.escape(label)}\s*:",
        re.I
    )

    for index, cell in enumerate(
        cells
    ):

        if not pattern.search(
            cell
        ):
            continue

        same_cell = pattern.sub(
            "",
            cell
        ).strip()

        if same_cell:
            return same_cell

        for next_index in range(
            index + 1,
            len(cells)
        ):

            if cells[
                next_index
            ]:
                return cells[
                    next_index
                ]

    return ""


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
        "race_id":
            (
                f"{racecourse_code}_"
                f"{meeting_date.strftime('%Y%m%d')}_"
                f"R{race_no:02d}"
            ),

        "race_date":
            meeting_date.strftime(
                "%Y-%m-%d"
            ),

        "racecourse_code":
            racecourse_code,

        "racecourse_name":
            racecourse_name,

        "race_number":
            race_no,

        "race_index":
            None,

        "race_name":
            "",

        # Header race class is diagnostic only.
        #
        # IMPORTANT:
        # Final race_class written to all_results.csv
        # comes ONLY from the matching horse form row.
        "race_class":
            "",

        "distance_m":
            None,

        "rating_band":
            "",

        "going":
            "",

        "surface":
            "",

        "course":
            "",

        "prize_money_hkd":
            None,

        "race_url":
            race_url,
    }

    header = soup.select_one(
        "table.race_tab"
    )

    if header is None:
        header = soup.select_one(
            ".race_tab"
        )

    if header is None:
        print(
            "WARNING: race header "
            "not found"
        )

        return metadata

    rows = header.find_all(
        "tr"
    )

    full_text = clean_text(
        header.get_text(
            " ",
            strip=True
        )
    )

    race_match = re.search(
        r"\bRACE\s+(\d+)\s*"
        r"\(\s*(\d+)\s*\)",
        full_text,
        re.I
    )

    if race_match:

        metadata[
            "race_number"
        ] = int(
            race_match.group(1)
        )

        metadata[
            "race_index"
        ] = int(
            race_match.group(2)
        )

    for row in rows:

        cells = get_cell_texts(
            row
        )

        if not cells:
            continue

        for cell in cells:

            class_match = re.search(
                r"\b(Class\s+\d+)\b",
                cell,
                re.I
            )

            if class_match:

                metadata[
                    "race_class"
                ] = (
                    class_match
                    .group(1)
                    .title()
                )

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

        going = extract_label_value(
            cells,
            "Going"
        )

        if going:
            metadata[
                "going"
            ] = going

        course = extract_label_value(
            cells,
            "Course"
        )

        if course:
            metadata[
                "course"
            ] = course

        for index, cell in enumerate(
            cells
        ):

            if not re.search(
                r"\bCourse\s*:",
                cell,
                re.I
            ):
                continue

            for candidate in cells[
                :index
            ]:

                candidate = clean_text(
                    candidate
                )

                if not candidate:
                    continue

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

                metadata[
                    "race_name"
                ] = candidate

                break

        for cell in cells:

            prize = parse_prize_money(
                cell
            )

            if prize is not None:
                metadata[
                    "prize_money_hkd"
                ] = prize

    if not metadata[
        "race_class"
    ]:

        match = re.search(
            r"\b(Class\s+\d+)\b",
            full_text,
            re.I
        )

        if match:
            metadata[
                "race_class"
            ] = (
                match.group(1)
                .title()
            )

    if metadata[
        "distance_m"
    ] is None:

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

    if not metadata[
        "rating_band"
    ]:

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

    if metadata[
        "prize_money_hkd"
    ] is None:

        metadata[
            "prize_money_hkd"
        ] = parse_prize_money(
            full_text
        )

    course_upper = clean_text(
        metadata[
            "course"
        ]
    ).upper()

    if "TURF" in course_upper:
        metadata[
            "surface"
        ] = "TURF"

    elif (
        "ALL WEATHER"
        in course_upper
        or
        "AWT"
        in course_upper
    ):
        metadata[
            "surface"
        ] = "ALL WEATHER TRACK"

    print(
        "PARSED RACE:",
        {
            "race_number":
                metadata[
                    "race_number"
                ],

            "race_index":
                metadata[
                    "race_index"
                ],

            "race_name":
                metadata[
                    "race_name"
                ],

            "header_race_class":
                metadata[
                    "race_class"
                ],

            "distance_m":
                metadata[
                    "distance_m"
                ],

            "rating_band":
                metadata[
                    "rating_band"
                ],

            "going":
                metadata[
                    "going"
                ],

            "surface":
                metadata[
                    "surface"
                ],

            "course":
                metadata[
                    "course"
                ],

            "prize_money_hkd":
                metadata[
                    "prize_money_hkd"
                ],
        }
    )

    return metadata


# ============================================================
# RESULT TABLE PARSER
# ============================================================

RESULT_HEADER_ALIASES = {
    "finishing_position": {
        "pla",
        "place",
        "placing",
        "position",
        "pos",
    },

    "horse_number": {
        "horse no",
        "horse number",
        "horse num",
    },

    "horse": {
        "horse",
        "horse name",
    },

    "jockey": {
        "jockey",
    },

    "trainer": {
        "trainer",
    },

    "actual_weight": {
        "act wt",
        "actual wt",
        "actual weight",
    },

    "declared_horse_weight": {
        "declar horse wt",
        "declared horse wt",
        "declar horse weight",
        "declared horse weight",
    },

    "draw": {
        "dr",
        "draw",
    },

    "margin": {
        "lbw",
        "margin",
        "lengths behind winner",
    },

    "running_position": {
        "running position",
        "running pos",
        "running positions",
    },

    "finish_time": {
        "finish time",
        "finishing time",
    },

    "odds": {
        "win odds",
        "winning odds",
        "odds",
    },
}


def normalise_result_header(
    value
):
    text = clean_text(
        value
    ).lower()

    text = text.replace(
        "&",
        " and "
    )

    text = re.sub(
        r"[.()/\\_-]+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def identify_result_header(
    value
):
    normalised = normalise_result_header(
        value
    )

    if not normalised:
        return None

    for (
        canonical_name,
        aliases
    ) in RESULT_HEADER_ALIASES.items():

        if normalised in aliases:
            return canonical_name

    return None


def build_result_column_map(
    table
):
    best_map = {}
    best_score = 0

    for row in table.find_all(
        "tr"
    ):

        cells = row.find_all(
            [
                "th",
                "td"
            ]
        )

        if not cells:
            continue

        current_map = {}

        for index, cell in enumerate(
            cells
        ):

            heading = clean_text(
                cell.get_text(
                    " ",
                    strip=True
                )
            )

            canonical_name = (
                identify_result_header(
                    heading
                )
            )

            if (
                canonical_name
                and
                canonical_name
                not in current_map
            ):
                current_map[
                    canonical_name
                ] = index

        score = len(
            current_map
        )

        if score > best_score:
            best_score = score
            best_map = current_map

    if best_score < 5:
        return {}

    return best_map


def get_result_cell(
    cells,
    column_map,
    field_name
):
    index = column_map.get(
        field_name
    )

    if (
        index is None
        or
        index < 0
        or
        index >= len(cells)
    ):
        return None

    return cells[
        index
    ]


def get_result_cell_text(
    cells,
    column_map,
    field_name
):
    cell = get_result_cell(
        cells,
        column_map,
        field_name
    )

    if cell is None:
        return ""

    return clean_text(
        cell.get_text(
            " ",
            strip=True
        )
    )


def extract_results(
    html,
    race_metadata
):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    table = soup.select_one(
        "table.performance"
    )

    if table is None:
        table = soup.select_one(
            ".performance table"
        )

    if table is None:
        print(
            "WARNING: results "
            "table not found"
        )

        return None

    column_map = (
        build_result_column_map(
            table
        )
    )

    if not column_map:
        print(
            "WARNING: could not detect "
            "results-table headings"
        )

        return None

    print(
        "RESULT COLUMN MAP:",
        column_map
    )

    core_fields = [
        "finishing_position",
        "horse_number",
        "horse",
        "jockey",
        "trainer",
        "actual_weight",
        "declared_horse_weight",
        "draw",
        "margin",
    ]

    missing_core = [
        field
        for field in core_fields
        if field not in column_map
    ]

    if missing_core:
        print(
            "WARNING: results table is "
            "missing expected headings:",
            missing_core
        )

        return None

    candidate_rows = []

    for row in table.find_all(
        "tr"
    ):

        cells = row.find_all(
            "td"
        )

        if not cells:
            continue

        horse_cell = get_result_cell(
            cells,
            column_map,
            "horse"
        )

        horse_number = parse_integer(
            get_result_cell_text(
                cells,
                column_map,
                "horse_number"
            )
        )

        horse_link = None

        if horse_cell is not None:
            horse_link = horse_cell.find(
                "a",
                href=re.compile(
                    r"horse",
                    re.I
                )
            )

        if horse_link is None:
            horse_link = row.find(
                "a",
                href=re.compile(
                    r"horse",
                    re.I
                )
            )

        horse_id = ""

        if horse_link is not None:
            horse_id = extract_horse_id(
                horse_link.get(
                    "href",
                    ""
                )
            )

        if (
            not horse_id
            and
            horse_number is None
        ):
            continue

        candidate_rows.append(
            (
                row,
                cells,
                horse_link,
                horse_id,
                horse_number
            )
        )

    field_size = len(
        candidate_rows
    )

    results = []

    for (
        row,
        cells,
        horse_link,
        horse_id,
        horse_number
    ) in candidate_rows:

        horse_url = ""

        if horse_link is not None:

            horse_url = (
                ensure_horse_option_1(
                    urljoin(
                        "https://racing.hkjc.com",
                        horse_link.get(
                            "href",
                            ""
                        )
                    )
                )
            )

        horse_cell = get_result_cell(
            cells,
            column_map,
            "horse"
        )

        if horse_link is not None:

            horse_name = clean_text(
                horse_link.get_text(
                    " ",
                    strip=True
                )
            )

        elif horse_cell is not None:

            horse_name = clean_text(
                horse_cell.get_text(
                    " ",
                    strip=True
                )
            )

        else:
            horse_name = ""

        finishing_position = (
            parse_integer(
                get_result_cell_text(
                    cells,
                    column_map,
                    "finishing_position"
                )
            )
        )

        result_id = (
            f"{race_metadata['race_id']}_"
            f"{horse_id or horse_number}"
        )

        results.append({
            "result_id":
                result_id,

            "race_id":
                race_metadata[
                    "race_id"
                ],

            "race_date":
                race_metadata[
                    "race_date"
                ],

            "racecourse_code":
                race_metadata[
                    "racecourse_code"
                ],

            "racecourse_name":
                race_metadata[
                    "racecourse_name"
                ],

            "race_number":
                race_metadata[
                    "race_number"
                ],

            "race_index":
                race_metadata[
                    "race_index"
                ],

            "race_name":
                race_metadata[
                    "race_name"
                ],

            # IMPORTANT:
            # deliberately blank.
            #
            # This gets populated ONLY from
            # the matching horse form-history row.
            "race_class":
                "",

            "distance_m":
                race_metadata[
                    "distance_m"
                ],

            "rating_band":
                race_metadata[
                    "rating_band"
                ],

            "going":
                race_metadata[
                    "going"
                ],

            "surface":
                race_metadata[
                    "surface"
                ],

            "course":
                race_metadata[
                    "course"
                ],

            "prize_money_hkd":
                race_metadata[
                    "prize_money_hkd"
                ],

            "field_size":
                field_size,

            "horse_id":
                horse_id,

            "horse_number":
                horse_number,

            "horse_name":
                horse_name,

            "finishing_position":
                finishing_position,

            "is_winner":
                finishing_position == 1,

            "is_top_three":
                (
                    finishing_position
                    is not None
                    and
                    finishing_position <= 3
                ),

            "jockey":
                get_result_cell_text(
                    cells,
                    column_map,
                    "jockey"
                ),

            "trainer":
                get_result_cell_text(
                    cells,
                    column_map,
                    "trainer"
                ),

            "actual_weight":
                parse_integer(
                    get_result_cell_text(
                        cells,
                        column_map,
                        "actual_weight"
                    )
                ),

            "declared_horse_weight":
                parse_integer(
                    get_result_cell_text(
                        cells,
                        column_map,
                        "declared_horse_weight"
                    )
                ),

            "draw":
                parse_integer(
                    get_result_cell_text(
                        cells,
                        column_map,
                        "draw"
                    )
                ),

            "margin":
                get_result_cell_text(
                    cells,
                    column_map,
                    "margin"
                ),

            "finish_time":
                clean_finish_time(
                    get_result_cell_text(
                        cells,
                        column_map,
                        "finish_time"
                    )
                ),

            "odds":
                parse_float(
                    get_result_cell_text(
                        cells,
                        column_map,
                        "odds"
                    )
                ),

            "horse_profile_url":
                horse_url,

            "race_url":
                race_metadata[
                    "race_url"
                ],
        })

    if not results:
        return None

    return pd.DataFrame(
        results
    )


# ============================================================
# HORSE PROFILE PARSER
# ============================================================

PROFILE_LABELS = [
    "Country of Origin / Age",
    "Country of Origin",
    "Colour / Sex",
    "Import Type",
    "Import Date",
    "Owner",
    "Sire",
    "Dam",
    "Dam's Sire",
]


# These can immediately follow the profile block on Option=1
# pages. Including them prevents the final profile field,
# especially Dam's Sire, from swallowing the form-table text.
PROFILE_SECTION_STOP_LABELS = [
    "Current Rating",
    "Last Rating",
]


def normalise_profile_text(
    soup
):
    return clean_text(
        soup.get_text(
            " ",
            strip=True
        )
    )


def extract_profile_value(
    text,
    label
):
    other_labels = [
        item
        for item in (
            PROFILE_LABELS
            +
            PROFILE_SECTION_STOP_LABELS
        )
        if (
            item.lower()
            !=
            label.lower()
        )
    ]

    stop_pattern = "|".join(
        re.escape(item)
        for item in sorted(
            other_labels,
            key=len,
            reverse=True
        )
    )

    pattern = (
        rf"{re.escape(label)}"
        rf"\s*:\s*"
        rf"(.*?)"
        rf"(?="
        rf"\s+(?:{stop_pattern})\s*:|"
        rf"$"
        rf")"
    )

    match = re.search(
        pattern,
        text,
        re.I
    )

    if not match:
        return ""

    return clean_text(
        match.group(1)
    )


def extract_profile_value_any(
    text,
    labels
):
    for label in labels:

        value = extract_profile_value(
            text,
            label
        )

        if value:
            return value

    return ""


def extract_horse_profile(
    html,
    horse_id,
    profile_url,
    fallback_name=""
):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    text = normalise_profile_text(
        soup
    )

    profile_url = (
        ensure_horse_option_1(
            profile_url
        )
    )

    profile = {
        column: ""
        for column in HORSE_COLUMNS
    }

    profile[
        "horse_id"
    ] = horse_id

    profile[
        "horse_name"
    ] = fallback_name

    profile[
        "profile_url"
    ] = profile_url

    profile[
        "profile_scraped"
    ] = False

    heading_match = re.search(
        r"\b([A-Z][A-Z0-9 '&.\-]+?)"
        r"\s+\(([A-Z]{1,3}\d{3})\)",
        text,
        re.I
    )

    if heading_match:

        name = clean_text(
            heading_match.group(1)
        )

        if (
            name
            and
            len(name) <= 80
        ):
            profile[
                "horse_name"
            ] = name

        profile[
            "brand_number"
        ] = clean_text(
            heading_match.group(2)
        )

    origin_age = extract_profile_value_any(
        text,
        [
            "Country of Origin / Age",
            "Country of Origin",
        ]
    )

    if origin_age:

        parts = [
            clean_text(
                item
            )
            for item in origin_age.split(
                "/",
                1
            )
        ]

        if (
            parts
            and
            parts[0]
        ):
            profile[
                "country_of_origin"
            ] = parts[0]

        if len(parts) >= 2:

            age = parse_integer(
                parts[1]
            )

            if age is not None:
                profile[
                    "horse_age"
                ] = age

    profile[
        "hemisphere_of_origin"
    ] = get_hemisphere_of_origin(
        profile.get(
            "country_of_origin",
            ""
        )
    )

    colour_sex = extract_profile_value(
        text,
        "Colour / Sex"
    )

    if colour_sex:

        parts = [
            clean_text(
                item
            )
            for item in colour_sex.rsplit(
                "/",
                1
            )
        ]

        if parts:
            profile[
                "horse_colour"
            ] = parts[0]

        if len(parts) >= 2:
            profile[
                "horse_sex"
            ] = parts[1]

    profile[
        "sire"
    ] = extract_profile_value(
        text,
        "Sire"
    )

    profile[
        "dam"
    ] = extract_profile_value(
        text,
        "Dam"
    )

    profile[
        "dam_sire"
    ] = extract_profile_value(
        text,
        "Dam's Sire"
    )

    useful_fields = [
        profile.get(
            "country_of_origin"
        ),

        profile.get(
            "horse_age"
        ),

        profile.get(
            "horse_colour"
        ),

        profile.get(
            "horse_sex"
        ),

        profile.get(
            "sire"
        ),

        profile.get(
            "dam"
        ),
    ]

    success = any(
        clean_text(
            item
        )
        for item in useful_fields
        if item is not None
    )

    profile[
        "profile_scraped"
    ] = success

    if success:
        profile[
            "profile_scraped_at"
        ] = utc_now_string()

    print(
        "HORSE PROFILE:",
        {
            "horse_id":
                profile[
                    "horse_id"
                ],

            "horse_name":
                profile[
                    "horse_name"
                ],

            "brand":
                profile[
                    "brand_number"
                ],

            "origin":
                profile[
                    "country_of_origin"
                ],

            "hemisphere":
                profile[
                    "hemisphere_of_origin"
                ],

            "age":
                profile[
                    "horse_age"
                ],

            "colour":
                profile[
                    "horse_colour"
                ],

            "sex":
                profile[
                    "horse_sex"
                ],

            "sire":
                profile[
                    "sire"
                ],

            "dam":
                profile[
                    "dam"
                ],

            "dam_sire":
                profile[
                    "dam_sire"
                ],

            "success":
                profile[
                    "profile_scraped"
                ],
        }
    )

    return profile


# ============================================================
# HORSE MASTER
# ============================================================

def load_horse_master():
    horse_master = {}
    age_refresh_pending = set()

    if not os.path.exists(
        HORSE_MASTER_FILE
    ):
        return (
            horse_master,
            age_refresh_pending
        )

    try:
        df = pd.read_csv(
            HORSE_MASTER_FILE,
            dtype=object
        ).fillna("")

    except Exception as exc:

        print(
            "Could not read "
            "horse master:",
            exc
        )

        return (
            horse_master,
            age_refresh_pending
        )

    had_horse_age_column = (
        "horse_age"
        in df.columns
    )

    for column in HORSE_COLUMNS:

        if column not in df.columns:
            df[
                column
            ] = ""

    for _, row in df.iterrows():

        horse_id = clean_text(
            row.get(
                "horse_id",
                ""
            )
        )

        if not horse_id:
            continue

        record = {
            column:
                row.get(
                    column,
                    ""
                )
            for column in HORSE_COLUMNS
        }

        record[
            "hemisphere_of_origin"
        ] = get_hemisphere_of_origin(
            record.get(
                "country_of_origin",
                ""
            )
        )

        profile_url = clean_text(
            record.get(
                "profile_url",
                ""
            )
        )

        if profile_url:

            record[
                "profile_url"
            ] = ensure_horse_option_1(
                profile_url
            )

        horse_master[
            horse_id
        ] = record

        if not had_horse_age_column:
            age_refresh_pending.add(
                horse_id
            )

    if age_refresh_pending:

        print(
            f"{len(age_refresh_pending)} "
            f"existing horse profiles "
            f"need a one-time age refresh."
        )

    return (
        horse_master,
        age_refresh_pending
    )


def save_horse_master(
    horse_master
):
    if not horse_master:
        return

    for horse in horse_master.values():

        horse[
            "hemisphere_of_origin"
        ] = get_hemisphere_of_origin(
            horse.get(
                "country_of_origin",
                ""
            )
        )

        profile_url = clean_text(
            horse.get(
                "profile_url",
                ""
            )
        )

        if profile_url:

            horse[
                "profile_url"
            ] = ensure_horse_option_1(
                profile_url
            )

    df = pd.DataFrame(
        list(
            horse_master.values()
        )
    )

    for column in HORSE_COLUMNS:

        if column not in df.columns:
            df[
                column
            ] = ""

    df = df[
        HORSE_COLUMNS
    ]

    df = df.drop_duplicates(
        subset=[
            "horse_id"
        ],
        keep="last"
    )

    df = df.sort_values(
        by=[
            "horse_id"
        ]
    )

    df.to_csv(
        HORSE_MASTER_FILE,
        index=False
    )


def horse_profile_is_scraped(
    horse
):
    if not horse:
        return False

    return clean_text(
        horse.get(
            "profile_scraped",
            ""
        )
    ).lower() in {
        "true",
        "1",
        "yes",
    }


def merge_horse_profile(
    existing,
    new_profile
):
    if not existing:
        return new_profile

    if (
        not new_profile
        or
        not new_profile.get(
            "profile_scraped"
        )
    ):
        return existing

    merged = dict(
        existing
    )

    for column in HORSE_COLUMNS:

        new_value = (
            new_profile.get(
                column,
                ""
            )
        )

        if column in {
            "profile_scraped",
            "profile_scraped_at",
            "profile_url",
        }:

            merged[
                column
            ] = new_value

        elif clean_text(
            new_value
        ):

            merged[
                column
            ] = new_value

    return merged


# ============================================================
# HORSE FORM / RATING HISTORY PARSER
# ============================================================

HORSE_FORM_HEADER_ALIASES = {
    "race_index": {
        "race index",
        "race no index",
    },

    "date": {
        "date",
        "race date",
    },

    "race_class": {
        "race class",
        "class",
    },

    "rating": {
        "rtg",
        "rating",
    },
}


def normalise_horse_form_header(
    value
):
    text = clean_text(
        value
    ).lower()

    text = re.sub(
        r"[.()/\\_-]+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def identify_horse_form_header(
    value
):
    normalised = (
        normalise_horse_form_header(
            value
        )
    )

    if not normalised:
        return None

    for (
        canonical_name,
        aliases
    ) in (
        HORSE_FORM_HEADER_ALIASES.items()
    ):

        if normalised in aliases:
            return canonical_name

    return None


def build_horse_form_column_map(
    table
):
    best_map = {}
    best_score = 0

    for row in table.find_all(
        "tr"
    ):

        cells = row.find_all(
            [
                "th",
                "td"
            ]
        )

        if not cells:
            continue

        current_map = {}

        for index, cell in enumerate(
            cells
        ):

            heading = clean_text(
                cell.get_text(
                    " ",
                    strip=True
                )
            )

            canonical_name = (
                identify_horse_form_header(
                    heading
                )
            )

            if (
                canonical_name
                and
                canonical_name
                not in current_map
            ):

                current_map[
                    canonical_name
                ] = index

        score = len(
            current_map
        )

        if score > best_score:
            best_score = score
            best_map = current_map

    required = {
        "race_index",
        "date",
        "race_class",
        "rating",
    }

    if not required.issubset(
        best_map.keys()
    ):
        return {}

    return best_map


def find_horse_form_table(
    soup
):
    for table in soup.find_all(
        "table"
    ):

        column_map = (
            build_horse_form_column_map(
                table
            )
        )

        if column_map:
            return (
                table,
                column_map
            )

    return (
        None,
        {}
    )


def get_form_cell_text(
    cells,
    column_map,
    field_name
):
    index = column_map.get(
        field_name
    )

    if (
        index is None
        or
        index < 0
        or
        index >= len(cells)
    ):
        return ""

    return clean_text(
        cells[
            index
        ].get_text(
            " ",
            strip=True
        )
    )


def parse_hkjc_form_date(
    value
):
    text = clean_text(
        value
    )

    if not text:
        return None

    for format_string in [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
    ]:

        try:
            return datetime.strptime(
                text,
                format_string
            ).date()

        except ValueError:
            pass

    parsed = pd.to_datetime(
        text,
        errors="coerce",
        dayfirst=True
    )

    if pd.isna(
        parsed
    ):
        return None

    return parsed.date()


def extract_header_rating(
    text
):
    for label in [
        "Last Rating",
        "Current Rating",
    ]:

        match = re.search(
            rf"\b{re.escape(label)}"
            rf"\s*:\s*(-?\d+)",
            text,
            re.I
        )

        if match:

            try:
                return int(
                    match.group(1)
                )

            except ValueError:
                return None

    return None


def extract_horse_rating_history(
    html,
    horse_id,
    source_url
):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    full_text = clean_text(
        soup.get_text(
            " ",
            strip=True
        )
    )

    latest_header_rating = (
        extract_header_rating(
            full_text
        )
    )

    (
        table,
        column_map
    ) = find_horse_form_table(
        soup
    )

    if table is None:

        print(
            f"WARNING: horse form "
            f"table not found for "
            f"{horse_id}"
        )

        return pd.DataFrame(
            columns=
                HORSE_RATING_CACHE_COLUMNS
        )

    print(
        f"HORSE FORM COLUMN MAP "
        f"{horse_id}:",
        column_map
    )

    records = []

    for row in table.find_all(
        "tr"
    ):

        cells = row.find_all(
            "td"
        )

        if not cells:
            continue

        race_index = parse_integer(
            get_form_cell_text(
                cells,
                column_map,
                "race_index"
            )
        )

        race_date = parse_hkjc_form_date(
            get_form_cell_text(
                cells,
                column_map,
                "date"
            )
        )

        race_class = clean_text(
            get_form_cell_text(
                cells,
                column_map,
                "race_class"
            )
        )

        rating_before = parse_integer(
            get_form_cell_text(
                cells,
                column_map,
                "rating"
            )
        )

        if (
            race_index is None
            or
            race_date is None
        ):
            continue

        records.append({
            "horse_id":
                horse_id,

            "race_date":
                race_date.strftime(
                    "%Y-%m-%d"
                ),

            "race_index":
                race_index,

            # IMPORTANT:
            # This comes directly from the horse
            # profile/form-history Race Class column.
            "race_class":
                race_class,

            "horse_rating_before":
                rating_before,

            "horse_rating_after":
                None,

            "rating_source_url":
                ensure_horse_option_1(
                    source_url
                ),

            "rating_scraped_at":
                utc_now_string(),
        })

    if not records:

        return pd.DataFrame(
            columns=
                HORSE_RATING_CACHE_COLUMNS
        )

    unique_records = {}

    for record in records:

        key = (
            record[
                "race_date"
            ],

            int(
                record[
                    "race_index"
                ]
            )
        )

        unique_records[
            key
        ] = record

    records = list(
        unique_records.values()
    )

    records.sort(
        key=lambda item: (
            datetime.strptime(
                item[
                    "race_date"
                ],
                "%Y-%m-%d"
            ).date(),

            int(
                item[
                    "race_index"
                ]
            )
        )
    )

    # HKJC's form table rating corresponds to the rating
    # going INTO that race.
    #
    # Therefore:
    #
    # current race's rating_after =
    # next race's rating_before
    #
    # For the latest race, use the profile page's
    # Current Rating / Last Rating header.
    for index, record in enumerate(
        records
    ):

        if (
            index + 1
            <
            len(records)
        ):

            record[
                "horse_rating_after"
            ] = records[
                index + 1
            ][
                "horse_rating_before"
            ]

        else:

            record[
                "horse_rating_after"
            ] = latest_header_rating

    df = pd.DataFrame(
        records
    )

    for column in (
        HORSE_RATING_CACHE_COLUMNS
    ):

        if column not in df.columns:
            df[
                column
            ] = ""

    return df[
        HORSE_RATING_CACHE_COLUMNS
    ]


# ============================================================
# HORSE RATING / FORM CACHE
# ============================================================

def load_horse_rating_cache():
    if not os.path.exists(
        HORSE_RATINGS_CACHE_FILE
    ):

        return pd.DataFrame(
            columns=
                HORSE_RATING_CACHE_COLUMNS
        )

    try:
        df = pd.read_csv(
            HORSE_RATINGS_CACHE_FILE,
            dtype=object
        ).fillna("")

    except Exception as exc:

        print(
            "Could not read horse "
            "rating cache:",
            exc
        )

        return pd.DataFrame(
            columns=
                HORSE_RATING_CACHE_COLUMNS
        )

    for column in (
        HORSE_RATING_CACHE_COLUMNS
    ):

        if column not in df.columns:
            df[
                column
            ] = ""

    df[
        "rating_source_url"
    ] = df[
        "rating_source_url"
    ].apply(
        lambda value:
            ensure_horse_option_1(
                value
            )
            if clean_text(value)
            else ""
    )

    return df[
        HORSE_RATING_CACHE_COLUMNS
    ]


def save_horse_rating_cache(
    cache_df
):
    if cache_df is None:
        return

    df = cache_df.copy()

    for column in (
        HORSE_RATING_CACHE_COLUMNS
    ):

        if column not in df.columns:
            df[
                column
            ] = ""

    df[
        "rating_source_url"
    ] = df[
        "rating_source_url"
    ].apply(
        lambda value:
            ensure_horse_option_1(
                value
            )
            if clean_text(value)
            else ""
    )

    df = df[
        HORSE_RATING_CACHE_COLUMNS
    ]

    if not df.empty:

        df[
            "_race_index_sort"
        ] = pd.to_numeric(
            df[
                "race_index"
            ],
            errors="coerce"
        )

        df[
            "_race_date_sort"
        ] = pd.to_datetime(
            df[
                "race_date"
            ],
            errors="coerce"
        )

        df = df.drop_duplicates(
            subset=[
                "horse_id",
                "race_date",
                "race_index",
            ],
            keep="last"
        )

        df = df.sort_values(
            by=[
                "horse_id",
                "_race_date_sort",
                "_race_index_sort",
            ],
            kind="stable"
        )

        df = df.drop(
            columns=[
                "_race_index_sort",
                "_race_date_sort",
            ],
            errors="ignore"
        )

    df.to_csv(
        HORSE_RATINGS_CACHE_FILE,
        index=False
    )


def normalise_rating_key(
    horse_id,
    race_date,
    race_index
):
    horse_id = clean_text(
        horse_id
    )

    parsed_date = parse_date_only(
        race_date
    )

    parsed_index = parse_integer(
        race_index
    )

    if (
        not horse_id
        or
        parsed_date is None
        or
        parsed_index is None
    ):
        return None

    return (
        horse_id,

        parsed_date.strftime(
            "%Y-%m-%d"
        ),

        int(
            parsed_index
        )
    )


def build_rating_cache_lookup(
    cache_df
):
    lookup = {}

    if (
        cache_df is None
        or
        cache_df.empty
    ):
        return lookup

    for _, row in cache_df.iterrows():

        key = normalise_rating_key(
            row.get(
                "horse_id",
                ""
            ),

            row.get(
                "race_date",
                ""
            ),

            row.get(
                "race_index",
                ""
            )
        )

        if key is None:
            continue

        lookup[
            key
        ] = {
            "race_class":
                row.get(
                    "race_class",
                    ""
                ),

            "horse_rating_before":
                row.get(
                    "horse_rating_before",
                    ""
                ),

            "horse_rating_after":
                row.get(
                    "horse_rating_after",
                    ""
                ),
        }

    return lookup


def apply_rating_cache_to_results(
    results_df,
    cache_df
):
    df = results_df.copy()

    for column in [
        "race_class",
        "horse_rating_before",
        "horse_rating_after",
    ]:

        if column not in df.columns:

            df[
                column
            ] = pd.Series(
                [None] * len(df),
                index=df.index,
                dtype="object"
            )

        else:

            df[
                column
            ] = df[
                column
            ].astype(
                "object"
            )

    lookup = build_rating_cache_lookup(
        cache_df
    )

    matched = 0
    class_matched = 0

    for index, row in df.iterrows():

        key = normalise_rating_key(
            row.get(
                "horse_id",
                ""
            ),

            row.get(
                "race_date",
                ""
            ),

            row.get(
                "race_index",
                ""
            )
        )

        if key is None:
            continue

        rating_record = lookup.get(
            key
        )

        if rating_record is None:
            continue

        race_class = clean_text(
            rating_record.get(
                "race_class",
                ""
            )
        )

        # IMPORTANT:
        # race_class is populated ONLY from
        # the matching horse form-history row.
        df.at[
            index,
            "race_class"
        ] = race_class

        df.at[
            index,
            "horse_rating_before"
        ] = rating_record.get(
            "horse_rating_before",
            ""
        )

        df.at[
            index,
            "horse_rating_after"
        ] = rating_record.get(
            "horse_rating_after",
            ""
        )

        matched += 1

        if race_class:
            class_matched += 1

    print(
        f"Rating/form cache matched "
        f"{matched} result rows; "
        f"race class populated on "
        f"{class_matched} rows."
    )

    return df


def replace_horse_in_rating_cache(
    cache_df,
    horse_id,
    new_history_df
):
    if cache_df is None:

        cache_df = pd.DataFrame(
            columns=
                HORSE_RATING_CACHE_COLUMNS
        )

    if (
        cache_df.empty
        or
        "horse_id"
        not in cache_df.columns
    ):

        remaining = pd.DataFrame(
            columns=
                HORSE_RATING_CACHE_COLUMNS
        )

    else:

        remaining = cache_df[
            cache_df[
                "horse_id"
            ].astype(
                str
            )
            !=
            str(
                horse_id
            )
        ].copy()

    if (
        new_history_df is None
        or
        new_history_df.empty
    ):
        return remaining

    return pd.concat(
        [
            remaining,
            new_history_df,
        ],
        ignore_index=True
    )


def horse_form_rows_complete_for_results(
    horse_id,
    horse_results_df,
    cache_lookup
):
    """
    Return True when every valid result row for this horse
    already has a matching cached form-history row.

    A non-empty race_class is used as the completeness test.

    Rating itself may legitimately be blank for an unrated
    horse, so rating presence is NOT used.
    """

    found_valid_key = False

    for _, result_row in (
        horse_results_df.iterrows()
    ):

        key = normalise_rating_key(
            horse_id,

            result_row.get(
                "race_date",
                ""
            ),

            result_row.get(
                "race_index",
                ""
            ),
        )

        if key is None:
            continue

        found_valid_key = True

        cached = cache_lookup.get(
            key
        )

        if (
            cached is None
            or
            not clean_text(
                cached.get(
                    "race_class",
                    ""
                )
            )
        ):
            return False

    return found_valid_key


# ============================================================
# ONE HORSE REQUEST -> PROFILE + FORM HISTORY
# ============================================================

def ensure_horse_data(
    results_df,
    horse_master,
    age_refresh_pending,
    rating_cache_df
):
    """
    THE MAIN SPEED IMPROVEMENT.

    For each horse:

        GET Option=1 horse page ONCE

    Then use that exact same HTML response for:

        - horse profile
        - age/origin
        - sire/dam
        - race class history
        - rating before
        - rating after

    There is NO separate rating-page HTTP request.
    """

    if (
        results_df is None
        or
        results_df.empty
    ):
        return rating_cache_df

    unique_horses = (
        results_df[
            [
                "horse_id",
                "horse_name",
            ]
        ]
        .drop_duplicates(
            subset=[
                "horse_id"
            ]
        )
    )

    cache_lookup = (
        build_rating_cache_lookup(
            rating_cache_df
        )
    )

    for _, row in unique_horses.iterrows():

        horse_id = clean_text(
            row.get(
                "horse_id",
                ""
            )
        )

        horse_name = clean_text(
            row.get(
                "horse_name",
                ""
            )
        )

        if not horse_id:
            continue

        existing = horse_master.get(
            horse_id
        )

        force_age_refresh = (
            horse_id
            in
            age_refresh_pending
        )

        need_profile = (
            not horse_profile_is_scraped(
                existing
            )
            or
            force_age_refresh
        )

        horse_results = (
            results_df[
                results_df[
                    "horse_id"
                ].astype(
                    str
                )
                ==
                str(
                    horse_id
                )
            ]
        )

        have_form_rows = (
            horse_form_rows_complete_for_results(
                horse_id,
                horse_results,
                cache_lookup
            )
        )

        need_form = (
            not have_form_rows
        )

        # ----------------------------------------------------
        # We already have everything required.
        # NO HTTP request.
        # ----------------------------------------------------

        if (
            not need_profile
            and
            not need_form
        ):

            print(
                f"Using cached horse data: "
                f"{horse_id} "
                f"{horse_name}"
            )

            continue

        reasons = []

        if need_profile:
            reasons.append(
                "profile"
            )

        if need_form:
            reasons.append(
                "form/rating"
            )

        print(
            f"Opening horse page once for "
            f"{horse_id} "
            f"{horse_name} "
            f"({', '.join(reasons)})"
        )

        response = request_horse_page(
            horse_id
        )

        if response is None:
            continue

        source_url = (
            ensure_horse_option_1(
                response.url
            )
        )

        # ====================================================
        # PROFILE
        # Same HTML response
        # ====================================================

        new_profile = (
            extract_horse_profile(
                response.text,
                horse_id,
                source_url,
                horse_name
            )
        )

        merged_profile = (
            merge_horse_profile(
                existing,
                new_profile
            )
        )

        if merged_profile:

            merged_profile[
                "hemisphere_of_origin"
            ] = get_hemisphere_of_origin(
                merged_profile.get(
                    "country_of_origin",
                    ""
                )
            )

            merged_profile[
                "profile_url"
            ] = ensure_horse_option_1(
                merged_profile.get(
                    "profile_url",
                    build_horse_url(
                        horse_id
                    )
                )
            )

            horse_master[
                horse_id
            ] = merged_profile

            if horse_profile_is_scraped(
                merged_profile
            ):

                age_refresh_pending.discard(
                    horse_id
                )

        # ====================================================
        # FORM / RATINGS / RACE CLASS
        #
        # IMPORTANT:
        # SAME response.text as above.
        # NO second GET.
        # ====================================================

        history_df = (
            extract_horse_rating_history(
                response.text,
                horse_id,
                source_url
            )
        )

        if not history_df.empty:

            class_count = (
                history_df[
                    "race_class"
                ]
                .astype(str)
                .map(
                    clean_text
                )
                .ne("")
                .sum()
            )

            print(
                f"FORM HISTORY FROM SAME PAGE: "
                f"{horse_id} -> "
                f"{len(history_df)} records, "
                f"{class_count} race classes"
            )

            rating_cache_df = (
                replace_horse_in_rating_cache(
                    rating_cache_df,
                    horse_id,
                    history_df
                )
            )

            cache_lookup = (
                build_rating_cache_lookup(
                    rating_cache_df
                )
            )

        else:

            print(
                f"WARNING: no form/rating "
                f"history found for "
                f"{horse_id}"
            )

        # Only one delay because there was only
        # ONE horse HTTP request.
        if DELAY_SECONDS > 0:

            time.sleep(
                DELAY_SECONDS
            )

    return rating_cache_df


# ============================================================
# ENRICH RESULTS FROM HORSE MASTER
# ============================================================

def enrich_results_with_horse_master(
    results_df,
    horse_master
):
    if results_df is None:
        return None

    df = results_df.copy()

    horse_to_result = {
        "brand_number":
            "brand_number",

        "country_of_origin":
            "country_of_origin",

        "hemisphere_of_origin":
            "hemisphere_of_origin",

        "horse_colour":
            "horse_colour",

        "horse_sex":
            "horse_sex",

        "sire":
            "sire",

        "dam":
            "dam",

        "dam_sire":
            "dam_sire",

        "profile_url":
            "horse_profile_url",

        "profile_scraped_at":
            "horse_profile_scraped_at",
    }

    for result_column in (
        horse_to_result.values()
    ):

        if result_column not in df.columns:

            df[
                result_column
            ] = pd.Series(
                [None] * len(df),
                index=df.index,
                dtype="object"
            )

        else:

            df[
                result_column
            ] = df[
                result_column
            ].astype(
                "object"
            )

    if (
        "horse_age_at_race"
        not in df.columns
    ):

        df[
            "horse_age_at_race"
        ] = pd.Series(
            [None] * len(df),
            index=df.index,
            dtype="object"
        )

    else:

        df[
            "horse_age_at_race"
        ] = df[
            "horse_age_at_race"
        ].astype(
            "object"
        )

    for rating_column in [
        "horse_rating_before",
        "horse_rating_after",
    ]:

        if rating_column not in df.columns:

            df[
                rating_column
            ] = pd.Series(
                [None] * len(df),
                index=df.index,
                dtype="object"
            )

    for index, row in df.iterrows():

        horse_id = clean_text(
            row.get(
                "horse_id",
                ""
            )
        )

        if not horse_id:
            continue

        horse = horse_master.get(
            horse_id
        )

        if not horse:
            continue

        horse[
            "hemisphere_of_origin"
        ] = get_hemisphere_of_origin(
            horse.get(
                "country_of_origin",
                ""
            )
        )

        profile_url = clean_text(
            horse.get(
                "profile_url",
                ""
            )
        )

        if profile_url:

            horse[
                "profile_url"
            ] = ensure_horse_option_1(
                profile_url
            )

        for (
            horse_column,
            result_column
        ) in horse_to_result.items():

            df.at[
                index,
                result_column
            ] = horse.get(
                horse_column,
                ""
            )

        age_at_race = (
            calculate_horse_age_at_race(
                current_age=horse.get(
                    "horse_age",
                    ""
                ),

                country_of_origin=horse.get(
                    "country_of_origin",
                    ""
                ),

                profile_scraped_at=horse.get(
                    "profile_scraped_at",
                    ""
                ),

                race_date=row.get(
                    "race_date",
                    ""
                ),
            )
        )

        df.at[
            index,
            "horse_age_at_race"
        ] = age_at_race

    return df


# ============================================================
# EXISTING RESULTS / APPEND
# ============================================================

def backfill_existing_results(
    horse_master
):
    """
    Update existing all_results.csv using locally stored
    static horse-master information.

    No HTTP requests are made here.
    """

    if not os.path.exists(
        RACE_RESULTS_FILE
    ):
        return

    try:

        df = pd.read_csv(
            RACE_RESULTS_FILE,
            dtype=object
        ).fillna("")

    except Exception as exc:

        print(
            "Could not read existing "
            "all_results.csv:",
            exc
        )

        return

    if (
        df.empty
        or
        "horse_id"
        not in df.columns
    ):
        return

    print(
        f"Backfilling static horse "
        f"information and historical age "
        f"onto {len(df)} rows..."
    )

    df = (
        enrich_results_with_horse_master(
            df,
            horse_master
        )
    )

    for column in RACE_COLUMNS:

        if column not in df.columns:

            df[
                column
            ] = pd.Series(
                [None] * len(df),
                index=df.index,
                dtype="object"
            )

    df = df[
        RACE_COLUMNS
    ]

    df.to_csv(
        RACE_RESULTS_FILE,
        index=False
    )

    print(
        "Existing results static "
        "horse data backfilled."
    )


def apply_rating_cache_to_existing_results_file(
    cache_df
):
    """
    Apply locally cached horse form-history data to
    existing all_results.csv.

    IMPORTANT:
    This function NEVER makes HTTP requests.
    """

    if not os.path.exists(
        RACE_RESULTS_FILE
    ):
        return

    try:

        results_df = pd.read_csv(
            RACE_RESULTS_FILE,
            dtype=object
        ).fillna("")

    except Exception as exc:

        print(
            "Could not read "
            "all_results.csv for local "
            "rating/class backfill:",
            exc
        )

        return

    if results_df.empty:
        return

    required_columns = [
        "horse_id",
        "race_date",
        "race_index",
    ]

    for column in required_columns:

        if column not in results_df.columns:
            return

    results_df = (
        apply_rating_cache_to_results(
            results_df,
            cache_df
        )
    )

    if (
        "finish_time"
        in results_df.columns
    ):

        results_df[
            "finish_time"
        ] = results_df[
            "finish_time"
        ].apply(
            clean_finish_time
        )

    if (
        "horse_profile_url"
        in results_df.columns
    ):

        results_df[
            "horse_profile_url"
        ] = results_df[
            "horse_profile_url"
        ].apply(
            lambda value:
                ensure_horse_option_1(
                    value
                )
                if clean_text(value)
                else ""
        )

    for column in RACE_COLUMNS:

        if column not in results_df.columns:

            results_df[
                column
            ] = pd.Series(
                [None] * len(
                    results_df
                ),
                index=results_df.index,
                dtype="object"
            )

    results_df = results_df[
        RACE_COLUMNS
    ]

    results_df.to_csv(
        RACE_RESULTS_FILE,
        index=False
    )

    print(
        "Existing results updated "
        "from LOCAL horse-form cache only."
    )


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
            dtype=object
        )

        return set(
            df[
                "result_id"
            ]
            .dropna()
            .astype(str)
        )

    except Exception as exc:

        print(
            "Could not load "
            "existing result IDs:",
            exc
        )

        return set()


def append_results(
    results_df,
    existing_ids
):
    if (
        results_df is None
        or
        "result_id"
        not in results_df.columns
    ):
        return

    df = results_df.copy()

    df = df[
        ~df[
            "result_id"
        ]
        .astype(str)
        .isin(
            existing_ids
        )
    ].copy()

    if df.empty:

        print(
            "No new result rows."
        )

        return

    for column in RACE_COLUMNS:

        if column not in df.columns:

            df[
                column
            ] = pd.Series(
                [None] * len(df),
                index=df.index,
                dtype="object"
            )

    df = df[
        RACE_COLUMNS
    ]

    file_exists = os.path.exists(
        RACE_RESULTS_FILE
    )

    df.to_csv(
        RACE_RESULTS_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )

    existing_ids.update(
        df[
            "result_id"
        ].astype(
            str
        )
    )

    print(
        f"Added {len(df)} "
        f"new result rows to CSV."
    )


# ============================================================
# PRIZE PAYOUT MODEL
# ============================================================

def get_prize_payout_schedule(
    race_date
):
    if pd.isna(
        race_date
    ):
        return {}

    if (
        race_date
        <
        PAYOUT_MODEL_CUTOFF
    ):
        return {
            1: 0.56,
            2: 0.21,
            3: 0.115,
            4: 0.06,
            5: 0.055,
        }

    return {
        1: 0.56,
        2: 0.21,
        3: 0.115,
        4: 0.06,
        5: 0.035,
        6: 0.02,
    }


def calculate_dead_heat_payout_percentages(
    df
):
    payout_percentages = pd.Series(
        0.0,
        index=df.index,
        dtype="float64"
    )

    if (
        "race_id"
        not in df.columns
    ):
        return payout_percentages

    for (
        race_id,
        race_group
    ) in df.groupby(
        "race_id",
        sort=False,
        dropna=False
    ):

        if race_group.empty:
            continue

        race_date = race_group[
            "_race_date_sort"
        ].iloc[
            0
        ]

        payout_schedule = (
            get_prize_payout_schedule(
                race_date
            )
        )

        race_with_positions = (
            race_group[
                race_group[
                    "_finish_numeric"
                ].notna()
            ]
        )

        if race_with_positions.empty:
            continue

        for (
            finishing_position,
            position_group
        ) in (
            race_with_positions
            .groupby(
                "_finish_numeric",
                sort=True
            )
        ):

            try:
                position = int(
                    finishing_position
                )

            except (
                ValueError,
                TypeError
            ):
                continue

            number_dead_heating = len(
                position_group
            )

            if number_dead_heating == 1:

                payout_percentage = (
                    payout_schedule.get(
                        position,
                        0.0
                    )
                )

            else:

                combined_percentage = sum(
                    payout_schedule.get(
                        position + offset,
                        0.0
                    )
                    for offset in range(
                        number_dead_heating
                    )
                )

                payout_percentage = (
                    combined_percentage
                    /
                    number_dead_heating
                )

                print(
                    "DEAD HEAT PAYOUT:",
                    {
                        "race_id":
                            race_id,

                        "position":
                            position,

                        "horses":
                            number_dead_heating,

                        "combined_percentage":
                            round(
                                combined_percentage,
                                6
                            ),

                        "each_percentage":
                            round(
                                payout_percentage,
                                6
                            ),
                    }
                )

            payout_percentages.loc[
                position_group.index
            ] = payout_percentage

    return payout_percentages


# ============================================================
# HISTORICAL CAREER + PRIZE STATS
# ============================================================

def calculate_historical_career_stats():
    if not os.path.exists(
        RACE_RESULTS_FILE
    ):
        return

    try:

        df = pd.read_csv(
            RACE_RESULTS_FILE,
            dtype=object
        ).fillna("")

    except Exception as exc:

        print(
            "Could not calculate "
            "historical career stats:",
            exc
        )

        return

    if df.empty:
        return

    required_columns = [
        "horse_id",
        "race_id",
        "race_date",
        "race_number",
        "finishing_position",
        "prize_money_hkd",
    ]

    for column in required_columns:

        if column not in df.columns:
            return

    print(
        f"Calculating historical "
        f"career and prize-money "
        f"stats for {len(df)} rows..."
    )

    if (
        "finish_time"
        in df.columns
    ):

        df[
            "finish_time"
        ] = df[
            "finish_time"
        ].apply(
            clean_finish_time
        )

    if (
        "horse_profile_url"
        in df.columns
    ):

        df[
            "horse_profile_url"
        ] = df[
            "horse_profile_url"
        ].apply(
            lambda value:
                ensure_horse_option_1(
                    value
                )
                if clean_text(value)
                else ""
        )

    df[
        "_original_order"
    ] = range(
        len(df)
    )

    df[
        "_race_date_sort"
    ] = pd.to_datetime(
        df[
            "race_date"
        ],
        errors="coerce"
    )

    df[
        "_race_number_sort"
    ] = pd.to_numeric(
        df[
            "race_number"
        ],
        errors="coerce"
    ).fillna(
        0
    )

    df[
        "_finish_numeric"
    ] = pd.to_numeric(
        df[
            "finishing_position"
        ],
        errors="coerce"
    )

    df[
        "_race_prize_numeric"
    ] = pd.to_numeric(
        df[
            "prize_money_hkd"
        ],
        errors="coerce"
    ).fillna(
        0
    )

    sort_columns = [
        "horse_id",
        "_race_date_sort",
        "_race_number_sort",
    ]

    if (
        "race_index"
        in df.columns
    ):

        df[
            "_race_index_sort"
        ] = pd.to_numeric(
            df[
                "race_index"
            ],
            errors="coerce"
        ).fillna(
            0
        )

        sort_columns.append(
            "_race_index_sort"
        )

    df = df.sort_values(
        by=sort_columns,
        kind="stable"
    ).reset_index(
        drop=True
    )

    df[
        "_is_win"
    ] = (
        df[
            "_finish_numeric"
        ] == 1
    ).astype(
        int
    )

    df[
        "_is_second"
    ] = (
        df[
            "_finish_numeric"
        ] == 2
    ).astype(
        int
    )

    df[
        "_is_third"
    ] = (
        df[
            "_finish_numeric"
        ] == 3
    ).astype(
        int
    )

    df[
        "_is_top3"
    ] = (
        df[
            "_finish_numeric"
        ]
        .isin(
            [
                1,
                2,
                3
            ]
        )
    ).astype(
        int
    )

    df[
        "_is_start"
    ] = (
        df[
            "_finish_numeric"
        ]
        .notna()
    ).astype(
        int
    )

    df[
        "prize_payout_percentage"
    ] = (
        calculate_dead_heat_payout_percentages(
            df
        )
    )

    df[
        "prize_money_won_this_race"
    ] = (
        df[
            "_race_prize_numeric"
        ]
        *
        pd.to_numeric(
            df[
                "prize_payout_percentage"
            ],
            errors="coerce"
        ).fillna(
            0
        )
    ).round(
        2
    )

    grouped = df.groupby(
        "horse_id",
        sort=False,
        dropna=False
    )

    df[
        "career_starts_before"
    ] = (
        grouped[
            "_is_start"
        ].cumsum()
        -
        df[
            "_is_start"
        ]
    )

    df[
        "career_wins_before"
    ] = (
        grouped[
            "_is_win"
        ].cumsum()
        -
        df[
            "_is_win"
        ]
    )

    df[
        "career_seconds_before"
    ] = (
        grouped[
            "_is_second"
        ].cumsum()
        -
        df[
            "_is_second"
        ]
    )

    df[
        "career_thirds_before"
    ] = (
        grouped[
            "_is_third"
        ].cumsum()
        -
        df[
            "_is_third"
        ]
    )

    df[
        "career_top3_before"
    ] = (
        grouped[
            "_is_top3"
        ].cumsum()
        -
        df[
            "_is_top3"
        ]
    )

    df[
        "career_prize_money_after"
    ] = (
        grouped[
            "prize_money_won_this_race"
        ]
        .cumsum()
    ).round(
        2
    )

    df[
        "career_prize_money_before"
    ] = (
        df[
            "career_prize_money_after"
        ]
        -
        df[
            "prize_money_won_this_race"
        ]
    ).round(
        2
    )

    starts = pd.to_numeric(
        df[
            "career_starts_before"
        ],
        errors="coerce"
    ).fillna(
        0
    )

    wins = pd.to_numeric(
        df[
            "career_wins_before"
        ],
        errors="coerce"
    ).fillna(
        0
    )

    top3 = pd.to_numeric(
        df[
            "career_top3_before"
        ],
        errors="coerce"
    ).fillna(
        0
    )

    df[
        "career_win_rate_before"
    ] = 0.0

    df[
        "career_top3_rate_before"
    ] = 0.0

    has_starts = (
        starts > 0
    )

    df.loc[
        has_starts,
        "career_win_rate_before"
    ] = (
        wins[
            has_starts
        ]
        /
        starts[
            has_starts
        ]
    ).round(
        4
    )

    df.loc[
        has_starts,
        "career_top3_rate_before"
    ] = (
        top3[
            has_starts
        ]
        /
        starts[
            has_starts
        ]
    ).round(
        4
    )

    df = df.sort_values(
        "_original_order",
        kind="stable"
    )

    df = df.drop(
        columns=[
            "_original_order",
            "_race_date_sort",
            "_race_number_sort",
            "_race_index_sort",
            "_finish_numeric",
            "_race_prize_numeric",
            "_is_start",
            "_is_win",
            "_is_second",
            "_is_third",
            "_is_top3",
        ],
        errors="ignore"
    )

    for column in RACE_COLUMNS:

        if column not in df.columns:

            df[
                column
            ] = pd.Series(
                [None] * len(df),
                index=df.index,
                dtype="object"
            )

    df = df[
        RACE_COLUMNS
    ]

    df.to_csv(
        RACE_RESULTS_FILE,
        index=False
    )

    print(
        "Historical career statistics, "
        "dead-heat payouts and prize "
        "money updated."
    )


# ============================================================
# DAILY CHECKPOINT
# ============================================================

def save_daily_checkpoint(
    meeting_date,
    horse_master,
    rating_cache_df
):
    """
    Fast checkpoint.

    Only save the in-memory horse datasets.

    Do NOT:
        - reread all_results.csv
        - recalculate all career stats
        - scrape horse pages again
        - perform a separate rating backfill
    """

    print()
    print(
        "=" * 70
    )

    print(
        f"DAILY CHECKPOINT: "
        f"{meeting_date.strftime('%Y-%m-%d')}"
    )

    print(
        "=" * 70
    )

    save_horse_master(
        horse_master
    )

    save_horse_rating_cache(
        rating_cache_df
    )

    print(
        f"Checkpoint complete: "
        f"{meeting_date.strftime('%Y-%m-%d')}"
    )


# ============================================================
# PROCESS ONE DATE
# ============================================================

def process_date(
    meeting_date,
    existing_result_ids,
    horse_master,
    age_refresh_pending,
    rating_cache_df
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

    response = request_race_page(
        meeting_date
    )

    if response is None:
        return rating_cache_df

    meeting = detect_meeting(
        response.text
    )

    if meeting is None:

        print(
            "No HKJC meeting detected."
        )

        return rating_cache_df

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

        return rating_cache_df

    print(
        "Races:",
        race_numbers
    )

    completed_races = 0

    for race_no in race_numbers:

        print()

        print(
            f"Processing Race "
            f"{race_no}"
        )

        race_url = build_url(
            meeting_date,
            racecourse_code,
            race_no
        )

        race_response = (
            request_race_page(
                meeting_date,
                racecourse_code,
                race_no
            )
        )

        if race_response is None:
            continue

        metadata = (
            extract_race_metadata(
                race_response.text,
                meeting_date,
                racecourse_code,
                racecourse_name,
                race_no,
                race_url
            )
        )

        results_df = (
            extract_results(
                race_response.text,
                metadata
            )
        )

        if results_df is None:

            print(
                "No horse results found."
            )

            continue

        print(
            f"Found "
            f"{len(results_df)} "
            f"horse result rows."
        )

        # ====================================================
        # ONE HORSE PAGE REQUEST
        #
        # The same response supplies:
        #   - profile
        #   - race class history
        #   - rating before
        #   - rating after
        # ====================================================

        rating_cache_df = (
            ensure_horse_data(
                results_df,
                horse_master,
                age_refresh_pending,
                rating_cache_df
            )
        )

        # Static horse information
        results_df = (
            enrich_results_with_horse_master(
                results_df,
                horse_master
            )
        )

        # ====================================================
        # IMPORTANT:
        #
        # race_class is populated ONLY from:
        #
        # horse_id
        # + race_date
        # + race_index
        #
        # matched against the horse form-history table.
        #
        # It is NOT taken from the race header.
        # ====================================================

        results_df = (
            apply_rating_cache_to_results(
                results_df,
                rating_cache_df
            )
        )

        append_results(
            results_df,
            existing_result_ids
        )

        completed_races += 1

        print(
            f"Race "
            f"{race_no} "
            f"saved."
        )

        if DELAY_SECONDS > 0:

            time.sleep(
                DELAY_SECONDS
            )

    if completed_races > 0:

        print()

        print(
            f"Completed "
            f"{completed_races} races "
            f"for "
            f"{meeting_date.strftime('%Y-%m-%d')}"
        )

        save_daily_checkpoint(
            meeting_date,
            horse_master,
            rating_cache_df
        )

    return rating_cache_df


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_folders()

    try:

        start_date = (
            datetime.strptime(
                START_DATE,
                "%Y-%m-%d"
            )
            .date()
        )

        end_date = (
            datetime.strptime(
                END_DATE,
                "%Y-%m-%d"
            )
            .date()
        )

    except ValueError:

        print(
            "Dates must use YYYY-MM-DD."
        )

        return

    if (
        start_date
        >
        end_date
    ):

        print(
            "START_DATE cannot "
            "be after END_DATE."
        )

        return

    (
        horse_master,
        age_refresh_pending
    ) = load_horse_master()

    rating_cache_df = (
        load_horse_rating_cache()
    )

    existing_result_ids = (
        load_existing_result_ids()
    )

    print(
        "Horse master records:",
        len(
            horse_master
        )
    )

    print(
        "Horse form/rating cache rows:",
        len(
            rating_cache_df
        )
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
        "Existing results:",
        len(
            existing_result_ids
        )
    )

    print(
        "Existing horses:",
        len(
            horse_master
        )
    )

    print(
        "Horse profiles needing "
        "age refresh:",
        len(
            age_refresh_pending
        )
    )

    for meeting_date in date_range(
        start_date,
        end_date
    ):

        try:

            rating_cache_df = (
                process_date(
                    meeting_date,
                    existing_result_ids,
                    horse_master,
                    age_refresh_pending,
                    rating_cache_df
                )
            )

        except Exception as exc:

            print()

            print(
                "ERROR while processing "
                f"{meeting_date}:",
                exc
            )

            print(
                "Performing emergency "
                "save of in-memory "
                "horse data..."
            )

            try:

                save_horse_master(
                    horse_master
                )

                save_horse_rating_cache(
                    rating_cache_df
                )

                print(
                    "Emergency save complete."
                )

            except Exception as save_exc:

                print(
                    "Emergency save failed:",
                    save_exc
                )

            continue

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL SAVE"
    )

    print(
        "=" * 70
    )

    # Save local horse datasets.
    save_horse_master(
        horse_master
    )

    save_horse_rating_cache(
        rating_cache_df
    )

    # ========================================================
    # BULK OPERATIONS
    #
    # Do them ONCE at the end rather than after every
    # meeting date.
    #
    # None of these functions makes horse-page HTTP requests.
    # ========================================================

    backfill_existing_results(
        horse_master
    )

    apply_rating_cache_to_existing_results_file(
        rating_cache_df
    )

    calculate_historical_career_stats()

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

    print(
        "Horse rating/form cache:",
        HORSE_RATINGS_CACHE_FILE
    )

    if age_refresh_pending:

        print(
            "Horse profiles still "
            "awaiting age refresh:",
            len(
                age_refresh_pending
            )
        )


if __name__ == "__main__":
    main()
