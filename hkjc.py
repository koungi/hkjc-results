import os
import re
import time
from datetime import datetime, timedelta, timezone
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

HORSE_BASE_URL = (
    "https://racing.hkjc.com/en-us/local/information/horse"
)

START_DATE = os.getenv(
    "START_DATE",
    "2006-01-01"
)

END_DATE = os.getenv(
    "END_DATE",
    "2006-12-31"
)

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

PAYOUT_MODEL_CUTOFF = pd.Timestamp(
    "2023-09-10"
)


# ============================================================
# HORSE MASTER
#
# Only relatively static horse information.
# ============================================================

HORSE_COLUMNS = [
    "horse_id",
    "horse_name",
    "brand_number",

    "country_of_origin",
    "horse_colour",
    "horse_sex",

    "import_type",
    "import_date",

    "owner",

    "sire",
    "dam",
    "dam_sire",

    "profile_url",
    "profile_scraped",
    "profile_scraped_at",
]


# ============================================================
# ALL RESULTS
# ============================================================

RACE_COLUMNS = [
    # IDs
    "result_id",
    "race_id",

    # Race information
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

    # Horse identity
    "horse_id",
    "horse_number",
    "horse_name",

    # Static horse master information
    "brand_number",
    "country_of_origin",
    "horse_colour",
    "horse_sex",
    "import_type",
    "import_date",
    "owner",
    "sire",
    "dam",
    "dam_sire",

    # Historical career stats BEFORE this race
    "career_starts_before",
    "career_wins_before",
    "career_seconds_before",
    "career_thirds_before",
    "career_top3_before",
    "career_win_rate_before",
    "career_top3_rate_before",

    # Prize calculations
    "prize_payout_percentage",
    "prize_money_won_this_race",
    "career_prize_money_before",
    "career_prize_money_after",

    # Race result
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

    # URLs / audit
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
        current += timedelta(days=1)


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
    return (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
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

    text = clean_text(
        value
    ).replace(
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

    match = re.search(
        r"HK\$\s*([\d,]+)",
        text,
        re.I
    )

    if not match:
        return None

    try:
        return int(
            match.group(1)
            .replace(",", "")
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


# ============================================================
# URL BUILDERS
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


def build_horse_url(
    horse_id
):
    return (
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
# RACE NUMBER DETECTION
# ============================================================

def detect_race_numbers(html):
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
                int(
                    match.group(1)
                )
            )

    return sorted(
        race_numbers
    )


# ============================================================
# RACE HEADER HELPERS
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

            if cells[next_index]:
                return cells[
                    next_index
                ]

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
            "WARNING: race header not found"
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

    # --------------------------------------------------------
    # Race number / index
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Parse rows
    # --------------------------------------------------------

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
                        distance_match
                        .group(1)
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

        # Going

        going = extract_label_value(
            cells,
            "Going"
        )

        if going:
            metadata[
                "going"
            ] = going

        # Course

        course = extract_label_value(
            cells,
            "Course"
        )

        if course:
            metadata[
                "course"
            ] = course

        # Race name

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

        # Prize

        for cell in cells:

            prize = parse_prize_money(
                cell
            )

            if prize is not None:

                metadata[
                    "prize_money_hkd"
                ] = prize

    # --------------------------------------------------------
    # Fallbacks
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Surface
    # --------------------------------------------------------

    course_upper = clean_text(
        metadata["course"]
    ).upper()

    if "TURF" in course_upper:

        metadata[
            "surface"
        ] = "TURF"

    elif (
        "ALL WEATHER" in course_upper
        or
        "AWT" in course_upper
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

            "race_class":
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
# RACE RESULTS
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
        "table.performance"
    )

    if table is None:

        table = soup.select_one(
            ".performance table"
        )

    if table is None:

        print(
            "WARNING: results table not found"
        )

        return None

    rows = table.select(
        "tbody tr"
    )

    if not rows:

        rows = table.find_all(
            "tr"
        )

    valid_rows = []

    for row in rows:

        cells = row.find_all(
            "td"
        )

        if len(cells) >= 10:

            valid_rows.append(
                row
            )

    field_size = len(
        valid_rows
    )

    results = []

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
        horse_url = ""

        if horse_link:

            href = horse_link.get(
                "href",
                ""
            )

            horse_id = extract_horse_id(
                href
            )

            horse_url = urljoin(
                "https://racing.hkjc.com",
                href
            )

        finishing_position = parse_integer(
            cells[0].get_text(
                " ",
                strip=True
            )
        )

        horse_number = parse_integer(
            cells[1].get_text(
                " ",
                strip=True
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

        actual_weight = parse_integer(
            cells[5].get_text(
                " ",
                strip=True
            )
        )

        declared_horse_weight = parse_integer(
            cells[6].get_text(
                " ",
                strip=True
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

            "race_class":
                race_metadata[
                    "race_class"
                ],

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
                jockey,

            "trainer":
                trainer,

            "actual_weight":
                actual_weight,

            "declared_horse_weight":
                declared_horse_weight,

            "draw":
                draw,

            "margin":
                margin,

            "finish_time":
                finish_time,

            "odds":
                odds,

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
# HORSE PROFILE LABELS
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
        for item in PROFILE_LABELS
        if item.lower()
        != label.lower()
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


# ============================================================
# HORSE PROFILE PARSER
# ============================================================

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

    # --------------------------------------------------------
    # Horse name / brand
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Country of origin
    #
    # Ignore age because age changes over time.
    # --------------------------------------------------------

    origin_age = extract_profile_value_any(
        text,
        [
            "Country of Origin / Age",
            "Country of Origin",
        ]
    )

    if origin_age:

        parts = [
            clean_text(item)
            for item in origin_age.split(
                "/"
            )
        ]

        if parts:

            profile[
                "country_of_origin"
            ] = parts[0]

    # --------------------------------------------------------
    # Colour / sex
    # --------------------------------------------------------

    colour_sex = extract_profile_value(
        text,
        "Colour / Sex"
    )

    if colour_sex:

        parts = [
            clean_text(item)
            for item in colour_sex.split(
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

    # --------------------------------------------------------
    # Static fields
    # --------------------------------------------------------

    profile[
        "import_type"
    ] = extract_profile_value(
        text,
        "Import Type"
    )

    profile[
        "import_date"
    ] = extract_profile_value(
        text,
        "Import Date"
    )

    profile[
        "owner"
    ] = extract_profile_value(
        text,
        "Owner"
    )

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

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    useful_fields = [
        profile.get(
            "country_of_origin"
        ),
        profile.get(
            "horse_colour"
        ),
        profile.get(
            "horse_sex"
        ),
        profile.get(
            "owner"
        ),
        profile.get(
            "sire"
        ),
        profile.get(
            "dam"
        ),
    ]

    success = any(
        clean_text(item)
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

            "colour":
                profile[
                    "horse_colour"
                ],

            "sex":
                profile[
                    "horse_sex"
                ],

            "import_type":
                profile[
                    "import_type"
                ],

            "import_date":
                profile[
                    "import_date"
                ],

            "owner":
                profile[
                    "owner"
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
# LOAD HORSE MASTER
# ============================================================

def load_horse_master():
    horse_master = {}

    if not os.path.exists(
        HORSE_MASTER_FILE
    ):

        return horse_master

    try:

        df = pd.read_csv(
            HORSE_MASTER_FILE,
            dtype=object
        ).fillna("")

    except Exception as exc:

        print(
            "Could not read horse master:",
            exc
        )

        return horse_master

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

        horse_master[
            horse_id
        ] = {
            column: row.get(
                column,
                ""
            )
            for column in HORSE_COLUMNS
        }

    return horse_master


# ============================================================
# SAVE HORSE MASTER
# ============================================================

def save_horse_master(
    horse_master
):
    if not horse_master:
        return

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


# ============================================================
# PROFILE ALREADY SCRAPED?
# ============================================================

def horse_profile_is_scraped(
    horse
):
    if not horse:
        return False

    value = clean_text(
        horse.get(
            "profile_scraped",
            ""
        )
    ).lower()

    return value in {
        "true",
        "1",
        "yes",
    }


# ============================================================
# ENSURE HORSE PROFILE EXISTS
# ============================================================

def ensure_horse_profiles(
    results_df,
    horse_master
):
    if results_df is None:
        return

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

        # ----------------------------------------------------
        # Reuse horse master
        # ----------------------------------------------------

        if horse_profile_is_scraped(
            existing
        ):

            print(
                f"Using horse master: "
                f"{horse_id} "
                f"{horse_name}"
            )

            continue

        # ----------------------------------------------------
        # Scrape only new / incomplete horses
        # ----------------------------------------------------

        print(
            f"Scraping horse: "
            f"{horse_id} "
            f"{horse_name}"
        )

        response = request_horse_page(
            horse_id
        )

        if response is None:
            continue

        profile = extract_horse_profile(
            response.text,
            horse_id,
            response.url,
            horse_name
        )

        horse_master[
            horse_id
        ] = profile

        # Save immediately for resumability

        save_horse_master(
            horse_master
        )

        time.sleep(
            DELAY_SECONDS
        )


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

        "horse_colour":
            "horse_colour",

        "horse_sex":
            "horse_sex",

        "import_type":
            "import_type",

        "import_date":
            "import_date",

        "owner":
            "owner",

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
            ] = (
                df[
                    result_column
                ]
                .astype(
                    "object"
                )
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

    return df


# ============================================================
# BACKFILL EXISTING RESULTS
# ============================================================

def backfill_existing_results(
    horse_master
):
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

    if df.empty:
        return

    if "horse_id" not in df.columns:

        print(
            "Existing results does "
            "not contain horse_id."
        )

        return

    print(
        f"Backfilling static horse "
        f"information onto "
        f"{len(df)} rows..."
    )

    df = enrich_results_with_horse_master(
        df,
        horse_master
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
        "Existing results backfilled."
    )


# ============================================================
# LOAD EXISTING RESULT IDS
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


# ============================================================
# APPEND RESULTS
# ============================================================

def append_results(
    results_df,
    existing_ids
):
    if results_df is None:
        return

    df = results_df.copy()

    if "result_id" not in df.columns:
        return

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
        ]
        .astype(str)
    )

    print(
        f"Added {len(df)} "
        f"new result rows."
    )


# ============================================================
# PAYOUT PERCENTAGE
# ============================================================

def get_prize_payout_percentage(
    race_date,
    finishing_position
):
    if pd.isna(race_date):
        return 0.0

    if pd.isna(finishing_position):
        return 0.0

    try:
        position = int(
            finishing_position
        )

    except (
        ValueError,
        TypeError
    ):
        return 0.0

    # --------------------------------------------------------
    # HISTORICAL MODEL
    # Before 10 September 2023
    # --------------------------------------------------------

    if race_date < PAYOUT_MODEL_CUTOFF:

        payout = {
            1: 0.56,
            2: 0.21,
            3: 0.115,
            4: 0.06,
            5: 0.055,
        }

        return payout.get(
            position,
            0.0
        )

    # --------------------------------------------------------
    # PRESENT-DAY MODEL
    # 10 September 2023 onwards
    # --------------------------------------------------------

    payout = {
        1: 0.56,
        2: 0.21,
        3: 0.115,
        4: 0.06,
        5: 0.035,
        6: 0.02,
    }

    return payout.get(
        position,
        0.0
    )


# ============================================================
# HISTORICAL CAREER + PRIZE STATS
#
# All "before" fields represent information available
# BEFORE the race on that row.
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
        "race_date",
        "race_number",
        "finishing_position",
        "prize_money_hkd",
    ]

    for column in required_columns:

        if column not in df.columns:

            print(
                "Career stats cannot "
                f"be calculated. Missing: "
                f"{column}"
            )

            return

    print(
        f"Calculating historical career "
        f"and prize-money stats for "
        f"{len(df)} rows..."
    )

    # --------------------------------------------------------
    # Preserve current CSV order
    # --------------------------------------------------------

    df[
        "_original_order"
    ] = range(
        len(df)
    )

    # --------------------------------------------------------
    # Parse values
    # --------------------------------------------------------

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
    ).fillna(0)

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
    ).fillna(0)

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    sort_columns = [
        "horse_id",
        "_race_date_sort",
        "_race_number_sort",
    ]

    if "race_index" in df.columns:

        df[
            "_race_index_sort"
        ] = pd.to_numeric(
            df[
                "race_index"
            ],
            errors="coerce"
        ).fillna(0)

        sort_columns.append(
            "_race_index_sort"
        )

    df = df.sort_values(
        by=sort_columns,
        kind="stable"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Result indicators
    # --------------------------------------------------------

    df[
        "_is_win"
    ] = (
        df[
            "_finish_numeric"
        ] == 1
    ).astype(int)

    df[
        "_is_second"
    ] = (
        df[
            "_finish_numeric"
        ] == 2
    ).astype(int)

    df[
        "_is_third"
    ] = (
        df[
            "_finish_numeric"
        ] == 3
    ).astype(int)

    df[
        "_is_top3"
    ] = (
        df[
            "_finish_numeric"
        ]
        .isin(
            [1, 2, 3]
        )
    ).astype(int)

    df[
        "_is_start"
    ] = (
        df[
            "_finish_numeric"
        ]
        .notna()
    ).astype(int)

    # --------------------------------------------------------
    # Prize percentage for each race
    # --------------------------------------------------------

    df[
        "prize_payout_percentage"
    ] = df.apply(
        lambda row:
            get_prize_payout_percentage(
                row[
                    "_race_date_sort"
                ],
                row[
                    "_finish_numeric"
                ]
            ),
        axis=1
    )

    # --------------------------------------------------------
    # Prize money won this race
    # --------------------------------------------------------

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
        ).fillna(0)
    ).round(2)

    # --------------------------------------------------------
    # Group by horse
    # --------------------------------------------------------

    grouped = df.groupby(
        "horse_id",
        sort=False,
        dropna=False
    )

    # --------------------------------------------------------
    # Career starts before
    # --------------------------------------------------------

    df[
        "career_starts_before"
    ] = (
        grouped[
            "_is_start"
        ]
        .cumsum()
        -
        df[
            "_is_start"
        ]
    )

    # --------------------------------------------------------
    # Wins before
    # --------------------------------------------------------

    df[
        "career_wins_before"
    ] = (
        grouped[
            "_is_win"
        ]
        .cumsum()
        -
        df[
            "_is_win"
        ]
    )

    # --------------------------------------------------------
    # Seconds before
    # --------------------------------------------------------

    df[
        "career_seconds_before"
    ] = (
        grouped[
            "_is_second"
        ]
        .cumsum()
        -
        df[
            "_is_second"
        ]
    )

    # --------------------------------------------------------
    # Thirds before
    # --------------------------------------------------------

    df[
        "career_thirds_before"
    ] = (
        grouped[
            "_is_third"
        ]
        .cumsum()
        -
        df[
            "_is_third"
        ]
    )

    # --------------------------------------------------------
    # Top 3 before
    # --------------------------------------------------------

    df[
        "career_top3_before"
    ] = (
        grouped[
            "_is_top3"
        ]
        .cumsum()
        -
        df[
            "_is_top3"
        ]
    )

    # --------------------------------------------------------
    # Prize money after current race
    # --------------------------------------------------------

    df[
        "career_prize_money_after"
    ] = (
        grouped[
            "prize_money_won_this_race"
        ]
        .cumsum()
    ).round(2)

    # --------------------------------------------------------
    # Prize money before current race
    # --------------------------------------------------------

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
    ).round(2)

    # --------------------------------------------------------
    # Career rates BEFORE race
    # --------------------------------------------------------

    starts = pd.to_numeric(
        df[
            "career_starts_before"
        ],
        errors="coerce"
    ).fillna(0)

    wins = pd.to_numeric(
        df[
            "career_wins_before"
        ],
        errors="coerce"
    ).fillna(0)

    top3 = pd.to_numeric(
        df[
            "career_top3_before"
        ],
        errors="coerce"
    ).fillna(0)

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
    ).round(4)

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
    ).round(4)

    # --------------------------------------------------------
    # Restore original CSV order
    # --------------------------------------------------------

    df = df.sort_values(
        "_original_order",
        kind="stable"
    )

    # --------------------------------------------------------
    # Drop helper fields
    # --------------------------------------------------------

    temporary_columns = [
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
    ]

    df = df.drop(
        columns=temporary_columns,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Enforce final schema
    # --------------------------------------------------------

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
        "Historical career statistics "
        "and prize money updated."
    )


# ============================================================
# PROCESS ONE DATE
# ============================================================

def process_date(
    meeting_date,
    existing_result_ids,
    horse_master
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

        race_response = request_race_page(
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
            f"Found "
            f"{len(results_df)} "
            f"horse result rows."
        )

        # ----------------------------------------------------
        # 1. Scrape only new horse profiles
        # ----------------------------------------------------

        ensure_horse_profiles(
            results_df,
            horse_master
        )

        # ----------------------------------------------------
        # 2. Enrich from horse master
        # ----------------------------------------------------

        results_df = enrich_results_with_horse_master(
            results_df,
            horse_master
        )

        # ----------------------------------------------------
        # 3. Append race rows
        # ----------------------------------------------------

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

    if start_date > end_date:

        print(
            "START_DATE cannot "
            "be after END_DATE."
        )

        return

    # --------------------------------------------------------
    # Load horse master
    # --------------------------------------------------------

    horse_master = load_horse_master()

    print(
        "Horse master records:",
        len(
            horse_master
        )
    )

    # --------------------------------------------------------
    # Rewrite horse master to current clean schema
    # --------------------------------------------------------

    if horse_master:

        save_horse_master(
            horse_master
        )

    # --------------------------------------------------------
    # Backfill old result rows
    # --------------------------------------------------------

    backfill_existing_results(
        horse_master
    )

    # --------------------------------------------------------
    # Recalculate any existing historical stats
    # --------------------------------------------------------

    calculate_historical_career_stats()

    # --------------------------------------------------------
    # Existing result IDs
    # --------------------------------------------------------

    existing_result_ids = (
        load_existing_result_ids()
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

    # --------------------------------------------------------
    # Process dates
    # --------------------------------------------------------

    for meeting_date in date_range(
        start_date,
        end_date
    ):

        process_date(
            meeting_date,
            existing_result_ids,
            horse_master
        )

        time.sleep(
            DELAY_SECONDS
        )

    # --------------------------------------------------------
    # Save master
    # --------------------------------------------------------

    save_horse_master(
        horse_master
    )

    # --------------------------------------------------------
    # Final backfill of static horse data
    # --------------------------------------------------------

    backfill_existing_results(
        horse_master
    )

    # --------------------------------------------------------
    # Final historical career + prize recalculation
    # --------------------------------------------------------

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


if __name__ == "__main__":
    main()
