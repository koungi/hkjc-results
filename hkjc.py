import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qsl, urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://racing.hkjc.com/en-us/local/information/archive/localresults"
HORSE_BASE_URL = "https://racing.hkjc.com/en-us/local/information/horse"

START_DATE = os.getenv("START_DATE", "2006-01-01")
END_DATE = os.getenv("END_DATE", "2006-12-31")

DATE_WORKERS = max(0, int(os.getenv("DATE_WORKERS", "16")))
RACE_WORKERS = max(0, int(os.getenv("RACE_WORKERS", "20")))
HORSE_WORKERS = max(
    0,
    int(
        os.getenv(
            "HORSE_WORKERS",
            os.getenv("MAX_HORSE_WORKERS", "20"),
        )
    ),
)

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
HORSE_REQUEST_TIMEOUT = float(os.getenv("HORSE_REQUEST_TIMEOUT", "60"))
HTTP_RETRIES = max(0, int(os.getenv("HTTP_RETRIES", "3")))
HTTP_BACKOFF_FACTOR = float(os.getenv("HTTP_BACKOFF_FACTOR", "1.0"))

RESULTS_DIR = "results"
RACES_DIR = os.path.join(RESULTS_DIR, "races")
HORSES_DIR = os.path.join(RESULTS_DIR, "horses")

RACE_RESULTS_FILE = os.path.join(RACES_DIR, "all_results.csv")
HORSE_MASTER_FILE = os.path.join(HORSES_DIR, "horse_master.csv")
HORSE_RATINGS_CACHE_FILE = os.path.join(
    HORSES_DIR,
    "horse_ratings_cache.csv",
)

PAYOUT_MODEL_CUTOFF = pd.Timestamp("2023-09-10")


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


COMMON_HEADERS = {
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
}


def create_http_session(pool_size=1):
    session = requests.Session()

    session.headers.update(
        COMMON_HEADERS
    )

    retry = Retry(
        total=HTTP_RETRIES,
        connect=HTTP_RETRIES,
        read=HTTP_RETRIES,
        status=HTTP_RETRIES,
        backoff_factor=HTTP_BACKOFF_FACTOR,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=frozenset(
            ["GET"]
        ),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=max(
            1,
            pool_size,
        ),
        pool_maxsize=max(
            1,
            pool_size,
        ),
    )

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    return session


_thread_local = threading.local()


def get_worker_session():
    session = getattr(
        _thread_local,
        "worker_session",
        None,
    )

    if session is None:
        session = create_http_session(
            pool_size=2
        )

        _thread_local.worker_session = (
            session
        )

    return session


def ensure_folders():
    os.makedirs(
        RACES_DIR,
        exist_ok=True,
    )

    os.makedirs(
        HORSES_DIR,
        exist_ok=True,
    )


def date_range(
    start_date,
    end_date,
):
    current = start_date

    while current <= end_date:
        yield current

        current += timedelta(
            days=1
        )


def choose_worker_count(
    configured_workers,
    task_count,
):
    if task_count <= 0:
        return 0

    if configured_workers <= 0:
        return task_count

    return max(
        1,
        min(
            configured_workers,
            task_count,
        ),
    )


def clean_text(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).replace(
            "\xa0",
            " ",
        ),
    ).strip()


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
        clean_text(value),
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

    try:
        return float(
            clean_text(value).replace(
                ",",
                "",
            )
        )

    except ValueError:
        return None


def parse_prize_money(value):
    if not value:
        return None

    match = re.search(
        r"(?:HK\s*)?\$\s*([\d,]+)",
        clean_text(value),
        re.I,
    )

    if not match:
        return None

    try:
        return int(
            match.group(1).replace(
                ",",
                "",
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
        re.I,
    )

    return (
        clean_text(
            match.group(1)
        )
        if match
        else ""
    )


def clean_finish_time(value):
    text = clean_text(
        value
    )

    if not text:
        return None

    normalised = text.replace(
        " ",
        "",
    )

    if re.fullmatch(
        r"0+(?:(?::|\.)0+)+",
        normalised,
    ):
        return None

    return text


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
    country_of_origin,
):
    country = clean_text(
        country_of_origin
    ).upper()

    if (
        country
        in
        SOUTHERN_HEMISPHERE_ORIGINS
    ):
        return "Southern"

    if (
        country
        in
        NORTHERN_HEMISPHERE_ORIGINS
    ):
        return "Northern"

    return "Other"


def get_official_horse_birthday(
    country_of_origin,
):
    return (
        (8, 1)
        if (
            get_hemisphere_of_origin(
                country_of_origin
            )
            ==
            "Southern"
        )
        else
        (1, 1)
    )


def parse_date_only(value):
    text = clean_text(
        value
    )

    if not text:
        return None

    parsed = pd.to_datetime(
        text,
        errors="coerce",
        utc=True,
    )

    return (
        None
        if pd.isna(parsed)
        else parsed.date()
    )


def infer_horse_birth_year(
    current_age,
    country_of_origin,
    profile_scraped_at,
):
    age = parse_integer(
        current_age
    )

    scraped_date = (
        parse_date_only(
            profile_scraped_at
        )
    )

    if (
        age is None
        or
        scraped_date is None
    ):
        return None

    (
        month,
        day,
    ) = get_official_horse_birthday(
        country_of_origin
    )

    birthday_this_year = (
        scraped_date.replace(
            month=month,
            day=day,
        )
    )

    reference_year = (
        scraped_date.year - 1
        if (
            scraped_date
            <
            birthday_this_year
        )
        else scraped_date.year
    )

    return (
        reference_year
        -
        age
    )


def calculate_horse_age_at_race(
    current_age,
    country_of_origin,
    profile_scraped_at,
    race_date,
):
    birth_year = (
        infer_horse_birth_year(
            current_age,
            country_of_origin,
            profile_scraped_at,
        )
    )

    race_date = parse_date_only(
        race_date
    )

    if (
        birth_year is None
        or
        race_date is None
    ):
        return None

    (
        month,
        day,
    ) = get_official_horse_birthday(
        country_of_origin
    )

    age = (
        race_date.year
        -
        birth_year
    )

    if (
        race_date.month,
        race_date.day,
    ) < (
        month,
        day,
    ):
        age -= 1

    return (
        age
        if age >= 0
        else None
    )


def ensure_horse_option_1(url):
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
            value,
        )
        for (
            key,
            value,
        ) in parse_qsl(
            parts.query,
            keep_blank_values=True,
        )
        if (
            key.lower()
            !=
            "option"
        )
    ]

    query_pairs.append(
        (
            "Option",
            "1",
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
    race_no=None,
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
    horse_id,
):
    return ensure_horse_option_1(
        f"{HORSE_BASE_URL}"
        f"?horseid="
        f"{horse_id}"
    )


def request_race_page(
    race_date,
    racecourse=None,
    race_no=None,
):
    url = build_url(
        race_date,
        racecourse,
        race_no,
    )

    try:
        response = (
            get_worker_session()
            .get(
                url,
                timeout=
                    REQUEST_TIMEOUT,
            )
        )

        print(
            f"GET "
            f"{url} "
            f"-> "
            f"{response.status_code}"
        )

        response.raise_for_status()

        return response

    except requests.RequestException as exc:
        print(
            f"Race request failed "
            f"{url}: "
            f"{exc}"
        )

        return None


def request_horse_page(
    horse_id,
):
    url = build_horse_url(
        horse_id
    )

    try:
        response = (
            get_worker_session()
            .get(
                url,
                timeout=
                    HORSE_REQUEST_TIMEOUT,
                allow_redirects=True,
            )
        )

        print(
            f"HORSE "
            f"{horse_id} "
            f"-> "
            f"{response.status_code}"
        )

        response.raise_for_status()

        return response

    except requests.RequestException as exc:
        print(
            f"Horse request failed "
            f"{horse_id}: "
            f"{exc}"
        )

        return None


def detect_meeting(html):
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    element = soup.select_one(
        ".raceMeeting_select"
    )

    if element is None:
        return None

    text = clean_text(
        element.get_text(
            " ",
            strip=True,
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


def detect_race_numbers(html):
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    race_numbers = set()

    for text in soup.stripped_strings:
        match = re.search(
            r"\bRACE\s+(\d+)\b",
            text,
            re.I,
        )

        if match:
            race_numbers.add(
                int(
                    match.group(1)
                )
            )

    for link in soup.find_all(
        "a",
        href=True,
    ):
        match = re.search(
            r"[?&]RaceNo=(\d+)",
            link.get(
                "href",
                "",
            ),
            re.I,
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


def get_cell_texts(row):
    return [
        clean_text(
            cell.get_text(
                " ",
                strip=True,
            )
        )
        for cell in row.find_all(
            [
                "td",
                "th",
            ]
        )
    ]


def extract_label_value(
    cells,
    label,
):
    pattern = re.compile(
        rf"\b"
        rf"{re.escape(label)}"
        rf"\s*:",
        re.I,
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
            cell,
        ).strip()

        if same_cell:
            return same_cell

        for next_index in range(
            index + 1,
            len(cells),
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
    race_url,
):
    soup = BeautifulSoup(
        html,
        "html.parser",
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

    header = (
        soup.select_one(
            "table.race_tab"
        )
        or
        soup.select_one(
            ".race_tab"
        )
    )

    if header is None:
        print(
            f"NO RACE HEADER: "
            f"{meeting_date} "
            f"{racecourse_code} "
            f"R{race_no}"
        )

        return metadata

    full_text = clean_text(
        header.get_text(
            " ",
            strip=True,
        )
    )

    race_match = re.search(
        r"\bRACE\s+(\d+)\s*"
        r"\(\s*(\d+)\s*\)",
        full_text,
        re.I,
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

    for row in header.find_all(
        "tr"
    ):
        cells = get_cell_texts(
            row
        )

        if not cells:
            continue

        for cell in cells:
            class_match = re.search(
                r"\b"
                r"(Class\s+\d+)"
                r"\b",
                cell,
                re.I,
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
                r"\b"
                r"(\d{3,4})"
                r"\s*M\b",
                cell,
                re.I,
            )

            if (
                distance_match
                and
                metadata[
                    "distance_m"
                ]
                is None
            ):
                metadata[
                    "distance_m"
                ] = int(
                    distance_match.group(1)
                )

            rating_match = re.search(
                r"\(\s*(\d+)"
                r"\s*-\s*"
                r"(\d+)\s*\)",
                cell,
            )

            if (
                rating_match
                and
                not metadata[
                    "rating_band"
                ]
            ):
                metadata[
                    "rating_band"
                ] = (
                    f"{rating_match.group(1)}-"
                    f"{rating_match.group(2)}"
                )

            prize = parse_prize_money(
                cell
            )

            if prize is not None:
                metadata[
                    "prize_money_hkd"
                ] = prize

        going = extract_label_value(
            cells,
            "Going",
        )

        if going:
            metadata[
                "going"
            ] = going

        course = extract_label_value(
            cells,
            "Course",
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
                re.I,
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
                    re.I,
                ):
                    continue

                if re.search(
                    r"\bRACE\s+\d+",
                    candidate,
                    re.I,
                ):
                    continue

                metadata[
                    "race_name"
                ] = candidate

                break

    if (
        metadata[
            "prize_money_hkd"
        ]
        is None
    ):
        metadata[
            "prize_money_hkd"
        ] = parse_prize_money(
            full_text
        )

    if not metadata[
        "race_class"
    ]:
        match = re.search(
            r"\b"
            r"(Class\s+\d+)"
            r"\b",
            full_text,
            re.I,
        )

        if match:
            metadata[
                "race_class"
            ] = (
                match.group(1)
                .title()
            )

    if (
        metadata[
            "distance_m"
        ]
        is None
    ):
        match = re.search(
            r"\b"
            r"(\d{3,4})"
            r"\s*M\b",
            full_text,
            re.I,
        )

        if match:
            metadata[
                "distance_m"
            ] = int(
                match.group(1)
            )

    course_upper = clean_text(
        metadata[
            "course"
        ]
    ).upper()

    if (
        "TURF"
        in
        course_upper
    ):
        metadata[
            "surface"
        ] = "TURF"

    elif (
        "ALL WEATHER"
        in
        course_upper
        or
        "AWT"
        in
        course_upper
    ):
        metadata[
            "surface"
        ] = (
            "ALL WEATHER TRACK"
        )

    print(
        "PARSED RACE:",
        {
            "race_date":
                metadata[
                    "race_date"
                ],
            "race_number":
                metadata[
                    "race_number"
                ],
            "race_index":
                metadata[
                    "race_index"
                ],
            "prize_money_hkd":
                metadata[
                    "prize_money_hkd"
                ],
            "header_race_class":
                metadata[
                    "race_class"
                ],
        },
    )

    return metadata


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
    value,
):
    text = (
        clean_text(
            value
        )
        .lower()
        .replace(
            "&",
            " and ",
        )
    )

    text = re.sub(
        r"[.()/\\_-]+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def identify_result_header(
    value,
):
    normalised = (
        normalise_result_header(
            value
        )
    )

    for (
        canonical_name,
        aliases,
    ) in (
        RESULT_HEADER_ALIASES.items()
    ):
        if normalised in aliases:
            return canonical_name

    return None


def build_result_column_map(
    table,
):
    best_map = {}
    best_score = 0

    for row in table.find_all(
        "tr"
    ):
        cells = row.find_all(
            [
                "th",
                "td",
            ]
        )

        current_map = {}

        for index, cell in enumerate(
            cells
        ):
            canonical = (
                identify_result_header(
                    cell.get_text(
                        " ",
                        strip=True,
                    )
                )
            )

            if (
                canonical
                and
                canonical
                not in current_map
            ):
                current_map[
                    canonical
                ] = index

        if (
            len(current_map)
            >
            best_score
        ):
            best_score = len(
                current_map
            )

            best_map = (
                current_map
            )

    return (
        best_map
        if best_score >= 5
        else {}
    )


def get_result_cell(
    cells,
    column_map,
    field_name,
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
    field_name,
):
    cell = get_result_cell(
        cells,
        column_map,
        field_name,
    )

    return (
        clean_text(
            cell.get_text(
                " ",
                strip=True,
            )
        )
        if cell is not None
        else ""
    )


def extract_results(
    html,
    race_metadata,
):
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    table = (
        soup.select_one(
            "table.performance"
        )
        or
        soup.select_one(
            ".performance table"
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # NO RESULTS TABLE IS A NORMAL SKIP.
    # --------------------------------------------------------

    if table is None:
        return None

    column_map = (
        build_result_column_map(
            table
        )
    )

    if not column_map:
        return None

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

    if any(
        field not in column_map
        for field in core_fields
    ):
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

        horse_cell = (
            get_result_cell(
                cells,
                column_map,
                "horse",
            )
        )

        horse_number = (
            parse_integer(
                get_result_cell_text(
                    cells,
                    column_map,
                    "horse_number",
                )
            )
        )

        horse_link = None

        if horse_cell is not None:
            horse_link = horse_cell.find(
                "a",
                href=re.compile(
                    r"horse",
                    re.I,
                ),
            )

        if horse_link is None:
            horse_link = row.find(
                "a",
                href=re.compile(
                    r"horse",
                    re.I,
                ),
            )

        horse_id = (
            extract_horse_id(
                horse_link.get(
                    "href",
                    "",
                )
            )
            if horse_link
            else ""
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
                horse_number,
            )
        )

    if not candidate_rows:
        return None

    field_size = len(
        candidate_rows
    )

    results = []

    for (
        row,
        cells,
        horse_link,
        horse_id,
        horse_number,
    ) in candidate_rows:

        horse_url = ""

        if horse_link is not None:
            horse_url = (
                ensure_horse_option_1(
                    urljoin(
                        "https://racing.hkjc.com",
                        horse_link.get(
                            "href",
                            "",
                        ),
                    )
                )
            )

        horse_cell = (
            get_result_cell(
                cells,
                column_map,
                "horse",
            )
        )

        if horse_link is not None:
            horse_name = clean_text(
                horse_link.get_text(
                    " ",
                    strip=True,
                )
            )

        elif horse_cell is not None:
            horse_name = clean_text(
                horse_cell.get_text(
                    " ",
                    strip=True,
                )
            )

        else:
            horse_name = ""

        finishing_position = (
            parse_integer(
                get_result_cell_text(
                    cells,
                    column_map,
                    "finishing_position",
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
            # Final class comes from horse form history.
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
                finishing_position
                ==
                1,

            "is_top_three":
                (
                    finishing_position
                    is not None
                    and
                    finishing_position
                    <=
                    3
                ),

            "jockey":
                get_result_cell_text(
                    cells,
                    column_map,
                    "jockey",
                ),

            "trainer":
                get_result_cell_text(
                    cells,
                    column_map,
                    "trainer",
                ),

            "actual_weight":
                parse_integer(
                    get_result_cell_text(
                        cells,
                        column_map,
                        "actual_weight",
                    )
                ),

            "declared_horse_weight":
                parse_integer(
                    get_result_cell_text(
                        cells,
                        column_map,
                        "declared_horse_weight",
                    )
                ),

            "draw":
                parse_integer(
                    get_result_cell_text(
                        cells,
                        column_map,
                        "draw",
                    )
                ),

            "margin":
                get_result_cell_text(
                    cells,
                    column_map,
                    "margin",
                ),

            "finish_time":
                clean_finish_time(
                    get_result_cell_text(
                        cells,
                        column_map,
                        "finish_time",
                    )
                ),

            "odds":
                parse_float(
                    get_result_cell_text(
                        cells,
                        column_map,
                        "odds",
                    )
                ),

            "horse_profile_url":
                horse_url,

            "race_url":
                race_metadata[
                    "race_url"
                ],
        })

    return (
        pd.DataFrame(
            results
        )
        if results
        else None
    )


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

PROFILE_SECTION_STOP_LABELS = [
    "Current Rating",
    "Last Rating",
]


def extract_profile_value(
    text,
    label,
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
            reverse=True,
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
        re.I,
    )

    return (
        clean_text(
            match.group(1)
        )
        if match
        else ""
    )


def extract_profile_value_any(
    text,
    labels,
):
    for label in labels:
        value = (
            extract_profile_value(
                text,
                label,
            )
        )

        if value:
            return value

    return ""


def extract_horse_profile(
    html,
    horse_id,
    profile_url,
    fallback_name="",
):
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    text = clean_text(
        soup.get_text(
            " ",
            strip=True,
        )
    )

    profile = {
        column: ""
        for column in
        HORSE_COLUMNS
    }

    profile[
        "horse_id"
    ] = horse_id

    profile[
        "horse_name"
    ] = fallback_name

    profile[
        "profile_url"
    ] = ensure_horse_option_1(
        profile_url
    )

    profile[
        "profile_scraped"
    ] = False

    heading_match = re.search(
        r"\b"
        r"([A-Z][A-Z0-9 '&.\-]+?)"
        r"\s+"
        r"\(([A-Z]{1,3}\d{3})\)",
        text,
        re.I,
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

    origin_age = (
        extract_profile_value_any(
            text,
            [
                "Country of Origin / Age",
                "Country of Origin",
            ],
        )
    )

    if origin_age:
        parts = [
            clean_text(item)
            for item in
            origin_age.split(
                "/",
                1,
            )
        ]

        if parts:
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
            "",
        )
    )

    colour_sex = (
        extract_profile_value(
            text,
            "Colour / Sex",
        )
    )

    if colour_sex:
        parts = [
            clean_text(item)
            for item in
            colour_sex.rsplit(
                "/",
                1,
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
        "Sire",
    )

    profile[
        "dam"
    ] = extract_profile_value(
        text,
        "Dam",
    )

    profile[
        "dam_sire"
    ] = extract_profile_value(
        text,
        "Dam's Sire",
    )

    useful = [
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
        clean_text(value)
        for value in useful
        if value is not None
    )

    profile[
        "profile_scraped"
    ] = success

    if success:
        profile[
            "profile_scraped_at"
        ] = utc_now_string()

    return profile


def load_horse_master():
    horse_master = {}
    age_refresh_pending = set()

    if not os.path.exists(
        HORSE_MASTER_FILE
    ):
        return (
            horse_master,
            age_refresh_pending,
        )

    try:
        df = pd.read_csv(
            HORSE_MASTER_FILE,
            dtype=object,
        ).fillna("")

    except Exception as exc:
        print(
            f"Could not read "
            f"horse master: "
            f"{exc}"
        )

        return (
            horse_master,
            age_refresh_pending,
        )

    had_age = (
        "horse_age"
        in
        df.columns
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
                "",
            )
        )

        if not horse_id:
            continue

        record = {
            column:
                row.get(
                    column,
                    "",
                )
            for column in
            HORSE_COLUMNS
        }

        record[
            "hemisphere_of_origin"
        ] = get_hemisphere_of_origin(
            record.get(
                "country_of_origin",
                "",
            )
        )

        horse_master[
            horse_id
        ] = record

        if not had_age:
            age_refresh_pending.add(
                horse_id
            )

    return (
        horse_master,
        age_refresh_pending,
    )


def save_horse_master(
    horse_master,
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

    df = (
        df[
            HORSE_COLUMNS
        ]
        .drop_duplicates(
            subset=[
                "horse_id"
            ],
            keep="last",
        )
        .sort_values(
            "horse_id"
        )
    )

    df.to_csv(
        HORSE_MASTER_FILE,
        index=False,
    )


def horse_profile_is_scraped(
    horse,
):
    if not horse:
        return False

    return (
        clean_text(
            horse.get(
                "profile_scraped",
                "",
            )
        ).lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def merge_horse_profile(
    existing,
    new_profile,
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
        value = new_profile.get(
            column,
            "",
        )

        if column in {
            "profile_scraped",
            "profile_scraped_at",
            "profile_url",
        }:
            merged[
                column
            ] = value

        elif clean_text(
            value
        ):
            merged[
                column
            ] = value

    return merged


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
    value,
):
    text = clean_text(
        value
    ).lower()

    text = re.sub(
        r"[.()/\\_-]+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def identify_horse_form_header(
    value,
):
    normalised = (
        normalise_horse_form_header(
            value
        )
    )

    for (
        canonical,
        aliases,
    ) in (
        HORSE_FORM_HEADER_ALIASES.items()
    ):
        if normalised in aliases:
            return canonical

    return None


def build_horse_form_column_map(
    table,
):
    best_map = {}

    for row in table.find_all(
        "tr"
    ):
        current = {}

        for index, cell in enumerate(
            row.find_all(
                [
                    "th",
                    "td",
                ]
            )
        ):
            canonical = (
                identify_horse_form_header(
                    cell.get_text(
                        " ",
                        strip=True,
                    )
                )
            )

            if canonical:
                current[
                    canonical
                ] = index

        if (
            len(current)
            >
            len(best_map)
        ):
            best_map = current

    required = {
        "race_index",
        "date",
        "race_class",
        "rating",
    }

    return (
        best_map
        if required.issubset(
            best_map
        )
        else {}
    )


def find_horse_form_table(
    soup,
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
                column_map,
            )

    return (
        None,
        {},
    )


def get_form_cell_text(
    cells,
    column_map,
    field_name,
):
    index = column_map.get(
        field_name
    )

    if (
        index is None
        or
        index >= len(cells)
    ):
        return ""

    return clean_text(
        cells[
            index
        ].get_text(
            " ",
            strip=True,
        )
    )


def parse_hkjc_form_date(
    value,
):
    text = clean_text(
        value
    )

    for fmt in [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
    ]:
        try:
            return datetime.strptime(
                text,
                fmt,
            ).date()

        except ValueError:
            pass

    parsed = pd.to_datetime(
        text,
        errors="coerce",
        dayfirst=True,
    )

    return (
        None
        if pd.isna(parsed)
        else parsed.date()
    )


def extract_header_rating(
    text,
):
    for label in [
        "Last Rating",
        "Current Rating",
    ]:
        match = re.search(
            rf"\b"
            rf"{re.escape(label)}"
            rf"\s*:\s*"
            rf"(-?\d+)",
            text,
            re.I,
        )

        if match:
            return int(
                match.group(1)
            )

    return None


def extract_horse_rating_history(
    html,
    horse_id,
    source_url,
):
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    full_text = clean_text(
        soup.get_text(
            " ",
            strip=True,
        )
    )

    latest_rating = (
        extract_header_rating(
            full_text
        )
    )

    (
        table,
        column_map,
    ) = find_horse_form_table(
        soup
    )

    if table is None:
        return pd.DataFrame(
            columns=
                HORSE_RATING_CACHE_COLUMNS
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

        race_index = (
            parse_integer(
                get_form_cell_text(
                    cells,
                    column_map,
                    "race_index",
                )
            )
        )

        race_date = (
            parse_hkjc_form_date(
                get_form_cell_text(
                    cells,
                    column_map,
                    "date",
                )
            )
        )

        race_class = (
            get_form_cell_text(
                cells,
                column_map,
                "race_class",
            )
        )

        rating_before = (
            parse_integer(
                get_form_cell_text(
                    cells,
                    column_map,
                    "rating",
                )
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

    unique = {
        (
            record[
                "race_date"
            ],
            record[
                "race_index"
            ],
        ):
            record
        for record in
        records
    }

    records = list(
        unique.values()
    )

    records.sort(
        key=lambda x:
            (
                x[
                    "race_date"
                ],
                x[
                    "race_index"
                ],
            )
    )

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
            ] = latest_rating

    return pd.DataFrame(
        records,
        columns=
            HORSE_RATING_CACHE_COLUMNS,
    )


def fetch_and_parse_horse(
    horse_id,
    horse_name,
):
    response = request_horse_page(
        horse_id
    )

    if response is None:
        return (
            horse_id,
            horse_name,
            None,
            None,
        )

    source_url = (
        ensure_horse_option_1(
            response.url
        )
    )

    profile = extract_horse_profile(
        response.text,
        horse_id,
        source_url,
        horse_name,
    )

    history = (
        extract_horse_rating_history(
            response.text,
            horse_id,
            source_url,
        )
    )

    return (
        horse_id,
        horse_name,
        profile,
        history,
    )


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
            dtype=object,
        ).fillna("")

    except Exception as exc:
        print(
            f"Could not read "
            f"horse rating cache: "
            f"{exc}"
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

    return df[
        HORSE_RATING_CACHE_COLUMNS
    ]


def save_horse_rating_cache(
    cache_df,
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

    if not df.empty:
        df = df.drop_duplicates(
            subset=[
                "horse_id",
                "race_date",
                "race_index",
            ],
            keep="last",
        )

        df[
            "_date"
        ] = pd.to_datetime(
            df[
                "race_date"
            ],
            errors="coerce",
        )

        df[
            "_index"
        ] = pd.to_numeric(
            df[
                "race_index"
            ],
            errors="coerce",
        )

        df = (
            df.sort_values(
                [
                    "horse_id",
                    "_date",
                    "_index",
                ]
            )
            .drop(
                columns=[
                    "_date",
                    "_index",
                ]
            )
        )

    df[
        HORSE_RATING_CACHE_COLUMNS
    ].to_csv(
        HORSE_RATINGS_CACHE_FILE,
        index=False,
    )


def normalise_rating_key(
    horse_id,
    race_date,
    race_index,
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
        parsed_index,
    )


def build_rating_cache_lookup(
    cache_df,
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
                "horse_id"
            ),
            row.get(
                "race_date"
            ),
            row.get(
                "race_index"
            ),
        )

        if key is None:
            continue

        lookup[
            key
        ] = {
            "race_class":
                row.get(
                    "race_class",
                    "",
                ),

            "horse_rating_before":
                row.get(
                    "horse_rating_before",
                    "",
                ),

            "horse_rating_after":
                row.get(
                    "horse_rating_after",
                    "",
                ),
        }

    return lookup


def replace_horse_in_rating_cache(
    cache_df,
    horse_id,
    new_history,
):
    if cache_df is None:
        cache_df = pd.DataFrame(
            columns=
                HORSE_RATING_CACHE_COLUMNS
        )

    if cache_df.empty:
        remaining = cache_df

    else:
        remaining = cache_df[
            cache_df[
                "horse_id"
            ].astype(str)
            !=
            str(horse_id)
        ]

    if (
        new_history is None
        or
        new_history.empty
    ):
        return remaining.copy()

    return pd.concat(
        [
            remaining,
            new_history,
        ],
        ignore_index=True,
    )


def horse_form_rows_complete_for_results(
    horse_id,
    horse_results,
    cache_lookup,
):
    valid = False

    for _, row in horse_results.iterrows():
        key = normalise_rating_key(
            horse_id,
            row.get(
                "race_date"
            ),
            row.get(
                "race_index"
            ),
        )

        if key is None:
            continue

        valid = True

        cached = cache_lookup.get(
            key
        )

        if (
            cached is None
            or
            not clean_text(
                cached.get(
                    "race_class",
                    "",
                )
            )
        ):
            return False

    return valid


def apply_rating_cache_to_results(
    results_df,
    cache_df,
):
    df = results_df.copy()

    lookup = (
        build_rating_cache_lookup(
            cache_df
        )
    )

    for column in [
        "race_class",
        "horse_rating_before",
        "horse_rating_after",
    ]:
        if column not in df.columns:
            df[
                column
            ] = None

        df[
            column
        ] = df[
            column
        ].astype(
            object
        )

    matched = 0

    for index, row in df.iterrows():
        key = normalise_rating_key(
            row.get(
                "horse_id"
            ),
            row.get(
                "race_date"
            ),
            row.get(
                "race_index"
            ),
        )

        record = lookup.get(
            key
        )

        if record is None:
            continue

        df.at[
            index,
            "race_class"
        ] = record.get(
            "race_class",
            "",
        )

        df.at[
            index,
            "horse_rating_before"
        ] = record.get(
            "horse_rating_before",
            "",
        )

        df.at[
            index,
            "horse_rating_after"
        ] = record.get(
            "horse_rating_after",
            "",
        )

        matched += 1

    print(
        f"Horse form rows matched: "
        f"{matched}"
    )

    return df


def ensure_horse_data(
    results_df,
    horse_master,
    age_refresh_pending,
    rating_cache_df,
):
    if results_df.empty:
        return (
            rating_cache_df,
            0,
        )

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
        .copy()
    )

    unique_horses[
        "horse_id"
    ] = (
        unique_horses[
            "horse_id"
        ]
        .astype(str)
        .map(
            clean_text
        )
    )

    unique_horses = (
        unique_horses[
            unique_horses[
                "horse_id"
            ].ne("")
        ]
    )

    cache_lookup = (
        build_rating_cache_lookup(
            rating_cache_df
        )
    )

    horses_to_fetch = []

    for _, row in unique_horses.iterrows():
        horse_id = row[
            "horse_id"
        ]

        horse_name = clean_text(
            row.get(
                "horse_name",
                "",
            )
        )

        existing = (
            horse_master.get(
                horse_id
            )
        )

        need_profile = (
            not horse_profile_is_scraped(
                existing
            )
            or
            horse_id
            in
            age_refresh_pending
        )

        horse_results = (
            results_df[
                results_df[
                    "horse_id"
                ].astype(str)
                ==
                horse_id
            ]
        )

        need_form = (
            not
            horse_form_rows_complete_for_results(
                horse_id,
                horse_results,
                cache_lookup,
            )
        )

        if (
            need_profile
            or
            need_form
        ):
            horses_to_fetch.append(
                (
                    horse_id,
                    horse_name,
                )
            )

    print()
    print(
        "=" * 70
    )

    print(
        "PHASE 3 - HORSES"
    )

    print(
        "=" * 70
    )

    print(
        "Unique horses:",
        len(
            unique_horses
        ),
    )

    print(
        "Horse pages required:",
        len(
            horses_to_fetch
        ),
    )

    if not horses_to_fetch:
        return (
            rating_cache_df,
            0,
        )

    workers = choose_worker_count(
        HORSE_WORKERS,
        len(
            horses_to_fetch
        ),
    )

    print(
        "Concurrent horse workers:",
        workers,
    )

    failures = 0

    with ThreadPoolExecutor(
        max_workers=
            workers
    ) as executor:

        futures = {
            executor.submit(
                fetch_and_parse_horse,
                horse_id,
                horse_name,
            ):
                (
                    horse_id,
                    horse_name,
                )

            for (
                horse_id,
                horse_name,
            ) in horses_to_fetch
        }

        for future in as_completed(
            futures
        ):
            (
                horse_id,
                horse_name,
            ) = futures[
                future
            ]

            try:
                (
                    returned_id,
                    _,
                    profile,
                    history,
                ) = future.result()

            except Exception as exc:
                failures += 1

                print(
                    f"HORSE WORKER FAILED "
                    f"{horse_id} "
                    f"{horse_name}: "
                    f"{exc}"
                )

                continue

            if returned_id != horse_id:
                failures += 1
                continue

            if (
                profile is None
                and
                history is None
            ):
                failures += 1
                continue

            if profile is not None:
                horse_master[
                    horse_id
                ] = merge_horse_profile(
                    horse_master.get(
                        horse_id
                    ),
                    profile,
                )

                if horse_profile_is_scraped(
                    horse_master.get(
                        horse_id
                    )
                ):
                    age_refresh_pending.discard(
                        horse_id
                    )

            if (
                history is not None
                and
                not history.empty
            ):
                rating_cache_df = (
                    replace_horse_in_rating_cache(
                        rating_cache_df,
                        horse_id,
                        history,
                    )
                )

    return (
        rating_cache_df,
        failures,
    )


def enrich_results_with_horse_master(
    results_df,
    horse_master,
):
    df = results_df.copy()

    mapping = {
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

    for target in mapping.values():
        if target not in df.columns:
            df[
                target
            ] = None

        df[
            target
        ] = df[
            target
        ].astype(
            object
        )

    if (
        "horse_age_at_race"
        not in
        df.columns
    ):
        df[
            "horse_age_at_race"
        ] = None

    df[
        "horse_age_at_race"
    ] = df[
        "horse_age_at_race"
    ].astype(
        object
    )

    for index, row in df.iterrows():
        horse_id = clean_text(
            row.get(
                "horse_id"
            )
        )

        horse = horse_master.get(
            horse_id
        )

        if not horse:
            continue

        for (
            source,
            target,
        ) in mapping.items():
            df.at[
                index,
                target
            ] = horse.get(
                source,
                "",
            )

        df.at[
            index,
            "horse_age_at_race"
        ] = (
            calculate_horse_age_at_race(
                horse.get(
                    "horse_age"
                ),
                horse.get(
                    "country_of_origin"
                ),
                horse.get(
                    "profile_scraped_at"
                ),
                row.get(
                    "race_date"
                ),
            )
        )

    return df


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
            dtype=object,
        )

    except Exception:
        return set()

    return set(
        df[
            "result_id"
        ]
        .dropna()
        .astype(str)
    )


def append_results(
    results_df,
    existing_ids,
):
    df = (
        results_df[
            ~results_df[
                "result_id"
            ]
            .astype(str)
            .isin(
                existing_ids
            )
        ]
        .copy()
    )

    if df.empty:
        print(
            "No new result rows."
        )

        return

    for column in RACE_COLUMNS:
        if column not in df.columns:
            df[
                column
            ] = None

    df = df[
        RACE_COLUMNS
    ]

    file_exists = os.path.exists(
        RACE_RESULTS_FILE
    )

    df.to_csv(
        RACE_RESULTS_FILE,
        mode="a",
        header=
            not file_exists,
        index=False,
    )

    existing_ids.update(
        df[
            "result_id"
        ].astype(str)
    )

    print(
        "Added result rows:",
        len(df),
    )


def backfill_existing_results(
    horse_master,
):
    if not os.path.exists(
        RACE_RESULTS_FILE
    ):
        return

    try:
        df = pd.read_csv(
            RACE_RESULTS_FILE,
            dtype=object,
        ).fillna("")

    except Exception as exc:
        print(
            f"Could not backfill "
            f"existing results: "
            f"{exc}"
        )

        return

    if df.empty:
        return

    df = (
        enrich_results_with_horse_master(
            df,
            horse_master,
        )
    )

    for column in RACE_COLUMNS:
        if column not in df.columns:
            df[
                column
            ] = None

    df[
        RACE_COLUMNS
    ].to_csv(
        RACE_RESULTS_FILE,
        index=False,
    )


def apply_rating_cache_to_existing_results_file(
    cache_df,
):
    if not os.path.exists(
        RACE_RESULTS_FILE
    ):
        return

    try:
        df = pd.read_csv(
            RACE_RESULTS_FILE,
            dtype=object,
        ).fillna("")

    except Exception as exc:
        print(
            f"Could not apply "
            f"rating cache to "
            f"existing results: "
            f"{exc}"
        )

        return

    if df.empty:
        return

    df = (
        apply_rating_cache_to_results(
            df,
            cache_df,
        )
    )

    for column in RACE_COLUMNS:
        if column not in df.columns:
            df[
                column
            ] = None

    df[
        RACE_COLUMNS
    ].to_csv(
        RACE_RESULTS_FILE,
        index=False,
    )


def get_prize_payout_schedule(
    race_date,
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
    df,
):
    output = pd.Series(
        0.0,
        index=df.index,
        dtype="float64",
    )

    for (
        _,
        race_group,
    ) in df.groupby(
        "race_id",
        sort=False,
        dropna=False,
    ):
        if race_group.empty:
            continue

        race_date = race_group[
            "_race_date_sort"
        ].iloc[0]

        schedule = (
            get_prize_payout_schedule(
                race_date
            )
        )

        valid = race_group[
            race_group[
                "_finish_numeric"
            ].notna()
        ]

        for (
            finishing_position,
            position_group,
        ) in valid.groupby(
            "_finish_numeric",
            sort=True,
        ):
            position = int(
                finishing_position
            )

            count = len(
                position_group
            )

            if count == 1:
                percentage = (
                    schedule.get(
                        position,
                        0.0,
                    )
                )

            else:
                total = sum(
                    schedule.get(
                        position
                        +
                        offset,
                        0.0,
                    )
                    for offset in
                    range(count)
                )

                percentage = (
                    total
                    /
                    count
                )

            output.loc[
                position_group.index
            ] = percentage

    return output


def calculate_historical_career_stats():
    if not os.path.exists(
        RACE_RESULTS_FILE
    ):
        return

    try:
        df = pd.read_csv(
            RACE_RESULTS_FILE,
            dtype=object,
        ).fillna("")

    except Exception as exc:
        print(
            f"Could not calculate "
            f"career stats: "
            f"{exc}"
        )

        return

    if df.empty:
        return

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
        errors="coerce",
    )

    df[
        "_race_number_sort"
    ] = pd.to_numeric(
        df[
            "race_number"
        ],
        errors="coerce",
    ).fillna(0)

    df[
        "_finish_numeric"
    ] = pd.to_numeric(
        df[
            "finishing_position"
        ],
        errors="coerce",
    )

    df[
        "_race_prize_numeric"
    ] = pd.to_numeric(
        df[
            "prize_money_hkd"
        ],
        errors="coerce",
    ).fillna(0)

    df[
        "_is_start"
    ] = (
        df[
            "_finish_numeric"
        ].notna()
    ).astype(int)

    df[
        "_is_win"
    ] = (
        df[
            "_finish_numeric"
        ]
        ==
        1
    ).astype(int)

    df[
        "_is_second"
    ] = (
        df[
            "_finish_numeric"
        ]
        ==
        2
    ).astype(int)

    df[
        "_is_third"
    ] = (
        df[
            "_finish_numeric"
        ]
        ==
        3
    ).astype(int)

    df[
        "_is_top3"
    ] = (
        df[
            "_finish_numeric"
        ].isin(
            [
                1,
                2,
                3,
            ]
        )
    ).astype(int)

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
            errors="coerce",
        ).fillna(0)
    ).round(2)

    sort_columns = [
        "horse_id",
        "_race_date_sort",
        "_race_number_sort",
    ]

    if (
        "race_index"
        in
        df.columns
    ):
        df[
            "_race_index_sort"
        ] = pd.to_numeric(
            df[
                "race_index"
            ],
            errors="coerce",
        ).fillna(0)

        sort_columns.append(
            "_race_index_sort"
        )

    df = (
        df.sort_values(
            sort_columns,
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    grouped = df.groupby(
        "horse_id",
        sort=False,
        dropna=False,
    )

    for (
        output_column,
        source_column,
    ) in [
        (
            "career_starts_before",
            "_is_start",
        ),
        (
            "career_wins_before",
            "_is_win",
        ),
        (
            "career_seconds_before",
            "_is_second",
        ),
        (
            "career_thirds_before",
            "_is_third",
        ),
        (
            "career_top3_before",
            "_is_top3",
        ),
    ]:
        df[
            output_column
        ] = (
            grouped[
                source_column
            ].cumsum()
            -
            df[
                source_column
            ]
        )

    df[
        "career_prize_money_after"
    ] = (
        grouped[
            "prize_money_won_this_race"
        ]
        .cumsum()
        .round(2)
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
    ).round(2)

    starts = pd.to_numeric(
        df[
            "career_starts_before"
        ],
        errors="coerce",
    ).fillna(0)

    wins = pd.to_numeric(
        df[
            "career_wins_before"
        ],
        errors="coerce",
    ).fillna(0)

    top3 = pd.to_numeric(
        df[
            "career_top3_before"
        ],
        errors="coerce",
    ).fillna(0)

    df[
        "career_win_rate_before"
    ] = 0.0

    df[
        "career_top3_rate_before"
    ] = 0.0

    valid = (
        starts
        >
        0
    )

    df.loc[
        valid,
        "career_win_rate_before"
    ] = (
        wins[
            valid
        ]
        /
        starts[
            valid
        ]
    ).round(4)

    df.loc[
        valid,
        "career_top3_rate_before"
    ] = (
        top3[
            valid
        ]
        /
        starts[
            valid
        ]
    ).round(4)

    df = df.sort_values(
        "_original_order",
        kind="stable",
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
        errors="ignore",
    )

    for column in RACE_COLUMNS:
        if column not in df.columns:
            df[
                column
            ] = None

    df[
        RACE_COLUMNS
    ].to_csv(
        RACE_RESULTS_FILE,
        index=False,
    )


def discover_date(
    meeting_date,
):
    response = request_race_page(
        meeting_date
    )

    if response is None:
        return {
            "date":
                meeting_date,

            "request_failed":
                True,

            "meeting":
                None,

            "race_numbers":
                [],
        }

    meeting = detect_meeting(
        response.text
    )

    if meeting is None:
        return {
            "date":
                meeting_date,

            "request_failed":
                False,

            "meeting":
                None,

            "race_numbers":
                [],
        }

    return {
        "date":
            meeting_date,

        "request_failed":
            False,

        "meeting":
            meeting,

        "race_numbers":
            detect_race_numbers(
                response.text
            ),
    }


def discover_month(
    start_date,
    end_date,
):
    dates = list(
        date_range(
            start_date,
            end_date,
        )
    )

    workers = choose_worker_count(
        DATE_WORKERS,
        len(
            dates
        ),
    )

    print()
    print(
        "=" * 70
    )

    print(
        "PHASE 1 - DATE DISCOVERY"
    )

    print(
        "=" * 70
    )

    print(
        "Dates queued:",
        len(
            dates
        ),
    )

    print(
        "Concurrent date workers:",
        workers,
    )

    discoveries = []
    failures = 0

    with ThreadPoolExecutor(
        max_workers=
            workers
    ) as executor:

        futures = {
            executor.submit(
                discover_date,
                date,
            ):
                date
            for date in
            dates
        }

        for future in as_completed(
            futures
        ):
            meeting_date = futures[
                future
            ]

            try:
                result = future.result()

            except Exception as exc:
                failures += 1

                print(
                    f"DATE FAILURE "
                    f"{meeting_date}: "
                    f"{exc}"
                )

                continue

            if result[
                "request_failed"
            ]:
                failures += 1
                continue

            discoveries.append(
                result
            )

            if (
                result[
                    "meeting"
                ]
                is not None
            ):
                print(
                    f"MEETING "
                    f"{meeting_date}: "
                    f"{result['meeting']['racecourse_name']} "
                    f"races="
                    f"{result['race_numbers']}"
                )

    discoveries.sort(
        key=lambda x:
            x[
                "date"
            ]
    )

    return (
        discoveries,
        failures,
    )


def build_race_tasks(
    discoveries,
):
    tasks = []

    for item in discoveries:
        meeting = item[
            "meeting"
        ]

        if meeting is None:
            continue

        for race_no in item[
            "race_numbers"
        ]:
            tasks.append({
                "date":
                    item[
                        "date"
                    ],

                "racecourse_code":
                    meeting[
                        "racecourse_code"
                    ],

                "racecourse_name":
                    meeting[
                        "racecourse_name"
                    ],

                "race_no":
                    race_no,
            })

    return tasks


def fetch_and_parse_race(
    task,
):
    meeting_date = task[
        "date"
    ]

    racecourse_code = task[
        "racecourse_code"
    ]

    racecourse_name = task[
        "racecourse_name"
    ]

    race_no = task[
        "race_no"
    ]

    race_url = build_url(
        meeting_date,
        racecourse_code,
        race_no,
    )

    response = request_race_page(
        meeting_date,
        racecourse_code,
        race_no,
    )

    if response is None:
        return (
            task,
            None,
            "REQUEST_FAILED",
        )

    try:
        metadata = (
            extract_race_metadata(
                response.text,
                meeting_date,
                racecourse_code,
                racecourse_name,
                race_no,
                race_url,
            )
        )

        # ====================================================
        # $0 PRIZE MONEY = NORMAL SKIP
        # ====================================================

        if (
            metadata.get(
                "prize_money_hkd"
            )
            ==
            0
        ):
            print(
                f"SKIP ZERO-PRIZE RACE: "
                f"{meeting_date} "
                f"{racecourse_code} "
                f"R{race_no}"
            )

            return (
                task,
                None,
                "SKIP_ZERO_PRIZE",
            )

        results_df = (
            extract_results(
                response.text,
                metadata,
            )
        )

        # ====================================================
        # NO RESULTS TABLE / EMPTY RESULTS = NORMAL SKIP
        # ====================================================

        if (
            results_df is None
            or
            results_df.empty
        ):
            print(
                f"SKIP NO-RESULTS RACE: "
                f"{meeting_date} "
                f"{racecourse_code} "
                f"R{race_no}"
            )

            return (
                task,
                None,
                "SKIP_NO_RESULTS",
            )

    except Exception as exc:
        return (
            task,
            None,
            f"PARSE_FAILED: {exc}",
        )

    return (
        task,
        results_df,
        None,
    )


def fetch_month_races(
    discoveries,
):
    tasks = build_race_tasks(
        discoveries
    )

    if not tasks:
        return (
            pd.DataFrame(),
            0,
        )

    workers = choose_worker_count(
        RACE_WORKERS,
        len(
            tasks
        ),
    )

    print()
    print(
        "=" * 70
    )

    print(
        "PHASE 2 - ALL RACES"
    )

    print(
        "=" * 70
    )

    print(
        "Race pages queued:",
        len(
            tasks
        ),
    )

    print(
        "Concurrent race workers:",
        workers,
    )

    race_frames = []

    failures = 0
    skipped_zero_prize = 0
    skipped_no_results = 0
    successful_races = 0

    with ThreadPoolExecutor(
        max_workers=
            workers
    ) as executor:

        futures = {
            executor.submit(
                fetch_and_parse_race,
                task,
            ):
                task
            for task in
            tasks
        }

        for future in as_completed(
            futures
        ):
            task = futures[
                future
            ]

            try:
                (
                    returned_task,
                    results_df,
                    status,
                ) = future.result()

            except Exception as exc:
                failures += 1

                print(
                    f"RACE WORKER FAILED "
                    f"{task['date']} "
                    f"R{task['race_no']}: "
                    f"{exc}"
                )

                continue

            # =================================================
            # NORMAL SKIPS
            # =================================================

            if (
                status
                ==
                "SKIP_ZERO_PRIZE"
            ):
                skipped_zero_prize += 1
                continue

            if (
                status
                ==
                "SKIP_NO_RESULTS"
            ):
                skipped_no_results += 1
                continue

            # =================================================
            # REAL FAILURE
            # =================================================

            if status is not None:
                failures += 1

                print(
                    f"RACE FAILED: "
                    f"{returned_task['date']} "
                    f"{returned_task['racecourse_code']} "
                    f"R{returned_task['race_no']} "
                    f"{status}"
                )

                continue

            race_frames.append(
                results_df
            )

            successful_races += 1

    print()
    print(
        "Race phase summary:"
    )

    print(
        "  Successful races:",
        successful_races,
    )

    print(
        "  $0 races ignored:",
        skipped_zero_prize,
    )

    print(
        "  No-results races ignored:",
        skipped_no_results,
    )

    print(
        "  Genuine failures:",
        failures,
    )

    if not race_frames:
        return (
            pd.DataFrame(),
            failures,
        )

    month_results = pd.concat(
        race_frames,
        ignore_index=True,
    )

    month_results = (
        month_results
        .drop_duplicates(
            subset=[
                "result_id"
            ],
            keep="last",
        )
    )

    month_results[
        "_date_sort"
    ] = pd.to_datetime(
        month_results[
            "race_date"
        ],
        errors="coerce",
    )

    month_results[
        "_race_sort"
    ] = pd.to_numeric(
        month_results[
            "race_number"
        ],
        errors="coerce",
    )

    month_results[
        "_horse_sort"
    ] = pd.to_numeric(
        month_results[
            "horse_number"
        ],
        errors="coerce",
    )

    month_results = (
        month_results
        .sort_values(
            [
                "_date_sort",
                "_race_sort",
                "_horse_sort",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_date_sort",
                "_race_sort",
                "_horse_sort",
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )

    return (
        month_results,
        failures,
    )


def main():
    ensure_folders()

    try:
        start_date = (
            datetime.strptime(
                START_DATE,
                "%Y-%m-%d",
            ).date()
        )

        end_date = (
            datetime.strptime(
                END_DATE,
                "%Y-%m-%d",
            ).date()
        )

    except ValueError:
        raise SystemExit(
            "Dates must use YYYY-MM-DD."
        )

    if (
        start_date
        >
        end_date
    ):
        raise SystemExit(
            "START_DATE cannot "
            "be after END_DATE."
        )

    (
        horse_master,
        age_refresh_pending,
    ) = load_horse_master()

    rating_cache_df = (
        load_horse_rating_cache()
    )

    existing_result_ids = (
        load_existing_result_ids()
    )

    print()
    print(
        "=" * 70
    )

    print(
        "HKJC WHOLE-MONTH "
        "CONCURRENT COLLECTOR"
    )

    print(
        "=" * 70
    )

    print(
        "Start:",
        start_date,
    )

    print(
        "End:",
        end_date,
    )

    print(
        "Existing results:",
        len(
            existing_result_ids
        ),
    )

    print(
        "DATE_WORKERS:",
        (
            "ALL"
            if DATE_WORKERS == 0
            else DATE_WORKERS
        ),
    )

    print(
        "RACE_WORKERS:",
        (
            "ALL"
            if RACE_WORKERS == 0
            else RACE_WORKERS
        ),
    )

    print(
        "HORSE_WORKERS:",
        (
            "ALL"
            if HORSE_WORKERS == 0
            else HORSE_WORKERS
        ),
    )

    # ========================================================
    # PHASE 1
    # DISCOVER ALL DATES CONCURRENTLY
    # ========================================================

    (
        discoveries,
        date_failures,
    ) = discover_month(
        start_date,
        end_date,
    )

    # ========================================================
    # PHASE 2
    # FETCH ALL RACES CONCURRENTLY
    #
    # $0 RACES AND NO-RESULTS RACES ARE NORMAL SKIPS.
    # ========================================================

    (
        month_results,
        race_failures,
    ) = fetch_month_races(
        discoveries
    )

    # ========================================================
    # NOTHING USABLE THIS MONTH
    # ========================================================

    if month_results.empty:
        total_failures = (
            date_failures
            +
            race_failures
        )

        print()
        print(
            "No usable race-result "
            "rows for this month."
        )

        print(
            "$0 prize-money races and "
            "race numbers with no results "
            "tables are normal skips."
        )

        if total_failures:
            print(
                "Unresolved genuine failures:",
                total_failures,
            )

            raise SystemExit(
                1
            )

        print(
            "No genuine errors detected."
        )

        return

    # ========================================================
    # PHASE 3
    # ALL UNIQUE HORSES FOR THE MONTH
    # ========================================================

    (
        rating_cache_df,
        horse_failures,
    ) = ensure_horse_data(
        month_results,
        horse_master,
        age_refresh_pending,
        rating_cache_df,
    )

    # ========================================================
    # LOCAL ENRICHMENT
    # ========================================================

    month_results = (
        enrich_results_with_horse_master(
            month_results,
            horse_master,
        )
    )

    month_results = (
        apply_rating_cache_to_results(
            month_results,
            rating_cache_df,
        )
    )

    append_results(
        month_results,
        existing_result_ids,
    )

    # ========================================================
    # FINAL SAVE
    # ========================================================

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

    save_horse_master(
        horse_master
    )

    save_horse_rating_cache(
        rating_cache_df
    )

    backfill_existing_results(
        horse_master
    )

    apply_rating_cache_to_existing_results_file(
        rating_cache_df
    )

    calculate_historical_career_stats()

    # ========================================================
    # ONLY REAL FAILURES COUNT HERE
    #
    # THESE DO NOT COUNT:
    #
    # - prize_money_hkd == 0
    # - no results table
    # - empty results table
    # ========================================================

    total_failures = (
        date_failures
        +
        race_failures
        +
        horse_failures
    )

    print()
    print(
        "=" * 70
    )

    print(
        "MONTH COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Date failures:",
        date_failures,
    )

    print(
        "Race failures:",
        race_failures,
    )

    print(
        "Horse failures:",
        horse_failures,
    )

    print(
        "Total genuine failures:",
        total_failures,
    )

    print(
        "$0 prize-money races: "
        "ignored, not failures."
    )

    print(
        "Race numbers with no "
        "results table: "
        "ignored, not failures."
    )

    if total_failures:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
