import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIG
# ============================================================

RACECARD_BASE_URL = (
    "https://racing.hkjc.com/en-us/local/information/racecard"
)

RACE_DATE = os.getenv(
    "RACE_DATE",
    "2026-09-06",
)

RACE_WORKERS = max(
    1,
    int(
        os.getenv(
            "RACE_WORKERS",
            "8",
        )
    ),
)

HORSE_WORKERS = max(
    1,
    int(
        os.getenv(
            "HORSE_WORKERS",
            "20",
        )
    ),
)

REQUEST_TIMEOUT = float(
    os.getenv(
        "REQUEST_TIMEOUT",
        "60",
    )
)

HTTP_RETRIES = max(
    0,
    int(
        os.getenv(
            "HTTP_RETRIES",
            "4",
        )
    ),
)

HTTP_BACKOFF_FACTOR = float(
    os.getenv(
        "HTTP_BACKOFF_FACTOR",
        "1.5",
    )
)


# ============================================================
# OUTPUT
# ============================================================

RESULTS_DIR = os.path.join(
    "results",
    "races",
)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "upcoming_races.csv",
)


OUTPUT_COLUMNS = [
    "race_date",
    "racecourse_code",
    "racecourse_name",
    "race_number",
    "race_id",
    "race_name",
    "race_time",
    "surface",
    "course",
    "distance_m",
    "going",
    "prize_money_hkd",
    "rating_band",
    "race_class",

    "horse_number",
    "horse_name",
    "handicap_weight",
    "jockey",
    "draw",
    "trainer",
    "horse_rating",
    "declared_horse_weight",
    "days_since_last_run",

    "horse_age",
    "horse_sex",
    "season_stakes_hkd",
    "total_stakes_hkd",
    "sire",
    "dam",

    "horse_id",
    "runner_id",
    "horse_profile_url",
    "race_url",
]


# ============================================================
# HTTP
# ============================================================

COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/140.0.0.0 "
        "Safari/537.36"
    ),
    "Accept": (
        "text/html,"
        "application/xhtml+xml,"
        "application/xml;q=0.9,"
        "*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


_thread_local = threading.local()


def create_http_session():

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
        pool_connections=2,
        pool_maxsize=2,
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


def get_session():

    session = getattr(
        _thread_local,
        "session",
        None,
    )

    if session is None:

        session = create_http_session()

        _thread_local.session = (
            session
        )

    return session


def http_get(url):

    try:

        response = get_session().get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        print(
            f"GET {url} "
            f"-> {response.status_code}"
        )

        response.raise_for_status()

        return response

    except requests.RequestException as exc:

        print(
            f"REQUEST FAILED "
            f"{url}: "
            f"{exc}"
        )

        return None


# ============================================================
# HELPERS
# ============================================================

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


def parse_int(value):

    match = re.search(
        r"-?\d+",
        clean_text(
            value
        ).replace(
            ",",
            "",
        ),
    )

    if not match:
        return None

    try:

        return int(
            match.group()
        )

    except ValueError:

        return None


def parse_money(value):

    match = re.search(
        r"(?:HK\s*)?"
        r"\$\s*"
        r"([\d,]+)",
        clean_text(
            value
        ),
        re.I,
    )

    if not match:
        return None

    try:

        return int(
            match.group(
                1
            ).replace(
                ",",
                "",
            )
        )

    except ValueError:

        return None


def normalise_header(value):

    text = (
        clean_text(
            value
        )
        .lower()
        .replace(
            "’",
            "'",
        )
    )

    text = (
        text
        .replace(
            "+/-",
            " vs ",
        )
        .replace(
            "+",
            " ",
        )
        .replace(
            "-",
            " ",
        )
    )

    text = text.replace(
        "&",
        " and ",
    )

    text = re.sub(
        r"[().'\"/\\]",
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


def parse_requested_date():

    try:

        return datetime.strptime(
            RACE_DATE,
            "%Y-%m-%d",
        ).date()

    except ValueError as exc:

        raise SystemExit(
            "RACE_DATE must be YYYY-MM-DD, "
            "for example 2026-09-06"
        ) from exc


def requested_date_slash():

    return (
        parse_requested_date()
        .strftime(
            "%Y/%m/%d"
        )
    )


def build_date_url():

    return (
        f"{RACECARD_BASE_URL}"
        f"?racedate="
        f"{requested_date_slash()}"
    )


def build_race_url(
    racecourse_code,
    race_number,
):

    return (
        f"{RACECARD_BASE_URL}"
        f"?racedate="
        f"{requested_date_slash()}"
        f"&Racecourse="
        f"{racecourse_code}"
        f"&RaceNo="
        f"{race_number}"
    )


def query_value_case_insensitive(
    query,
    key,
):

    for (
        existing_key,
        values,
    ) in query.items():

        if (
            existing_key.lower()
            ==
            key.lower()
            and
            values
        ):

            return values[0]

    return ""


def extract_horse_id(url):

    if not url:
        return ""

    query = parse_qs(
        urlparse(
            url
        ).query
    )

    return clean_text(
        query_value_case_insensitive(
            query,
            "horseid",
        )
    )


# ============================================================
# DISCOVER RACES
# ============================================================

def discover_races():

    response = http_get(
        build_date_url()
    )

    if response is None:
        return []

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    requested = (
        requested_date_slash()
    )

    racecourse = ""

    discovered_numbers = set()

    # --------------------------------------------------------
    # Detect local venue from valid HKJC racecard links.
    # --------------------------------------------------------

    for link in soup.find_all(
        "a",
        href=True,
    ):

        href = urljoin(
            response.url,
            link.get(
                "href",
                "",
            ),
        )

        parsed = urlparse(
            href
        )

        if (
            "racecard"
            not in
            parsed.path.lower()
        ):

            continue

        query = parse_qs(
            parsed.query
        )

        linked_date = (
            query_value_case_insensitive(
                query,
                "racedate",
            )
        )

        linked_course = (
            query_value_case_insensitive(
                query,
                "Racecourse",
            )
            .upper()
        )

        linked_race_no = parse_int(
            query_value_case_insensitive(
                query,
                "RaceNo",
            )
        )

        if (
            linked_date
            and
            linked_date
            !=
            requested
        ):

            continue

        if linked_course in {
            "ST",
            "HV",
        }:

            racecourse = (
                linked_course
            )

        if (
            linked_race_no
            is not None
        ):

            discovered_numbers.add(
                linked_race_no
            )

    # --------------------------------------------------------
    # Fallback venue detection.
    # --------------------------------------------------------

    if not racecourse:

        text = clean_text(
            soup.get_text(
                " ",
                strip=True,
            )
        )

        if "Sha Tin" in text:

            racecourse = "ST"

        elif (
            "Happy Valley"
            in
            text
        ):

            racecourse = "HV"

        else:

            print(
                "No local Hong Kong "
                "meeting found for "
                "this date."
            )

            return []

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Always probe R1-R12.
    #
    # This prevents Race 1 being missed because HKJC's
    # navigation markup omitted it.
    #
    # Non-existent race numbers are ignored later.
    # --------------------------------------------------------

    tasks = []

    for race_no in range(
        1,
        13,
    ):

        tasks.append({
            "racecourse_code":
                racecourse,

            "race_number":
                race_no,

            "race_url":
                build_race_url(
                    racecourse,
                    race_no,
                ),
        })

    print(
        f"Detected venue: "
        f"{racecourse}; "
        f"navigation race numbers: "
        f"{sorted(discovered_numbers)}"
    )

    print(
        "Queued Race 1 through Race 12; "
        "pages without a runner table "
        "will be ignored."
    )

    return tasks


# ============================================================
# CLASS
# ============================================================

def normalise_race_class(value):

    text = clean_text(
        value
    )

    if not text:
        return ""

    # Class 1 -> C1
    # Class 2 -> C2
    # ...
    # Class 5 -> C5

    class_match = re.search(
        r"\bClass\s*([1-5])\b",
        text,
        re.I,
    )

    if class_match:

        return (
            f"C"
            f"{class_match.group(1)}"
        )

    # Group One -> G1
    # Group Two -> G2
    # Group Three -> G3

    group_match = re.search(
        r"\bGroup\s*"
        r"(One|Two|Three|1|2|3)"
        r"\b",
        text,
        re.I,
    )

    if group_match:

        token = (
            group_match.group(
                1
            ).lower()
        )

        mapping = {
            "one": "1",
            "two": "2",
            "three": "3",
            "1": "1",
            "2": "2",
            "3": "3",
        }

        return (
            f"G"
            f"{mapping[token]}"
        )

    if re.search(
        r"\bListed\b",
        text,
        re.I,
    ):

        return "L"

    if re.search(
        r"\bGriffin\b",
        text,
        re.I,
    ):

        return "GRIFFIN"

    return ""


# ============================================================
# RATING BAND
# ============================================================

def extract_rating_band_from_header(
    value,
):

    text = clean_text(
        value
    )

    if not text:
        return ""

    # Example:
    # Rating: 40-0

    match = re.search(
        r"\bRating"
        r"\s*:\s*"
        r"([0-9]+"
        r"\s*-\s*"
        r"[0-9]+)"
        r"\b",
        text,
        re.I,
    )

    if match:

        return re.sub(
            r"\s+",
            "",
            match.group(
                1
            ),
        )

    # Fallback:
    # Rating 40-0

    match = re.search(
        r"\bRating"
        r"\s+"
        r"([0-9]+"
        r"\s*-\s*"
        r"[0-9]+)"
        r"\b",
        text,
        re.I,
    )

    if match:

        return re.sub(
            r"\s+",
            "",
            match.group(
                1
            ),
        )

    # Final fallback:
    # standalone range between commas.

    for part in [
        clean_text(
            item
        )
        for item in
        text.split(
            ","
        )
    ]:

        if re.fullmatch(
            r"[0-9]+"
            r"\s*-\s*"
            r"[0-9]+",
            part,
        ):

            return re.sub(
                r"\s+",
                "",
                part,
            )

    return ""


# ============================================================
# RACE HEADER
# ============================================================

def parse_race_header(
    soup,
    racecourse_code,
    fallback_race_number,
    race_url,
):

    strings = [
        clean_text(
            item
        )
        for item in
        soup.stripped_strings
        if clean_text(
            item
        )
    ]

    race_number = (
        fallback_race_number
    )

    race_name = ""

    race_time = ""

    racecourse_name = (
        "Sha Tin"
        if racecourse_code == "ST"
        else
        "Happy Valley"
    )

    surface = ""

    course = ""

    distance_m = None

    going = ""

    prize_money_hkd = None

    rating_band = ""

    race_class = ""

    race_heading_index = None

    # --------------------------------------------------------
    # Race 1 - EXAMPLE HANDICAP
    # --------------------------------------------------------

    for (
        index,
        text,
    ) in enumerate(
        strings
    ):

        match = re.match(
            r"^Race\s+"
            r"(\d+)"
            r"\s*-\s*"
            r"(.+)$",
            text,
            re.I,
        )

        if match:

            race_number = int(
                match.group(
                    1
                )
            )

            race_name = clean_text(
                match.group(
                    2
                )
            )

            race_heading_index = (
                index
            )

            break

    if (
        race_heading_index
        is not None
    ):

        nearby = strings[
            race_heading_index:
            race_heading_index + 20
        ]

    else:

        nearby = strings

    for text in nearby:

        # ----------------------------------------------------
        # Venue / race time
        # ----------------------------------------------------

        if (
            (
                "Sha Tin"
                in
                text
                or
                "Happy Valley"
                in
                text
            )
            and
            re.search(
                r"\b"
                r"\d{1,2}:\d{2}"
                r"\b",
                text,
            )
        ):

            if "Sha Tin" in text:

                racecourse_name = (
                    "Sha Tin"
                )

            else:

                racecourse_name = (
                    "Happy Valley"
                )

            time_match = re.search(
                r"\b"
                r"(\d{1,2}:\d{2})"
                r"\b",
                text,
            )

            if time_match:

                race_time = (
                    time_match.group(
                        1
                    )
                )

        # ----------------------------------------------------
        # Surface / course / distance / going
        #
        # Example:
        # Turf, "A" Course, 1200M, Good
        # ----------------------------------------------------

        distance_match = re.search(
            r"\b"
            r"(\d{3,4})"
            r"\s*M\b",
            text,
            re.I,
        )

        if (
            distance_match
            and
            (
                "Turf"
                in
                text
                or
                "All Weather"
                in
                text
                or
                "AWT"
                in
                text
            )
        ):

            distance_m = int(
                distance_match.group(
                    1
                )
            )

            parts = [
                clean_text(
                    part
                )
                for part in
                text.split(
                    ","
                )
            ]

            if parts:

                surface = (
                    parts[0]
                )

            course_match = re.search(
                r'["“]?'
                r'([^,"”]+)'
                r'["”]?'
                r'\s+Course',
                text,
                re.I,
            )

            if course_match:

                course = clean_text(
                    course_match.group(
                        1
                    )
                ).strip(
                    '"“”'
                )

            if len(parts) >= 2:

                going = (
                    parts[-1]
                )

        # ----------------------------------------------------
        # Prize / rating band / class
        #
        # Example:
        # Prize Money: $875,000, Rating: 40-0, Class 5
        #
        # Group example:
        # Prize Money: $4,200,000, -, Group Three
        # ----------------------------------------------------

        if (
            "Prize Money"
            in
            text
        ):

            prize_money_hkd = (
                parse_money(
                    text
                )
            )

            rating_band = (
                extract_rating_band_from_header(
                    text
                )
            )

            race_class = (
                normalise_race_class(
                    text
                )
            )

    # --------------------------------------------------------
    # Fallback search if HKJC separates header pieces
    # into multiple HTML elements.
    # --------------------------------------------------------

    combined_nearby = clean_text(
        " ".join(
            nearby
        )
    )

    if not rating_band:

        rating_band = (
            extract_rating_band_from_header(
                combined_nearby
            )
        )

    if not race_class:

        race_class = (
            normalise_race_class(
                combined_nearby
            )
        )

    race_date = (
        parse_requested_date()
        .strftime(
            "%Y-%m-%d"
        )
    )

    race_id = (
        f"{racecourse_code}_"
        f"{parse_requested_date().strftime('%Y%m%d')}_"
        f"R{race_number:02d}"
    )

    print(
        "RACE HEADER:",
        {
            "race_number":
                race_number,

            "race_name":
                race_name,

            "surface":
                surface,

            "course":
                course,

            "distance_m":
                distance_m,

            "going":
                going,

            "prize_money_hkd":
                prize_money_hkd,

            "rating_band":
                rating_band,

            "race_class":
                race_class,
        },
    )

    return {
        "race_date":
            race_date,

        "racecourse_code":
            racecourse_code,

        "racecourse_name":
            racecourse_name,

        "race_number":
            race_number,

        "race_id":
            race_id,

        "race_name":
            race_name,

        "race_time":
            race_time,

        "surface":
            surface,

        "course":
            course,

        "distance_m":
            distance_m,

        "going":
            going,

        "prize_money_hkd":
            prize_money_hkd,

        "rating_band":
            rating_band,

        "race_class":
            race_class,

        "race_url":
            race_url,
    }


# ============================================================
# RACECARD TABLE
# ============================================================

TABLE_HEADER_ALIASES = {

    "horse_number": {
        "horse no",
        "horse number",
    },

    "horse_name": {
        "horse",
        "name",
    },

    "handicap_weight": {
        "wt",
        "handicap weight",
    },

    "jockey": {
        "jockey",
    },

    "draw": {
        "draw",
        "dr",
    },

    "trainer": {
        "trainer",
    },

    "horse_rating": {
        "rtg",
        "rating",
    },

    "declared_horse_weight": {
        "horse wt declaration",
        "horse wt declaration horse weight",
        "horse weight declaration",
    },

    "days_since_last_run": {
        "days since last run",
    },
}


# Deliberately excluded:
#
# Last 6 Runs
# Colour
# Brand No.
# Probable Overweight
# International Rating
# Rating +/-
# Horse Weight Change
# Best Time
# WFA
# Priority
# Gear
# Owner
# Import Category
#
# Age / Sex / Stakes / Sire / Dam are fetched
# from each horse profile instead.


def identify_table_header(value):

    normalised = (
        normalise_header(
            value
        )
    )

    for (
        canonical,
        aliases,
    ) in (
        TABLE_HEADER_ALIASES.items()
    ):

        if normalised in aliases:

            return canonical

    return None


def direct_cells(
    row,
    names=("th", "td"),
):

    cells = row.find_all(
        list(
            names
        ),
        recursive=False,
    )

    if cells:

        return cells

    return row.find_all(
        list(
            names
        )
    )


# ============================================================
# FIND THE TRUE HKJC STARTER HEADER
# ============================================================

def build_column_map_from_header_row(
    row,
):

    cells = direct_cells(
        row
    )

    if not cells:

        return {}

    normalised_cells = [
        normalise_header(
            cell.get_text(
                " ",
                strip=True,
            )
        )
        for cell in
        cells
    ]

    normalised_set = set(
        normalised_cells
    )

    # --------------------------------------------------------
    # Fingerprint of the actual HKJC starter table.
    #
    # This prevents other tables / controls on the page being
    # mistaken for the horse table.
    # --------------------------------------------------------

    fingerprint = {
        "horse no",
        "last 6 runs",
        "colour",
        "horse",
        "wt",
        "jockey",
        "draw",
        "trainer",
        "rtg",
        "horse wt declaration",
        "days since last run",
    }

    if not fingerprint.issubset(
        normalised_set
    ):

        return {}

    column_map = {}

    for (
        index,
        cell,
    ) in enumerate(
        cells
    ):

        canonical = (
            identify_table_header(
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
            not in
            column_map
        ):

            column_map[
                canonical
            ] = index

    required = {
        "horse_number",
        "horse_name",
        "handicap_weight",
        "jockey",
        "draw",
        "trainer",
        "horse_rating",
        "declared_horse_weight",
        "days_since_last_run",
    }

    if required.issubset(
        column_map
    ):

        return column_map

    return {}


def find_main_runner_table(
    soup,
):

    # --------------------------------------------------------
    # Return:
    #
    # table
    # true header row
    # column map
    # --------------------------------------------------------

    for table in soup.find_all(
        "table"
    ):

        for row in table.find_all(
            "tr"
        ):

            column_map = (
                build_column_map_from_header_row(
                    row
                )
            )

            if column_map:

                return (
                    table,
                    row,
                    column_map,
                )

    return (
        None,
        None,
        {},
    )


def get_cell(
    cells,
    column_map,
    field,
):

    index = (
        column_map.get(
            field
        )
    )

    if (
        index is None
        or
        index < 0
        or
        index >= len(
            cells
        )
    ):

        return None

    return cells[
        index
    ]


def get_cell_text(
    cells,
    column_map,
    field,
):

    cell = get_cell(
        cells,
        column_map,
        field,
    )

    if cell is None:
        return ""

    return clean_text(
        cell.get_text(
            " ",
            strip=True,
        )
    )


def find_horse_link(
    container,
):

    if container is None:
        return None

    for link in container.find_all(
        "a",
        href=True,
    ):

        absolute = urljoin(
            "https://racing.hkjc.com",
            link.get(
                "href",
                "",
            ),
        )

        if extract_horse_id(
            absolute
        ):

            return link

    return None


# ============================================================
# DUPLICATE PROTECTION
# ============================================================

def runner_completeness_score(
    runner,
):

    fields = (
        "horse_name",
        "handicap_weight",
        "jockey",
        "draw",
        "trainer",
        "horse_rating",
        "declared_horse_weight",
        "days_since_last_run",
        "horse_id",
        "horse_profile_url",
    )

    score = 0

    for field in fields:

        value = clean_text(
            runner.get(
                field
            )
        )

        if value not in {
            "",
            "None",
            "nan",
        }:

            score += 1

    return score


def dedupe_race_runners(
    runners,
):

    # --------------------------------------------------------
    # FIRST GUARD:
    # One row per horse number within a race.
    # --------------------------------------------------------

    by_number = {}

    for runner in runners:

        key = (
            runner.get(
                "race_id"
            ),
            runner.get(
                "horse_number"
            ),
        )

        existing = (
            by_number.get(
                key
            )
        )

        if (
            existing is None
            or
            runner_completeness_score(
                runner
            )
            >
            runner_completeness_score(
                existing
            )
        ):

            if existing is not None:

                print(
                    f"DUPLICATE HORSE NUMBER "
                    f"REPLACED: "
                    f"{runner.get('race_id')} "
                    f"horse "
                    f"{runner.get('horse_number')}"
                )

            by_number[
                key
            ] = runner

        else:

            print(
                f"DUPLICATE HORSE NUMBER "
                f"IGNORED: "
                f"{runner.get('race_id')} "
                f"horse "
                f"{runner.get('horse_number')}"
            )

    # --------------------------------------------------------
    # SECOND GUARD:
    # The same horse ID cannot appear twice in one race.
    # --------------------------------------------------------

    by_horse_id = {}

    no_id_rows = []

    for runner in (
        by_number.values()
    ):

        horse_id = clean_text(
            runner.get(
                "horse_id"
            )
        )

        if not horse_id:

            no_id_rows.append(
                runner
            )

            continue

        key = (
            runner.get(
                "race_id"
            ),
            horse_id,
        )

        existing = (
            by_horse_id.get(
                key
            )
        )

        if (
            existing is None
            or
            runner_completeness_score(
                runner
            )
            >
            runner_completeness_score(
                existing
            )
        ):

            by_horse_id[
                key
            ] = runner

    return (
        list(
            by_horse_id.values()
        )
        +
        no_id_rows
    )


# ============================================================
# PARSE STARTERS
# ============================================================

def parse_runners(
    soup,
    race_header,
):

    (
        table,
        header_row,
        column_map,
    ) = find_main_runner_table(
        soup
    )

    if (
        table is None
        or
        header_row is None
        or
        not column_map
    ):

        return []

    rows = table.find_all(
        "tr"
    )

    try:

        header_index = (
            rows.index(
                header_row
            )
        )

    except ValueError:

        header_index = -1

    runners = []

    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # Parse only rows AFTER the true starter-table header.
    #
    # This prevents HKJC setup controls / duplicated responsive
    # markup from being interpreted as actual starters.
    # --------------------------------------------------------

    for row in rows[
        header_index + 1:
    ]:

        cells = direct_cells(
            row,
            names=("td",),
        )

        if not cells:

            continue

        horse_number = (
            parse_int(
                get_cell_text(
                    cells,
                    column_map,
                    "horse_number",
                )
            )
        )

        # Genuine starter numbers only.

        if (
            horse_number is None
            or
            horse_number < 1
            or
            horse_number > 30
        ):

            continue

        horse_cell = get_cell(
            cells,
            column_map,
            "horse_name",
        )

        horse_link = (
            find_horse_link(
                horse_cell
            )
        )

        if horse_link is None:

            horse_link = (
                find_horse_link(
                    row
                )
            )

        if horse_link is None:

            continue

        horse_profile_url = (
            urljoin(
                "https://racing.hkjc.com",
                horse_link.get(
                    "href",
                    "",
                ),
            )
        )

        horse_id = (
            extract_horse_id(
                horse_profile_url
            )
        )

        horse_name = clean_text(
            horse_link.get_text(
                " ",
                strip=True,
            )
        )

        if (
            not horse_id
            or
            not horse_name
        ):

            continue

        runner = dict(
            race_header
        )

        runner.update({

            "horse_number":
                horse_number,

            "horse_name":
                horse_name,

            "handicap_weight":
                parse_int(
                    get_cell_text(
                        cells,
                        column_map,
                        "handicap_weight",
                    )
                ),

            "jockey":
                get_cell_text(
                    cells,
                    column_map,
                    "jockey",
                ),

            "draw":
                parse_int(
                    get_cell_text(
                        cells,
                        column_map,
                        "draw",
                    )
                ),

            "trainer":
                get_cell_text(
                    cells,
                    column_map,
                    "trainer",
                ),

            "horse_rating":
                parse_int(
                    get_cell_text(
                        cells,
                        column_map,
                        "horse_rating",
                    )
                ),

            "declared_horse_weight":
                parse_int(
                    get_cell_text(
                        cells,
                        column_map,
                        "declared_horse_weight",
                    )
                ),

            "days_since_last_run":
                parse_int(
                    get_cell_text(
                        cells,
                        column_map,
                        "days_since_last_run",
                    )
                ),

            # Filled from horse profile.
            "horse_age":
                None,

            "horse_sex":
                "",

            "season_stakes_hkd":
                None,

            "total_stakes_hkd":
                None,

            "sire":
                "",

            "dam":
                "",

            "horse_id":
                horse_id,

            "runner_id":
                (
                    f"{race_header['race_id']}_"
                    f"{horse_number:02d}"
                ),

            "horse_profile_url":
                horse_profile_url,
        })

        runners.append(
            runner
        )

    # --------------------------------------------------------
    # Remove duplicate responsive-table copies.
    # --------------------------------------------------------

    runners = (
        dedupe_race_runners(
            runners
        )
    )

    runners.sort(
        key=lambda item:
            item.get(
                "horse_number"
            )
            or
            999
    )

    numbers = [
        runner.get(
            "horse_number"
        )
        for runner in
        runners
    ]

    print(
        f"STARTERS PARSED "
        f"{race_header['race_id']}: "
        f"{numbers}"
    )

    return runners


# ============================================================
# HORSE PROFILE
#
# ONLY:
#
# Age
# Sex
# Season Stakes
# Total Stakes
# Sire
# Dam
# ============================================================

def extract_horse_profile_fields(
    html,
    horse_id,
    profile_url,
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

    age = None

    sex = ""

    season_stakes_hkd = None

    total_stakes_hkd = None

    sire = ""

    dam = ""

    # --------------------------------------------------------
    # AGE
    #
    # Country of Origin / Age : NZ / 6
    # --------------------------------------------------------

    origin_age = re.search(
        r"Country of Origin"
        r"\s*/\s*"
        r"Age"
        r"\s*:\s*"
        r"(.*?)"
        r"\s+"
        r"Colour"
        r"\s*/\s*"
        r"Sex"
        r"\s*:",
        text,
        re.I,
    )

    if origin_age:

        age_match = re.search(
            r"/\s*"
            r"(\d+)"
            r"\s*$",
            clean_text(
                origin_age.group(
                    1
                )
            ),
        )

        if age_match:

            age = int(
                age_match.group(
                    1
                )
            )

    # --------------------------------------------------------
    # SEX
    #
    # Colour / Sex : Bay / Gelding
    #
    # Colour itself is not saved.
    # --------------------------------------------------------

    colour_sex = re.search(
        r"Colour"
        r"\s*/\s*"
        r"Sex"
        r"\s*:\s*"
        r"(.*?)"
        r"\s+"
        r"Import Type"
        r"\s*:",
        text,
        re.I,
    )

    if colour_sex:

        value = clean_text(
            colour_sex.group(
                1
            )
        )

        if "/" in value:

            sex = clean_text(
                value.rsplit(
                    "/",
                    1,
                )[1]
            )

        else:

            sex = value

    # --------------------------------------------------------
    # SEASON STAKES
    # --------------------------------------------------------

    season_match = re.search(
        r"Season Stakes"
        r"\*?"
        r"\s*:\s*"
        r"((?:HK\s*)?"
        r"\$\s*"
        r"[\d,]+)",
        text,
        re.I,
    )

    if season_match:

        season_stakes_hkd = (
            parse_money(
                season_match.group(
                    1
                )
            )
        )

    # --------------------------------------------------------
    # TOTAL STAKES
    # --------------------------------------------------------

    total_match = re.search(
        r"Total Stakes"
        r"\*?"
        r"\s*:\s*"
        r"((?:HK\s*)?"
        r"\$\s*"
        r"[\d,]+)",
        text,
        re.I,
    )

    if total_match:

        total_stakes_hkd = (
            parse_money(
                total_match.group(
                    1
                )
            )
        )

    # --------------------------------------------------------
    # SIRE
    # --------------------------------------------------------

    sire_match = re.search(
        r"\bSire"
        r"\s*:\s*"
        r"(.*?)"
        r"\s+Dam"
        r"\s*:",
        text,
        re.I,
    )

    if sire_match:

        sire = clean_text(
            sire_match.group(
                1
            )
        )

    # --------------------------------------------------------
    # DAM
    # --------------------------------------------------------

    dam_match = re.search(
        r"\bDam"
        r"\s*:\s*"
        r"(.*?)"
        r"\s+Dam['’]s Sire"
        r"\s*:",
        text,
        re.I,
    )

    if dam_match:

        dam = clean_text(
            dam_match.group(
                1
            )
        )

    return {
        "horse_id":
            horse_id,

        "horse_profile_url":
            profile_url,

        "horse_age":
            age,

        "horse_sex":
            sex,

        "season_stakes_hkd":
            season_stakes_hkd,

        "total_stakes_hkd":
            total_stakes_hkd,

        "sire":
            sire,

        "dam":
            dam,
    }


def fetch_horse_profile(
    horse_id,
    profile_url,
):

    response = http_get(
        profile_url
    )

    if response is None:

        return (
            horse_id,
            None,
        )

    try:

        profile = (
            extract_horse_profile_fields(
                response.text,
                horse_id,
                response.url,
            )
        )

        return (
            horse_id,
            profile,
        )

    except Exception as exc:

        print(
            f"HORSE PROFILE PARSE FAILED "
            f"{horse_id}: "
            f"{exc}"
        )

        return (
            horse_id,
            None,
        )


# ============================================================
# FETCH ONE RACE
# ============================================================

def fetch_race(
    task,
):

    response = http_get(
        task[
            "race_url"
        ]
    )

    if response is None:

        return (
            task,
            [],
            "request_failed",
        )

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    header = parse_race_header(
        soup,
        task[
            "racecourse_code"
        ],
        task[
            "race_number"
        ],
        response.url,
    )

    runners = parse_runners(
        soup,
        header,
    )

    # --------------------------------------------------------
    # Because we deliberately probe R1-R12,
    # a race number with no table is a normal skip.
    # --------------------------------------------------------

    if not runners:

        return (
            task,
            [],
            "no_runner_table",
        )

    return (
        task,
        runners,
        None,
    )


# ============================================================
# FETCH ALL RACES CONCURRENTLY
# ============================================================

def fetch_all_races(
    tasks,
):

    if not tasks:
        return []

    workers = min(
        RACE_WORKERS,
        len(
            tasks
        ),
    )

    all_runners = []

    print()

    print(
        "=" * 70
    )

    print(
        "RACE PHASE"
    )

    print(
        "=" * 70
    )

    print(
        f"Race pages queued: "
        f"{len(tasks)}"
    )

    print(
        f"Concurrent race workers: "
        f"{workers}"
    )

    with ThreadPoolExecutor(
        max_workers=
            workers
    ) as executor:

        futures = {
            executor.submit(
                fetch_race,
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
                    runners,
                    status,
                ) = future.result()

            except Exception as exc:

                print(
                    f"RACE WORKER FAILED "
                    f"{task['racecourse_code']} "
                    f"R{task['race_number']}: "
                    f"{exc}"
                )

                continue

            if (
                status
                ==
                "no_runner_table"
            ):

                print(
                    f"SKIP NO RACECARD TABLE: "
                    f"{returned_task['racecourse_code']} "
                    f"R{returned_task['race_number']}"
                )

                continue

            if status is not None:

                print(
                    f"RACE FAILED: "
                    f"{returned_task['racecourse_code']} "
                    f"R{returned_task['race_number']} "
                    f"{status}"
                )

                continue

            print(
                f"RACE OK: "
                f"{returned_task['racecourse_code']} "
                f"R{returned_task['race_number']} "
                f"-> "
                f"{len(runners)} "
                f"runners"
            )

            all_runners.extend(
                runners
            )

    return all_runners


# ============================================================
# FETCH UNIQUE HORSE PROFILES
# ============================================================

def enrich_horses(
    runners,
):

    unique_horses = {}

    for runner in runners:

        horse_id = clean_text(
            runner.get(
                "horse_id"
            )
        )

        profile_url = clean_text(
            runner.get(
                "horse_profile_url"
            )
        )

        if (
            horse_id
            and
            profile_url
        ):

            unique_horses[
                horse_id
            ] = profile_url

    if not unique_horses:

        return runners

    workers = min(
        HORSE_WORKERS,
        len(
            unique_horses
        ),
    )

    profiles = {}

    print()

    print(
        "=" * 70
    )

    print(
        "HORSE PROFILE PHASE"
    )

    print(
        "=" * 70
    )

    print(
        f"Unique horse pages: "
        f"{len(unique_horses)}"
    )

    print(
        f"Concurrent horse workers: "
        f"{workers}"
    )

    with ThreadPoolExecutor(
        max_workers=
            workers
    ) as executor:

        futures = {
            executor.submit(
                fetch_horse_profile,
                horse_id,
                profile_url,
            ):
                horse_id

            for (
                horse_id,
                profile_url,
            ) in (
                unique_horses.items()
            )
        }

        for future in as_completed(
            futures
        ):

            horse_id = futures[
                future
            ]

            try:

                (
                    returned_id,
                    profile,
                ) = future.result()

            except Exception as exc:

                print(
                    f"HORSE WORKER FAILED "
                    f"{horse_id}: "
                    f"{exc}"
                )

                continue

            if (
                returned_id
                ==
                horse_id
                and
                profile
            ):

                profiles[
                    horse_id
                ] = profile

    # --------------------------------------------------------
    # Merge profile info back into each runner.
    # --------------------------------------------------------

    for runner in runners:

        horse_id = clean_text(
            runner.get(
                "horse_id"
            )
        )

        profile = profiles.get(
            horse_id
        )

        if not profile:

            continue

        runner[
            "horse_age"
        ] = profile.get(
            "horse_age"
        )

        runner[
            "horse_sex"
        ] = profile.get(
            "horse_sex",
            "",
        )

        runner[
            "season_stakes_hkd"
        ] = profile.get(
            "season_stakes_hkd"
        )

        runner[
            "total_stakes_hkd"
        ] = profile.get(
            "total_stakes_hkd"
        )

        runner[
            "sire"
        ] = profile.get(
            "sire",
            "",
        )

        runner[
            "dam"
        ] = profile.get(
            "dam",
            "",
        )

        runner[
            "horse_profile_url"
        ] = profile.get(
            "horse_profile_url",
            runner.get(
                "horse_profile_url",
                "",
            ),
        )

    return runners


# ============================================================
# FINAL DUPLICATE PROTECTION
# ============================================================

def sort_runners(
    runners,
):

    # --------------------------------------------------------
    # Final safety net across the entire day.
    #
    # There must be exactly one row for each:
    #
    # race_id + horse_number
    #
    # If duplicate markup survives, keep whichever copy has
    # the most useful information.
    # --------------------------------------------------------

    unique = {}

    for row in runners:

        key = (
            row.get(
                "race_id"
            ),
            row.get(
                "horse_number"
            ),
        )

        existing = (
            unique.get(
                key
            )
        )

        if (
            existing is None
            or
            runner_completeness_score(
                row
            )
            >
            runner_completeness_score(
                existing
            )
        ):

            unique[
                key
            ] = row

    return sorted(
        unique.values(),
        key=lambda row:
            (
                clean_text(
                    row.get(
                        "racecourse_code"
                    )
                ),

                (
                    row.get(
                        "race_number"
                    )
                    if (
                        row.get(
                            "race_number"
                        )
                        is not None
                    )
                    else
                    999
                ),

                (
                    row.get(
                        "horse_number"
                    )
                    if (
                        row.get(
                            "horse_number"
                        )
                        is not None
                    )
                    else
                    999
                ),
            ),
    )


# ============================================================
# WRITE CSV ONLY
# ============================================================

def write_csv(
    runners,
):

    df = pd.DataFrame(
        runners
    )

    for column in (
        OUTPUT_COLUMNS
    ):

        if column not in df.columns:

            df[
                column
            ] = None

    df[
        OUTPUT_COLUMNS
    ].to_csv(
        OUTPUT_CSV,
        index=False,
    )


def write_outputs(
    runners,
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    runners = sort_runners(
        runners
    )

    write_csv(
        runners
    )

    print()

    print(
        f"Saved "
        f"{len(runners)} "
        f"runner rows"
    )

    print(
        f"CSV: "
        f"{OUTPUT_CSV}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    race_date = (
        parse_requested_date()
    )

    print(
        "=" * 70
    )

    print(
        "HKJC UPCOMING RACECARD COLLECTOR"
    )

    print(
        "=" * 70
    )

    print(
        f"Race date: "
        f"{race_date}"
    )

    print(
        f"Race workers: "
        f"{RACE_WORKERS}"
    )

    print(
        f"Horse workers: "
        f"{HORSE_WORKERS}"
    )

    # ========================================================
    # STEP 1
    # Find venue and queue R1-R12.
    # ========================================================

    tasks = discover_races()

    if not tasks:

        print(
            "No local HKJC "
            "races/racecards found "
            "for the requested date."
        )

        write_outputs(
            []
        )

        return

    # ========================================================
    # STEP 2
    # Fetch all racecards.
    # ========================================================

    runners = fetch_all_races(
        tasks
    )

    if not runners:

        print(
            "No published runner "
            "tables found for the "
            "requested date."
        )

        write_outputs(
            []
        )

        return

    # ========================================================
    # STEP 3
    # Fetch each unique horse profile once.
    # ========================================================

    runners = enrich_horses(
        runners
    )

    # ========================================================
    # STEP 4
    # Deduplicate + save CSV.
    # ========================================================

    write_outputs(
        runners
    )


if __name__ == "__main__":
    main()
