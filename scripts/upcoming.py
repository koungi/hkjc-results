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

    # Race information
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

    # Runner information
    "horse_number",
    "horse_name",
    "handicap_weight",
    "jockey",
    "draw",
    "trainer",
    "horse_rating",
    "declared_horse_weight",
    "days_since_last_run",

    # Horse profile information
    "horse_age",
    "horse_sex",
    "season_stakes_hkd",
    "total_stakes_hkd",
    "sire",
    "dam",

    # IDs / sources
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
# GENERAL HELPERS
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

    text = clean_text(
        value
    ).replace(
        ",",
        "",
    )

    match = re.search(
        r"-?\d+",
        text,
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

    text = clean_text(
        value
    )

    match = re.search(
        r"(?:HK\s*)?"
        r"\$\s*"
        r"([\d,]+)",
        text,
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

    text = clean_text(
        value
    ).lower()

    text = text.replace(
        "’",
        "'",
    )

    text = text.replace(
        "+/-",
        " vs ",
    )

    text = text.replace(
        "+",
        " ",
    )

    text = text.replace(
        "-",
        " ",
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
            "RACE_DATE must use "
            "YYYY-MM-DD, for example "
            "2026-09-06"
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
# DISCOVER LOCAL MEETING
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
    # First detect venue from valid racecard links.
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

        linked_race_no = (
            parse_int(
                query_value_case_insensitive(
                    query,
                    "RaceNo",
                )
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

        if linked_race_no is not None:

            discovered_numbers.add(
                linked_race_no
            )

    # --------------------------------------------------------
    # Fallback venue detection.
    # --------------------------------------------------------

    if not racecourse:

        page_text = clean_text(
            soup.get_text(
                " ",
                strip=True,
            )
        )

        if "Sha Tin" in page_text:

            racecourse = "ST"

        elif (
            "Happy Valley"
            in
            page_text
        ):

            racecourse = "HV"

        else:

            print(
                "No local Hong Kong "
                "meeting found for "
                "this date."
            )

            return []

    print(
        f"Detected venue: "
        f"{racecourse}"
    )

    print(
        f"Navigation race numbers: "
        f"{sorted(discovered_numbers)}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Always probe races 1 through 12.
    #
    # This prevents Race 1 being missed if the navigation
    # HTML doesn't expose it correctly.
    #
    # Non-existent race numbers simply get skipped later.
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
        "Queued Race 1 through "
        "Race 12."
    )

    return tasks


# ============================================================
# RACE CLASS
# ============================================================

def normalise_race_class(value):

    text = clean_text(
        value
    )

    if not text:
        return ""

    # --------------------------------------------------------
    # CLASS 1 TO CLASS 5
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # GROUP 1 / 2 / 3
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Current HKJC format:
    #
    # Rating: 40-0
    # Rating: 60-40
    # Rating: 80-60
    # Rating: 100-80
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fallback without colon.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final fallback:
    # standalone numeric range.
    # --------------------------------------------------------

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
    # RACE NAME
    #
    # Example:
    # Race 1 - THE EXAMPLE HANDICAP
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

    # --------------------------------------------------------
    # Look at strings near the race heading.
    # --------------------------------------------------------

    if race_heading_index is not None:

        nearby = strings[
            race_heading_index:
            race_heading_index + 20
        ]

    else:

        nearby = strings

    for text in nearby:

        # ----------------------------------------------------
        # DATE / VENUE / START TIME
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
        # TRACK / COURSE / DISTANCE / GOING
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
        # PRIZE MONEY / RATING BAND / CLASS
        #
        # Examples:
        #
        # Prize Money: $875,000, Rating: 40-0, Class 5
        #
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
    # FALLBACK SEARCH FOR RATING BAND / CLASS
    #
    # Some HKJC layouts split the header into separate HTML
    # elements. Search nearby strings again if needed.
    # --------------------------------------------------------

    if not rating_band:

        combined_nearby = clean_text(
            " ".join(
                nearby
            )
        )

        rating_band = (
            extract_rating_band_from_header(
                combined_nearby
            )
        )

    if not race_class:

        combined_nearby = clean_text(
            " ".join(
                nearby
            )
        )

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
# RACECARD RUNNER TABLE
#
# INCLUDED:
#
# horse number
# horse name
# handicap weight
# jockey
# draw
# trainer
# rating
# declared horse weight
# days since last run
#
# NOT INCLUDED:
#
# last 6 runs
# colour
# brand number
# probable overweight
# international rating
# rating +/-
# horse weight change
# best time
# WFA
# priority
# gear
# owner
# import category
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


def identify_table_header(value):

    normalised = (
        normalise_header(
            value
        )
    )

    if not normalised:
        return None

    for (
        canonical,
        aliases,
    ) in (
        TABLE_HEADER_ALIASES.items()
    ):

        if normalised in aliases:

            return canonical

    return None


def build_column_map(table):

    best_map = {}

    for row in table.find_all(
        "tr"
    ):

        cells = row.find_all(
            [
                "th",
                "td",
            ],
            recursive=False,
        )

        if not cells:

            cells = row.find_all(
                [
                    "th",
                    "td",
                ]
            )

        current = {}

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
                current
            ):

                current[
                    canonical
                ] = index

        required = {
            "horse_number",
            "horse_name",
            "jockey",
            "trainer",
        }

        if (
            required.issubset(
                current.keys()
            )
            and
            len(current)
            >
            len(best_map)
        ):

            best_map = (
                current
            )

    return best_map


def find_main_runner_table(soup):

    best_table = None

    best_map = {}

    for table in soup.find_all(
        "table"
    ):

        column_map = (
            build_column_map(
                table
            )
        )

        if (
            len(column_map)
            >
            len(best_map)
        ):

            best_table = (
                table
            )

            best_map = (
                column_map
            )

    return (
        best_table,
        best_map,
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


# ============================================================
# PARSE RUNNERS
# ============================================================

def parse_runners(
    soup,
    race_header,
):

    (
        table,
        column_map,
    ) = find_main_runner_table(
        soup
    )

    if (
        table is None
        or
        not column_map
    ):

        return []

    runners = []

    for row in table.find_all(
        "tr"
    ):

        cells = row.find_all(
            "td",
            recursive=False,
        )

        if not cells:

            cells = row.find_all(
                "td"
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

        horse_cell = get_cell(
            cells,
            column_map,
            "horse_name",
        )

        horse_link = None

        if horse_cell is not None:

            horse_link = (
                horse_cell.find(
                    "a",
                    href=re.compile(
                        r"horse\?horseid=",
                        re.I,
                    ),
                )
            )

        if horse_link is None:

            horse_link = row.find(
                "a",
                href=re.compile(
                    r"horse\?horseid=",
                    re.I,
                ),
            )

        if (
            horse_number is None
            or
            horse_link is None
        ):

            continue

        horse_profile_url = urljoin(
            "https://racing.hkjc.com",
            horse_link.get(
                "href",
                "",
            ),
        )

        horse_id = extract_horse_id(
            horse_profile_url
        )

        horse_name = clean_text(
            horse_link.get_text(
                " ",
                strip=True,
            )
        )

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
    # DUPLICATE PROTECTION
    #
    # HKJC pages can contain duplicated responsive markup.
    #
    # Keep exactly one copy of a horse within each race.
    # --------------------------------------------------------

    deduped = []

    seen = set()

    for runner in runners:

        horse_id = clean_text(
            runner.get(
                "horse_id"
            )
        )

        if horse_id:

            key = (
                runner.get(
                    "race_id"
                ),
                "horse_id",
                horse_id,
            )

        else:

            key = (
                runner.get(
                    "race_id"
                ),
                "horse_number",
                runner.get(
                    "horse_number"
                ),
            )

        if key in seen:

            print(
                f"DUPLICATE RUNNER IGNORED: "
                f"{runner.get('race_id')} "
                f"Horse "
                f"{runner.get('horse_number')} "
                f"{runner.get('horse_name')}"
            )

            continue

        seen.add(
            key
        )

        deduped.append(
            runner
        )

    return deduped


# ============================================================
# HORSE PROFILE
#
# ONLY:
#
# age
# sex
# season stakes
# total stakes
# sire
# dam
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
    # Example:
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
    # Example:
    # Colour / Sex : Bay / Gelding
    #
    # Colour is deliberately ignored.
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

def fetch_race(task):

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

    race_header = (
        parse_race_header(
            soup,
            task[
                "racecourse_code"
            ],
            task[
                "race_number"
            ],
            response.url,
        )
    )

    runners = parse_runners(
        soup,
        race_header,
    )

    # --------------------------------------------------------
    # NO TABLE = NORMAL SKIP
    #
    # This is not treated as an error because we deliberately
    # probe Race 1 through Race 12.
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

def fetch_all_races(tasks):

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
        max_workers=workers
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
                f"{len(runners)} runners"
            )

            all_runners.extend(
                runners
            )

    return all_runners


# ============================================================
# FETCH UNIQUE HORSE PROFILES
# ============================================================

def enrich_horses(runners):

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
        max_workers=workers
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
    # MERGE PROFILE DATA INTO EACH RUNNER
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
# FINAL DUPLICATE PROTECTION + SORT
# ============================================================

def sort_runners(runners):

    # --------------------------------------------------------
    # Final safety net:
    #
    # one horse = one row within a race.
    # --------------------------------------------------------

    unique = {}

    for row in runners:

        horse_id = clean_text(
            row.get(
                "horse_id"
            )
        )

        if horse_id:

            key = (
                row.get(
                    "race_id"
                ),
                "horse_id",
                horse_id,
            )

        else:

            key = (
                row.get(
                    "race_id"
                ),
                "horse_number",
                row.get(
                    "horse_number"
                ),
            )

        if key not in unique:

            unique[
                key
            ] = row

        else:

            print(
                f"FINAL DUPLICATE REMOVED: "
                f"{row.get('race_id')} "
                f"Horse "
                f"{row.get('horse_number')} "
                f"{row.get('horse_name')}"
            )

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
                    else 999
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
                    else 999
                ),
            ),
    )


# ============================================================
# WRITE CSV
# ============================================================

def write_csv(runners):

    df = pd.DataFrame(
        runners
    )

    for column in OUTPUT_COLUMNS:

        if column not in df.columns:

            df[
                column
            ] = None

    df = df[
        OUTPUT_COLUMNS
    ]

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )


def write_outputs(runners):

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
        "HKJC UPCOMING "
        "RACECARD COLLECTOR"
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
    # 1. DISCOVER MEETING AND QUEUE R1-R12
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
    # 2. FETCH RACECARDS
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
    # 3. FETCH UNIQUE HORSE PROFILE DATA
    # ========================================================

    runners = enrich_horses(
        runners
    )

    # ========================================================
    # 4. WRITE CSV
    # ========================================================

    write_outputs(
        runners
    )


if __name__ == "__main__":
    main()
