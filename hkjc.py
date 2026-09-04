import os
import time
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://racing.hkjc.com/en-us/local/information/localresults"

START_DATE = os.getenv("START_DATE", "2006-12-01")
END_DATE = os.getenv("END_DATE", "2006-12-31")

OUTPUT_DIR = "results"

# Be polite to the public website.
DELAY_SECONDS = 2

session = requests.Session()

session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/150.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
)


def date_range(start_date, end_date):
    current = start_date

    while current <= end_date:
        yield current
        current += timedelta(days=1)


def request_page(race_date, racecourse=None, race_no=None):
    params = {
        "racedate": race_date.strftime("%Y/%m/%d")
    }

    if racecourse:
        params["Racecourse"] = racecourse

    if race_no:
        params["RaceNo"] = race_no

    response = session.get(
        BASE_URL,
        params=params,
        timeout=30,
    )

    print(
        f"GET {response.url} "
        f"-> {response.status_code}"
    )

    response.raise_for_status()

    return response


def detect_meeting(html):
    soup = BeautifulSoup(html, "html.parser")

    meeting_text = soup.select_one(".raceMeeting_select")

    if not meeting_text:
        return None

    text = meeting_text.get_text(
        " ",
        strip=True
    )

    if "Happy Valley" in text:
        return "HV"

    if "Sha Tin" in text:
        return "ST"

    return None


def detect_race_numbers(html):
    soup = BeautifulSoup(html, "html.parser")

    race_numbers = set()

    for link in soup.select(
        'a[href*="RaceNo="]'
    ):
        href = link.get("href", "")

        if "RaceNo=" not in href:
            continue

        try:
            race_no = href.split(
                "RaceNo="
            )[1].split("&")[0]

            race_numbers.add(
                int(race_no)
            )
        except Exception:
            pass

    # Race 1 is often shown as the current
    # race and therefore may not have a link.
    if soup.find(
        string=lambda x:
        x and "RACE 1" in x
    ):
        race_numbers.add(1)

    return sorted(race_numbers)


def extract_race_table(
    html,
    meeting_date,
    racecourse,
    race_no
):
    soup = BeautifulSoup(html, "html.parser")

    performance = soup.select_one(
        ".performance table"
    )

    if performance is None:
        return None

    tables = pd.read_html(
        StringIO(str(performance))
    )

    if not tables:
        return None

    df = tables[0]

    df.insert(
        0,
        "RaceNo",
        race_no
    )

    df.insert(
        0,
        "Racecourse",
        racecourse
    )

    df.insert(
        0,
        "MeetingDate",
        meeting_date.strftime(
            "%Y-%m-%d"
        )
    )

    return df


def extract_dividends(
    html,
    meeting_date,
    racecourse,
    race_no
):
    soup = BeautifulSoup(html, "html.parser")

    dividend_table = soup.select_one(
        ".dividend_tab table"
    )

    if dividend_table is None:
        return None

    tables = pd.read_html(
        StringIO(str(dividend_table))
    )

    if not tables:
        return None

    df = tables[0]

    df.insert(
        0,
        "RaceNo",
        race_no
    )

    df.insert(
        0,
        "Racecourse",
        racecourse
    )

    df.insert(
        0,
        "MeetingDate",
        meeting_date.strftime(
            "%Y-%m-%d"
        )
    )

    return df


def process_date(meeting_date):
    print(
        "\nChecking",
        meeting_date.strftime(
            "%Y-%m-%d"
        )
    )

    try:
        first_response = request_page(
            meeting_date
        )
    except requests.RequestException as exc:
        print(
            "Request failed:",
            exc
        )
        return

    racecourse = detect_meeting(
        first_response.text
    )

    if not racecourse:
        print(
            "No HK meeting detected."
        )
        return

    print(
        "Meeting found:",
        racecourse
    )

    race_numbers = detect_race_numbers(
        first_response.text
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

    all_results = []
    all_dividends = []

    for race_no in race_numbers:
        try:
            response = request_page(
                meeting_date,
                racecourse,
                race_no
            )

            results = extract_race_table(
                response.text,
                meeting_date,
                racecourse,
                race_no
            )

            dividends = extract_dividends(
                response.text,
                meeting_date,
                racecourse,
                race_no
            )

            if results is not None:
                all_results.append(
                    results
                )

                print(
                    f"Race {race_no}: "
                    f"{len(results)} runners"
                )

            if dividends is not None:
                all_dividends.append(
                    dividends
                )

        except Exception as exc:
            print(
                f"Race {race_no} failed:",
                exc
            )

        time.sleep(
            DELAY_SECONDS
        )

    if not all_results:
        print(
            "No results extracted."
        )
        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    date_string = meeting_date.strftime(
        "%Y-%m-%d"
    )

    results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    results_file = os.path.join(
        OUTPUT_DIR,
        f"{date_string}_{racecourse}_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    print(
        "Saved:",
        results_file
    )

    if all_dividends:
        dividends_df = pd.concat(
            all_dividends,
            ignore_index=True
        )

        dividend_file = os.path.join(
            OUTPUT_DIR,
            f"{date_string}_{racecourse}_dividends.csv"
        )

        dividends_df.to_csv(
            dividend_file,
            index=False
        )

        print(
            "Saved:",
            dividend_file
        )


def main():
    start = datetime.strptime(
        START_DATE,
        "%Y-%m-%d"
    ).date()

    end = datetime.strptime(
        END_DATE,
        "%Y-%m-%d"
    ).date()

    print(
        "HKJC collection range:"
    )

    print(
        start,
        "to",
        end
    )

    for meeting_date in date_range(
        start,
        end
    ):
        process_date(
            meeting_date
        )

        time.sleep(
            DELAY_SECONDS
        )


if __name__ == "__main__":
    main()
