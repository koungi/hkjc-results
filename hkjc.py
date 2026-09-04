import os
import time
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://racing.hkjc.com/en-us/local/information/archive/localresults"

START_DATE = os.getenv("START_DATE", "2006-12-01")
END_DATE = os.getenv("END_DATE", "2006-12-31")

OUTPUT_DIR = "results"
DELAY_SECONDS = 2


session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
})


def date_range(start_date, end_date):
    current = start_date

    while current <= end_date:
        yield current
        current += timedelta(days=1)


def build_url(race_date, racecourse=None, race_no=None):
    date_string = race_date.strftime("%Y/%m/%d")

    url = f"{BASE_URL}?racedate={date_string}"

    if racecourse:
        url += f"&Racecourse={racecourse}"

    if race_no:
        url += f"&RaceNo={race_no}"

    return url


def request_page(race_date, racecourse=None, race_no=None):
    url = build_url(
        race_date,
        racecourse,
        race_no
    )

    try:
        response = session.get(
            url,
            timeout=30
        )

        print(
            f"GET {url} "
            f"-> {response.status_code}"
        )

        response.raise_for_status()

        return response

    except requests.RequestException as exc:
        print(
            f"Request failed: {exc}"
        )

        return None


def detect_meeting(html):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    meeting_element = soup.select_one(
        ".raceMeeting_select"
    )

    if meeting_element is None:
        return None

    text = meeting_element.get_text(
        " ",
        strip=True
    )

    if "Happy Valley" in text:
        return "HV"

    if "Sha Tin" in text:
        return "ST"

    return None


def detect_race_numbers(html):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    race_numbers = set()

    # Race 1 may already be displayed
    # and therefore may not have a clickable link.
    race_one_found = soup.find(
        string=lambda text:
        text and "RACE 1" in text.upper()
    )

    if race_one_found:
        race_numbers.add(1)

    # Find remaining RaceNo links.
    for link in soup.select(
        'a[href*="RaceNo="]'
    ):
        href = link.get(
            "href",
            ""
        )

        try:
            race_no = href.split(
                "RaceNo="
            )[1]

            race_no = race_no.split(
                "&"
            )[0]

            race_numbers.add(
                int(race_no)
            )

        except Exception:
            pass

    return sorted(
        race_numbers
    )


def extract_race_information(
    html,
    meeting_date,
    racecourse,
    race_no
):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    race_table = soup.select_one(
        ".race_tab table"
    )

    race_info = {
        "MeetingDate": meeting_date.strftime(
            "%Y-%m-%d"
        ),
        "Racecourse": racecourse,
        "RaceNo": race_no,
        "RaceName": "",
        "ClassDistance": "",
        "Going": "",
        "Course": "",
        "PrizeMoney": "",
        "RaceTime": "",
    }

    if race_table is None:
        return race_info

    rows = race_table.find_all("tr")

    if len(rows) > 1:
        cells = rows[1].find_all("td")

        if len(cells) >= 3:
            race_info["ClassDistance"] = cells[0].get_text(
                " ",
                strip=True
            )

            race_info["Going"] = cells[-1].get_text(
                " ",
                strip=True
            )

    if len(rows) > 2:
        cells = rows[2].find_all("td")

        if len(cells) >= 3:
            race_info["RaceName"] = cells[0].get_text(
                " ",
                strip=True
            )

            race_info["Course"] = cells[-1].get_text(
                " ",
                strip=True
            )

    if len(rows) > 3:
        cells = rows[3].find_all("td")

        if len(cells) >= 1:
            race_info["PrizeMoney"] = cells[0].get_text(
                " ",
                strip=True
            )

        times = []

        for cell in cells[2:]:
            value = cell.get_text(
                " ",
                strip=True
            )

            if value:
                times.append(
                    value
                )

        if times:
            race_info["RaceTime"] = times[-1]

    return race_info


def extract_results(
    html,
    meeting_date,
    racecourse,
    race_no
):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    performance_table = soup.select_one(
        ".performance table"
    )

    if performance_table is None:
        return None

    try:
        tables = pd.read_html(
            StringIO(
                str(
                    performance_table
                )
            )
        )

    except Exception as exc:
        print(
            "Could not parse results table:",
            exc
        )

        return None

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
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    dividend_table = soup.select_one(
        ".dividend_tab table"
    )

    if dividend_table is None:
        return None

    try:
        tables = pd.read_html(
            StringIO(
                str(
                    dividend_table
                )
            )
        )

    except Exception as exc:
        print(
            "Could not parse dividend table:",
            exc
        )

        return None

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


def extract_incident_report(
    html,
    meeting_date,
    racecourse,
    race_no
):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    incident = soup.select_one(
        ".race_incident_report .info_p"
    )

    if incident is None:
        return None

    return {
        "MeetingDate": meeting_date.strftime(
            "%Y-%m-%d"
        ),
        "Racecourse": racecourse,
        "RaceNo": race_no,
        "IncidentReport": incident.get_text(
            " ",
            strip=True
        )
    }


def process_date(meeting_date):
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

    racecourse = detect_meeting(
        response.text
    )

    if racecourse is None:
        print(
            "No HKJC race meeting found."
        )

        return

    print(
        "Meeting found:",
        racecourse
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
        "Races detected:",
        race_numbers
    )

    all_results = []
    all_dividends = []
    all_race_info = []
    all_incidents = []

    for race_no in race_numbers:

        print()
        print(
            f"Processing Race {race_no}"
        )

        race_response = request_page(
            meeting_date,
            racecourse,
            race_no
        )

        if race_response is None:
            continue

        race_info = extract_race_information(
            race_response.text,
            meeting_date,
            racecourse,
            race_no
        )

        all_race_info.append(
            race_info
        )

        results = extract_results(
            race_response.text,
            meeting_date,
            racecourse,
            race_no
        )

        if results is not None:
            print(
                f"Race {race_no}: "
                f"{len(results)} runners"
            )

            all_results.append(
                results
            )

        else:
            print(
                f"Race {race_no}: "
                "No runner table found"
            )

        dividends = extract_dividends(
            race_response.text,
            meeting_date,
            racecourse,
            race_no
        )

        if dividends is not None:
            all_dividends.append(
                dividends
            )

        incident = extract_incident_report(
            race_response.text,
            meeting_date,
            racecourse,
            race_no
        )

        if incident is not None:
            all_incidents.append(
                incident
            )

        time.sleep(
            DELAY_SECONDS
        )

    if not all_results:
        print(
            "No results extracted "
            "for this meeting."
        )

        return

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    date_filename = meeting_date.strftime(
        "%Y-%m-%d"
    )

    results_df = pd.concat(
        all_results,
        ignore_index=True
    )

    results_file = os.path.join(
        OUTPUT_DIR,
        f"{date_filename}_{racecourse}_results.csv"
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

        dividends_file = os.path.join(
            OUTPUT_DIR,
            f"{date_filename}_{racecourse}_dividends.csv"
        )

        dividends_df.to_csv(
            dividends_file,
            index=False
        )

        print(
            "Saved:",
            dividends_file
        )

    if all_race_info:

        race_info_df = pd.DataFrame(
            all_race_info
        )

        race_info_file = os.path.join(
            OUTPUT_DIR,
            f"{date_filename}_{racecourse}_race_info.csv"
        )

        race_info_df.to_csv(
            race_info_file,
            index=False
        )

        print(
            "Saved:",
            race_info_file
        )

    if all_incidents:

        incidents_df = pd.DataFrame(
            all_incidents
        )

        incidents_file = os.path.join(
            OUTPUT_DIR,
            f"{date_filename}_{racecourse}_incidents.csv"
        )

        incidents_df.to_csv(
            incidents_file,
            index=False
        )

        print(
            "Saved:",
            incidents_file
        )


def main():

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
            "START_DATE and END_DATE "
            "must use YYYY-MM-DD format."
        )

        return

    if start_date > end_date:
        print(
            "START_DATE cannot be "
            "after END_DATE."
        )

        return

    print()
    print(
        "HKJC Historical Results Collector"
    )

    print(
        "From:",
        start_date
    )

    print(
        "To:",
        end_date
    )

    print()

    for meeting_date in date_range(
        start_date,
        end_date
    ):

        process_date(
            meeting_date
        )

        time.sleep(
            DELAY_SECONDS
        )


if __name__ == "__main__":
    main()
