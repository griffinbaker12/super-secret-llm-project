import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from typing import List

import openai
import requests
from bs4 import BeautifulSoup as BS
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore
from tqdm import tqdm

load_dotenv()

client = openai.OpenAI()

INPUT_CSV = "input.csv"
OUTPUT_CSV = "output.csv"
COMPARE_CSV = "comparison.csv"

INIT_URL = "https://rentry.co/n9a2hwa8"
GR_URL = "https://www.goodreads.com"
GR_DELAY = 1
MAX_WORKERS = 5  # Adjust this based on your needs and Goodreads' rate limits


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 503, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def make_request(url, max_retries=5):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for attempt in range(max_retries):
        try:
            response = requests_retry_session().get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for URL {url}: {e}")
            if attempt == max_retries - 1:
                print(f"Max retries reached. Skipping URL: {url}")
                return None
            sleep(2**attempt)  # Exponential backoff


def init_bs(html):
    return BS(html, "html.parser")


def write_csv(file, headers: List, data: List):
    with open(file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(data)


def scrape_initial_info():
    response = make_request(INIT_URL)
    if not response:
        print("Failed to fetch initial data")
        return

    bs = init_bs(response.text)

    thead = bs.find("thead")
    header_els = thead.find_all("th")  # type: ignore
    headers = [header.text.strip() for header in header_els]

    # skip headers
    table = bs.find_all("tr")[1:]

    rows = []
    for row in table:
        cells = row.find_all("td")
        print(cells)
        row_data = [cell.text.strip() for cell in cells]
        rows.append(row_data)

    write_csv(INPUT_CSV, headers, rows)


def verify_book_info():
    updated_rows = []
    comparison_rows = []
    with open(INPUT_CSV, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            title, author, *rest = row
            res = verify_title_and_author(title, author)

            if res.lower().strip() == "correct":
                updated_rows.append(row)
                print("Original entry was correct.")
            else:
                try:
                    res_parts = res.replace("\n", " ").split(", author:")
                    if len(res_parts) == 2:
                        new_title = res_parts[0].split("title:")[1].strip().strip("'")
                        new_author = res_parts[1].strip().strip("'")
                    else:
                        raise ValueError("Unexpected response format")

                    print(f"Original: {title} by {author}")
                    print(f"Updated to: {new_title} by {new_author}")

                    updated_rows.append([new_title, new_author] + rest)
                    comparison_rows.append([title, new_title, author, new_author, None])
                    print("Update applied")
                except Exception as e:
                    print(f"Error processing suggestion: {e}")
                    print("Keeping original row due to error.")
                    updated_rows.append(row)
                    comparison_rows.append([title, title, author, author, str(e)])

            sleep(1)

    # Write updated data to INPUT_CSV
    with open(INPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(updated_rows)
    print(f"Updated data written to {INPUT_CSV}.")

    # Write comparison data to a new CSV file
    comparison_headers = ["Old Title", "New Title", "Old Author", "New Author", "Error"]
    with open(COMPARE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(comparison_headers)
        writer.writerows(comparison_rows)
    print(f"Comparison data written to {COMPARE_CSV}.")


def verify_title_and_author(title, author):
    prompt = f"Verify if '{title}' by {author} is the correct title and author pairing. If it's incorrect, provide the correct author and title pairing. If it's correct, just respond with 'Correct'. If incorrect, please format accordingly: 'title: 'place correct title here', author: 'place correct author here''."
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that verifies book titles and their authors. Please be very careful and check for any minor spelling errors to either the book title or author name.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or "correct"
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return "correct"


def scrape_book_data(row):
    title, author, *_ = row
    q = (title + " " + author).replace(" ", "+").lower()
    url = f"{GR_URL}/search?q={q}&qid="

    response = make_request(url)
    if not response:
        print(f"Failed to fetch data for {title} by {author}")
        return row + [None, None]

    pattern = r"(\d+\.\d+) avg rating.*?(\d+,?\d*) ratings"
    match = re.search(pattern, response.text, re.DOTALL)

    rating, num_ratings = None, None
    if match:
        rating = float(match.group(1))
        num_ratings = int(match.group(2).replace(",", ""))

    print(
        f"For {title} by {author}, we found a rating of {rating} and {num_ratings} total ratings."
    )

    return row + [rating, num_ratings]


def scrape_gr_data():
    with open(INPUT_CSV, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    results = []
    total_books = len(rows)

    print(f"Starting to scrape data for {total_books} books...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {executor.submit(scrape_book_data, row): row for row in rows}

        for future in tqdm(
            as_completed(future_to_row), total=total_books, desc="Scraping Progress"
        ):
            row = future_to_row[future]
            try:
                data = future.result()
                results.append(data)
            except Exception as exc:
                print(f"{row[0]} generated an exception: {exc}")
                results.append(row + [None, None])
            sleep(GR_DELAY)

    print("All books processed. Writing results to file...")

    headers.extend(["rating", "num_ratings"])
    try:
        write_csv(OUTPUT_CSV, headers, results)
        print(f"Results successfully written to {OUTPUT_CSV}")
    except Exception as e:
        print(f"Error writing results to file: {e}")
        print("Attempting to write results to a backup file...")
        try:
            write_csv("backup_output.csv", headers, results)
            print("Results written to backup_output.csv")
        except Exception as e:
            print(f"Failed to write to backup file: {e}")
            print("Printing results to console as a last resort:")
            for result in results:
                print(result)

    print("Scraping process completed.")


def main():
    # Uncomment the functions you want to run
    # scrape_initial_info()
    # verify_book_info()
    scrape_gr_data()


if __name__ == "__main__":
    main()
