"""
Google News + BeautifulSoup crawler for FinBERT sentiment extraction.

⚠️ OPERATIONAL STATUS (2024+):
    Google has tightened anti-bot measures since late 2023. The fetch_urls()
    function below relies on parsing CSS classes (`egMi0 kCrYT`) that are
    rotated/obfuscated by Google. This module is preserved for methodological
    transparency and reproducibility of the original thesis (Q3 2023 data).

    For new sentiment extraction work, prefer:
      - SEC EDGAR 10-Q parsing (free, structured, immutable URLs)
      - Bloomberg Terminal news export (paid, reliable)
      - NewsAPI.org / GDELT (paid, dedicated APIs)

Original author: P. Athouli (Columbia QMSS course material, 2019)
Adapted by:      Jean Treves (2023 thesis, refactored 2024)
License:         MIT (with attribution)
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Constants
DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT: int = 10
SCRAPER_RETRY_SLEEP: int = 5


def scrape_article(url: str) -> str:
    """
    Extract paragraph text from a single URL.

    Parameters
    ----------
    url : str
        Target URL to scrape.

    Returns
    -------
    str
        Concatenated text from all <p> tags, with whitespace normalized.
        Returns empty string on connection failure.

    Notes
    -----
    Cleans HTML entities (xa0) and collapses non-word characters to single spaces.
    """
    try:
        response: requests.Response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs: list[str] = [p.text for p in soup.findAll("p")]
        text: str = " ".join(paragraphs)
        text = re.sub(r"\W+", " ", re.sub("xa0", " ", text))
        return text.strip()

    except requests.exceptions.RequestException as exc:
        logger.warning("Connection failed for %s: %s — sleeping %ds", url, exc, SCRAPER_RETRY_SLEEP)
        time.sleep(SCRAPER_RETRY_SLEEP)
        return ""


def fetch_google_urls(query: str, count: int = 10) -> list[str]:
    """
    Fetch top-N Google search result URLs for a given query.

    ⚠️ This function relies on Google's HTML structure (div class `egMi0 kCrYT`)
    which changes frequently. Expected to return [] for most queries in 2024+.

    Parameters
    ----------
    query : str
        Search query (e.g., "Tesla third quarter 2023 results").
    count : int, default=10
        Number of results requested via &num= parameter.

    Returns
    -------
    list[str]
        Extracted result URLs. Empty list if Google returns no matches
        or if the HTML structure has changed.
    """
    headers: dict[str, str] = {"User-Agent": DEFAULT_USER_AGENT}
    encoded_query: str = "+".join(query.split())
    google_url: str = f"https://www.google.com/search?q={encoded_query}&num={count}"

    logger.info("Fetching: %s", google_url)
    response: requests.Response = requests.get(google_url, headers=headers, timeout=REQUEST_TIMEOUT)
    soup = BeautifulSoup(response.text, "html.parser")

    # Legacy CSS selector (Q3 2023 vintage) — likely obsolete
    result_divs = soup.find_all("div", attrs={"class": "egMi0 kCrYT"})

    links: list[str] = []
    for result_div in result_divs:
        try:
            raw_href: Optional[str] = result_div.a.get("href")
            if raw_href is None:
                continue
            # Google wraps URLs in /url?q=<real_url>&ved=...
            after_url: str = re.split("url=", raw_href)[1]
            clean_url: str = re.split("&ved", after_url)[0]
            if clean_url and clean_url not in links:
                links.append(clean_url)
        except (AttributeError, IndexError) as exc:
            logger.debug("Could not parse result div: %s", exc)
            continue

    if not links:
        logger.warning(
            "No URLs extracted for query=%r. Google HTML structure may have changed.",
            query,
        )
    return links


def crawl_queries(queries: list[str], count: int = 10) -> pd.DataFrame:
    """
    Run a multi-query crawl and return aggregated article texts.

    Parameters
    ----------
    queries : list[str]
        List of search queries to execute sequentially.
    count : int, default=10
        Number of results to fetch per query.

    Returns
    -------
    pd.DataFrame
        Columns: ['body', 'label'] where body is scraped article text
        and label is the originating query with spaces replaced by underscores.

    Examples
    --------
    >>> df = crawl_queries(["Tesla Q3 2023 results"], count=10)
    >>> df.columns.tolist()
    ['body', 'label']
    """
    rows: list[dict[str, str]] = []

    for query in queries:
        urls: list[str] = fetch_google_urls(query, count)
        label: str = re.sub(" ", "_", query)

        for url in urls:
            article_text: str = scrape_article(url)
            if article_text:
                rows.append({"body": article_text, "label": label})
                logger.info("Scraped: %s (%d chars)", url, len(article_text))

    return pd.DataFrame(rows)


# Backwards-compatible alias (original thesis API)
write_crawl_results = crawl_queries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.warning(
        "This crawler is preserved for reference. Expect empty results in 2024+ "
        "due to Google anti-bot defences. See module docstring for alternatives."
    )
