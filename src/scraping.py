from __future__ import annotations

from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from preprocess import normalize_text

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


@dataclass
class ScrapedDocument:
    url: str
    title: str
    text: str


def _extract_main_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in ["script", "style", "noscript", "svg", "footer", "header", "form"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = normalize_text(soup.title.string)

    main_container = soup.find("article") or soup.find("main") or soup.body or soup
    text = normalize_text(main_container.get_text(separator=" ", strip=True))
    return title, text


def scrape_url(url: str, timeout: int = 25) -> ScrapedDocument:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    if not response.encoding:
        response.encoding = response.apparent_encoding or "utf-8"
    title, text = _extract_main_text(response.text)
    return ScrapedDocument(url=url, title=title, text=text)
