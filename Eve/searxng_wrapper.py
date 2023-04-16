from typing import List, Dict
import requests
from config import SEARXNG_URL
import json
from langchain.utilities import SearxSearchWrapper


class SearxNGWrapper:
    def __init__(self, base_url: str = SEARXNG_URL):
        self.base_url = base_url

    def run(self, query: str, language: str = 'en', categories: List[str] = None, limit: int = 10) -> List[Dict[str, str]]:
        url = f"{self.base_url}/search"

        params = {
            "q": query,
            "language": language,
            "format": "json",
            "pageno": 1,
            "time_range": None,
            "categories": ",".join(categories) if categories else None,
            "engine": None,
            "safesearch": None,
            "layout": None,
            "count": limit
        }

        search = SearxSearchWrapper(searx_host=SEARXNG_URL)

        return search.results(query, num_results = 20)