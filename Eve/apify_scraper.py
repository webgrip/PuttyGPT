from apify_client import ApifyClient
from config import APIFY_API_KEY

class ApifyScraper:
    def __init__(self):
        self.client = ApifyClient(APIFY_API_KEY)

    def scrape(self):
        # Customize this call with your Apify actor and input
        return self.client.call()