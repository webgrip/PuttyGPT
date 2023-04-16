import os
import tiktoken
from dotenv import load_dotenv

load_dotenv()

APIFY_API_KEY = os.getenv("APIFY_API_KEY")
SEARXNG_URL = os.getenv("SEARXNG_URL")
WOLFRAM_ALPHA_APPID = os.getenv("WOLFRAM_ALPHA_APPID")


##### To get the tokeniser corresponding to a specific model in the OpenAI API:#####
# enc = tiktoken.encoding_for_model("gpt-4")

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

