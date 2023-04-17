import openai
OPENAI_API_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.7


class OpenAIConnector:
    def __init__(
        self,
        model: str = OPENAI_API_MODEL,
        temperature: float = OPENAI_TEMPERATURE,
        max_tokens: int = 100,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_ada_embedding(self, text: str) -> list:
        text = text.replace("\n", " ")
        response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
        return response["data"][0]["embedding"]

    def openai_call(self, prompt: str) -> str:
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
            except openai.error.RateLimitError:
                time.sleep(10)
            else:
                break