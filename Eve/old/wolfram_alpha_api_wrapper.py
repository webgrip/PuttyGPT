import wolframalpha


class WolframAlphaAPIWrapper:
    def __init__(self, app_id: str):
        self.client = wolframalpha.Client(app_id)

    def query(self, query: str) -> str:
        res = self.client.query(query)
        answer = next(res.results, None)
        return answer.text if answer else "Sorry, I could not find an answer to your question."