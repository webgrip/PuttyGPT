import wikipediaapi


class WikipediaAPIWrapper:
    def __init__(self, language: str):
        self.wiki = wikipediaapi.Wikipedia(language)

    def summary(self, query: str) -> str:
        page = self.wiki.page(query)
        return page.summary if page.exists() else "Sorry, I could not find an answer to your question."