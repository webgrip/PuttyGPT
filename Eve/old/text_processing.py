from langchain.chains import load_chain
import tiktoken

class TextProcessing:
    def __init__(self):

        self.chain = [
           # { "analyze_sentiment", load_chain("chains/sentiment/chain.json")},
           # { "summarize_concice", load_chain("chains/summarize/concice/chain.json")},
            #{ "summarize_concice", LLMSummarizationCheckerChain(llm=llm, verbose=True, max_checks=2)},
        ]

    def analyze_sentiment(self, text: str) -> str:
        sentiment = self.chain.summarize_concice.run(text)
        return sentiment

    def summarize_concice(self, text: str, max_length: int = 50) -> str:
        summary = self.chain.summarize_concice.run(text)
        return summary

    #def summarize_reduce(self, text: str, max_length: int = 50) -> str:
    #    summary = self.chain.run(text, step="summarization", max_length=max_length, min_length=10)
    #    return summary

    #def summarize_refine(self, text: str, max_length: int = 50) -> str:
    #    summary = self.chain.run(text, step="summarization", max_length=max_length, min_length=10)
    #    return summary

    def count_tokens(self, text: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = len(encoding.encode(encoding))
        return tokens