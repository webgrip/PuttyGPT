from babyagi import OpenAIConnector

class ContextAgent:
    def __init__(self,):
        self

    def run(self, query: str, top_results_num: int):
        """
        Retrieves context for a given query from an index of tasks.
        Args:
            query (str): The query or objective for retrieving context.
            top_results_num (int): The number of top results to retrieve.
        Returns:
            list: A list of tasks as context for the given query, sorted by relevance.
        """
        query_embedding = OpenAIConnector.get_ada_embedding(query)
        results = index.query(query_embedding, top_k=top_results_num, include_metadata=True, namespace=OBJECTIVE)
        # print("***** RESULTS *****")
        # print(results)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return [(str(item.metadata["task"])) for item in sorted_results]