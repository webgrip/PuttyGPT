from babyagi import OpenAIConnector

class ContextAgent:
    def __init__(self, context_storage):
        self.context_storage = context_storage

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

        results = self.context_storage.query(query, ['task'], top_results_num, OBJECTIVE) # TODO OBJECTIVE
        
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        return [(str(item.metadata["task"])) for item in sorted_results]