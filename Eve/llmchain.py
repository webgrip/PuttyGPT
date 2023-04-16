from typing import List
from haystack.retriever.base import BaseRetriever
from langchain.llms import OpenAI
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.callbacks.base import CallbackManager
from langchain.conversation_buffer import ConversationBufferMemory

class LLMChain:
    def __init__(self, openai: OpenAI, retriever: BaseRetriever, conversation_buffer_memory: ConversationBufferMemory):
        self.openai = openai
        self.retriever = retriever
        self.conversation_buffer_memory = conversation_buffer_memory
        self.hybrid_retriever = WeaviateHybridSearchRetriever(document_store=self.retriever.document_store)
        self.callback_manager = CallbackManager([])

    async def generate_response(self, query: str, top_k_reader: int = 3, top_k_retriever: int = 5) -> str:
        result = await self.openai.generate_text(prompt=query, max_tokens=100, top_p=0.95, temperature=0.7)
        response = result["choices"][0]["text"].strip()

        return response

    async def generate_response_with_context(self, query: str, context: str, top_k_reader: int = 3, top_k_retriever: int = 5) -> str:
        prompt = f"{context.strip()} {query.strip()}"
        result = await self.openai.generate_text(prompt=prompt, max_tokens=100, top_p=0.95, temperature=0.7)
        response = result["choices"][0]["text"].strip()

        return response

    async def generate_response_from_retriever(self, query: str, top_k_reader: int = 3, top_k_retriever: int = 5) -> str:
        retriever_results = await self.retriever.retrieve(query=query, top_k=top_k_retriever)
        documents = [r.document for r in retriever_results]
        responses = await self.openai.generate_responses_from_documents(
            documents=documents,
            prompt=query
        )

        if len(responses) > 0:
            return responses[0]
        else:
            return ""
        
    async def generate_multiple_responses_from_documents(self, documents: List[dict], query: str, top_k_reader: int = 3) -> List[str]:
        responses = await self.openai.generate_responses_from_documents(
            documents=documents,
            prompt=query
        )

        return responses