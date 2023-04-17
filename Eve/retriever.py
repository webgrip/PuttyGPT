from typing import Optional, List, Dict
import weaviate
from langchain.schema import BaseRetriever
#import DPR


class Document:
    def __init__(self, page_content: str, metadata: Dict[str, str] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class WeaviateRetrieverWrapper:
    def __init__(self, weaviate_client: weaviate.Client):
        self.weaviate_client = weaviate_client

    def retrieve(self, query: str, filters: Optional[Dict] = None, top_k: int = 10, **kwargs) -> List:
        if filters:
            raise NotImplementedError("Filtering is not supported in Weaviate retriever.")
        response = self.weaviate_client.query(data_object={
            "meta": {
                "query": {
                    "must": [{
                        "name": "search",
                        "value": query
                    }],
                    "should": [],
                    "not": []
                }
            },
            "fields": ["uuid", "text"]
        }, vectorize=True, vectorize_weights=None, vectorize_limit=top_k)
        documents = []
        for result in response.get("data", []):
            document = Document(page_content=result["text"], metadata={"uuid": result["uuid"]})
            documents.append(document)
        return documents


class WeaviateHybridSearchRetrieverWrapper(BaseRetriever):
    def __init__(self,
        client: weaviate.Client,
        index_name: str,
        text_key: str,
        alpha: float = 0.5,
        k: int = 4,
        attributes: Optional[List[str]] = None,
    ):
        self.weaviate_client = client

    def retrieve(self, query: str, filters: Optional[Dict] = None, top_k: int = 10, **kwargs) -> List:
        if filters:
            raise NotImplementedError("Filtering is not supported in Weaviate retriever.")
        documents = self.weaviate_retrieve(query=query, top_k=top_k)
        if len(documents) < top_k:
            passages = self.dpr.retrieve(query=query, top_k=top_k - len(documents))
            for passage in passages:
                document = Document(page_content=passage["passage_text"], metadata={"uuid": passage["document_id"]})
                documents.append(document)
        return documents

    def weaviate_retrieve(self, query: str, top_k: int):
        response = self.weaviate_client.query(data_object={
            "meta": {
                "query": {
                    "must": [{
                        "name": "search",
                        "value": query
                    }],
                    "should": [],
                    "not": []
                }
            },
            "fields": ["uuid", "text"]
        }, vectorize=True, vectorize_weights=None, vectorize_limit=top_k)
        documents = []
        for result in response.get("data", []):
            document = Document(page_content=result["text"], metadata={"uuid": result["uuid"]})
            documents.append(document)
        return documents