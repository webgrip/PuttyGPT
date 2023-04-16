from typing import List
from haystack.document_store.weaviate import WeaviateDocumentStore
from haystack.retriever.base import BaseRetriever
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.retriever.dense import DensePassageRetriever
from haystack.retriever.dense import EmbeddingRetriever


class WeaviateHybridSearchRetriever(BaseRetriever):
    def __init__(self, document_store: WeaviateDocumentStore, query_embedding_model: str = None,
                 passage_embedding_model: str = None, use_gpu: bool = True, embed_title: bool = True,
                 max_length: int = 128, batch_size: int = 16, remove_sep_token_from_doc: bool = True):
        self.document_store = document_store
        self.elastic_retriever = ElasticsearchRetriever(document_store=document_store)
        self.dpr_retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model=query_embedding_model,
            passage_embedding_model=passage_embedding_model,
            use_gpu=use_gpu,
            embed_title=embed_title,
            max_length=max_length,
            batch_size=batch_size,
            remove_sep_token_from_doc=remove_sep_token_from_doc,
        )
        self.retriever = EmbeddingRetriever(retrievers=[self.elastic_retriever, self.dpr_retriever])

    def retrieve(self, query: str, top_k: int = 10) -> List:
        return self.retriever.retrieve(query=query, top_k=top_k)