import math
from array import array
from langchain.llms.base import BaseLLM

from .memory import GenerativeAgentMemory

from langchain.vectorstores import Weaviate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever


import os

import weaviate 


WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "")
WEAVIATE_VECTORIZER = os.getenv("WEAVIATE_VECTORIZER", "")

def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)

def create_new_memory_retriever_default():
    """Create a new vector store retriever unique to the agent."""

    client = weaviate.Client(
        url=WEAVIATE_HOST,
        additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
        # auth_client_secret: Optional[AuthCredentials] = None,
        # timeout_config: Union[Tuple[Real, Real], Real] = (10, 60),
        # proxies: Union[dict, str, None] = None,
        # trust_env: bool = False,
        # additional_headers: Optional[dict] = None,
        # startup_period: Optional[int] = 5,
        # embedded_options=[],
    )

    embeddings_model = OpenAIEmbeddings(
        #deployment="your-embeddings-deployment-name",
        model="text-embedding-ada-002"
    )

    vectorstore = Weaviate(client, "Paragraph", "content", embedding=embeddings_model, relevance_score_fn=relevance_score_fn)
    
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)

class AutonomousAgent():

    def make(
        self,
        name: str,
        age: int,
        traits: str,
        status: str,
        llm: BaseLLM,
        daily_summaries: array,
        reflection_threshold: int = 8,
        memory_retriever: BaseRetriever = create_new_memory_retriever_default(),
        verbose: bool = False
    )->GenerativeAgentMemory:

        return GenerativeAgentMemory( # TODO current_plan
            name=name,
            age=age,
            traits=traits,
            status=status,
            reflection_threshold=reflection_threshold,
            memory_retriever=memory_retriever,
            llm=llm,
            daily_summaries=daily_summaries,
            verbose=verbose,
        )
