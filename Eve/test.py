#from agents.AutoGPT import AutoGPT
#from agents.BabyAGI import BabyAGI
from agents.GenerativeAgent import GenerativeAgent

from langchain.chat_models import ChatOpenAI

from langchain.vectorstores import Weaviate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings

import weaviate

import math
import os

from agents.memory import GenerativeAgentMemory

USER_NAME = "Person A" # The name you want to use when interviewing the agent.
LLM = ChatOpenAI(max_tokens=1500) # Can be any LLM you want.

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

def create_new_memory_retriever():
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
    embeddings_model = OpenAIEmbeddings()

    from langchain.vectorstores import FAISS
    from langchain.vectorstores import Chroma
    
    

    #embedding_size = 1536
    #index = faiss.IndexFlatL2(embedding_size)
    #vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)

    vectorstore = Weaviate(client, "Paragraph", "content", embedding=embeddings_model.embed_query)

    vectorstore = Chroma(embedding_function=embeddings_model.embed_query)

    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)    

ryan = GenerativeAgentMemory(
    name="Ryan", 
    age=28,
    traits="loyal, experimental, hopeful, smart, world class programmer", # You can add more persistent traits here 
    status="Executing the task", # When connected to a virtual world, we can have the characters update their status
    memory_retriever=create_new_memory_retriever(),
    verbose=True,
    llm=LLM,
    daily_summaries = [
        "Just woke up, ready and eager to start working"
    ],
    reflection_threshold = 8, # we will give this a relatively low number to show how reflection works
)


# The current "Summary" of a character can't be made because the agent hasn't made
# any observations yet.
print(ryan.get_summary())

# We can give the character memories directly
ryanMemories = [
    "Ryan remembers his first job, where he was working on a new concept for webshops at a startup",
    "Ryan feels energetic today",
    "Ryan knows a lot about writing clean code. Clean architecture.",
    "Ryan is a world class programmer",
    "Ryan is completely devoted to the goal of creating AGI, and every step he takes is one on that path",
]
for memory in ryanMemories:
    ryan.add_memory(memory)
# Now that Tommie has 'memories', their self-summary is more descriptive, though still rudimentary.
# We will see how this summary updates after more observations to create a more rich description.
print(ryan.get_summary(force_refresh=True))