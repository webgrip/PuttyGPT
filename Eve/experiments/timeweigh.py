import os
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain.vectorstores import Weaviate
import weaviate
import datetime
import faiss
from langchain.vectorstores import FAISS


client = weaviate.Client(
    url=os.getenv("WEAVIATE_HOST"),
    additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
    # auth_client_secret: Optional[AuthCredentials] = None,
    # timeout_config: Union[Tuple[Real, Real], Real] = (10, 60),
    # proxies: Union[dict, str, None] = None,
    # trust_env: bool = False,
    # additional_headers: Optional[dict] = None,
    # startup_period: Optional[int] = 5,
    # embedded_options=[],
)
client.schema.delete_all()
schema = client.schema.get()
print(schema)

vectorstore = Weaviate(client, "Paragraph", "content")

#retriever = WeaviateHybridSearchRetriever(
#    client, index_name="LangChain", text_key="text"
#)



retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.33, k=1) 

index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
now = datetime.datetime.now()
retriever.add_documents(
    [
        Document(
            page_content="hello world",
            #metadata={"last_accessed_at": now}
        ),
    ]
)
#

retriever.get_relevant_documents("hello world")

# "Hello Foo" is returned first because "hello world" is mostly forgotten
print()
