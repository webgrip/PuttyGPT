from langchain.retrievers import TimeWeightedVectorStoreRetriever
import faiss

from datetime import datetime, timedelta
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores import Weaviate


schema = client.schema.get()


print(schema)
vectorstore = Weaviate(client, "Paragraph", "content")

retriever = WeaviateHybridSearchRetriever(
    client, index_name="LangChain", text_key="text"
)


# Long term memory


# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.0000000000000000000000001, k=1) 








yesterday = datetime.now() - timedelta(days=1)
retriever.add_documents([Document(page_content="hello world", metadata={"last_accessed_at": yesterday})])
retriever.add_documents([Document(page_content="hello foo")])

['5c9f7c06-c9eb-45f2-aea5-efce5fb9f2bd']


# "Hello World" is returned first because it is most salient, and the decay rate is close to 0., meaning it's still recent enough
retriever.get_relevant_documents("hello world")



[Document(page_content='hello world', metadata={'last_accessed_at': datetime.datetime(2023, 4, 16, 22, 9, 1, 966261), 'created_at': datetime.datetime(2023, 4, 16, 22, 9, 0, 374683), 'buffer_idx': 0})]





semantic_similarity + (1.0 - decay_rate) ** hours_passed




# Define your embedding model
embeddings_model = OpenAIEmbeddings()
# Initialize the vectorstore as empty
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.33, k=1) 
yesterday = datetime.now() - timedelta(days=1)
retriever.add_documents([Document(page_content="hello world", metadata={"last_accessed_at": yesterday})])
retriever.add_documents([Document(page_content="hello foo")])
# "Hello Foo" is returned first because "hello world" is mostly forgotten
retriever.get_relevant_documents("hello world")

[Document(page_content='hello foo', metadata={'last_accessed_at': datetime.datetime(2023, 4, 16, 22, 9, 2, 494798), 'created_at': datetime.datetime(2023, 4, 16, 22, 9, 2, 178722), 'buffer_idx': 1})]