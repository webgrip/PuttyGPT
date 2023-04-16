from haystack.document_store.weaviate import WeaviateDocumentStore

class DocumentStore:
    def __init__(self, host: str, port: int, index: str):
        self.document_store = WeaviateDocumentStore(host=host, port=port, index=index)

    def write_documents(self, documents):
        self.document_store.write_documents(documents)

    def get_document_by_id(self, document_id):
        return self.document_store.get_document_by_id(document_id)

    def get_all_documents(self):
        return self.document_store.get_all_documents()