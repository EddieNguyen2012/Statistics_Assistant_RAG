import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def connect_db(path='./vector_db'):
    client = chromadb.PersistentClient(path=path)
    if len(client.list_collections()) == 0:
        client.create_collection('Stat-RAG', embedding_function=SentenceTransformerEmbeddingFunction(token=hf_token))
    return client


class Database:
    def __init__(self, path='./vector_db'):
        self.client = connect_db(path=path)

    def inspect(self):
        collection_list = self.client.list_collections()
        for collection in collection_list:
            print(f'-----------Collection: {collection.name}\n\n '
                  f'Doc counts: {collection.count()}\n\n '
                  f'First 10 items: {collection.peek()}\n\n'
                  f'-----------')

    def get_collection(self, collection_name):
        return self.client.get_collection(name=collection_name)

    def create_collection(self, collection_name, embedding_function=SentenceTransformerEmbeddingFunction(token=hf_token)):
        return self.client.create_collection(name=collection_name, embedding_function=embedding_function)

    def insert_doc(self, ids, docs: list[Document], collection):
        collection = self.get_collection(collection)
        collection.add(ids=ids, documents=[doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])

    def get_docs_by_ids(self, doc_ids, collection, embedding: bool):
        collection = self.get_collection(collection)
        return collection.get(ids=doc_ids, include=["embeddings"]) if embedding else collection.get(ids=doc_ids)

    def get_docs_by_text(self, text, collection, n_results: int):
        collection = self.get_collection(collection)
        return collection.query(query_texts=text, n_results=n_results)

    def update_docs(self, new_ids, new_docs, new_metadatas, collection):
        collection = self.get_collection(collection)
        collection.update(ids=new_ids, documents=new_docs, metadatas=new_metadatas)

    def upsert_docs(self, ids, docs: list[Document], collection):
        collection = self.get_collection(collection)
        collection.upsert(ids=ids, documents=[doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])

    def delete_docs_by_id(self, ids, collection):
        collection = self.get_collection(collection)
        collection.delete(ids=ids)

    def delete_docs_by_text(self, metadata, collection):
        collection = self.get_collection(collection)
        collection.delete(where=metadata)

    def update_metadatas(self, ids, new_metadata, collection):
        collection = self.get_collection(collection)
        collection.update(ids=ids, documents=new_metadata)

    def reset_db(self):
        self.client.reset()
