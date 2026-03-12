import os
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from src.pipeline import populate_db
from src.Ingestion import DocIngestion
from chromadb.config import Settings

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
#

def connect_db(path='./vector_db'):
    client = chromadb.PersistentClient(path=path)
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

    def get_or_init_collection(self):
        try:
            self.client.get_collection('Stat-RAG-200-100')
            return self
        except Exception as e:
            print(f"Error: No collection found in the database. Populating the database...")
            self.client.create_collection('Stat-RAG-200-100',
                                          embedding_function=SentenceTransformerEmbeddingFunction(token=hf_token))
            populate_db(conn=self, ingestor=DocIngestion(docs_path='../RAG_Docs', chunk_size=200, chunk_overlap=100),
                        collection='Stat-RAG-200-100')
            return self


    def adjust_ingestor(self, chunk_size=-1, chunk_overlap=-1):
        if chunk_size != -1 and chunk_overlap != -1:
            collection_name = f'Stat-RAG-{chunk_size}-{chunk_overlap}'
            self.client.create_collection(collection_name,
                                     embedding_function=SentenceTransformerEmbeddingFunction(token=hf_token))
            populate_db(self, ingestor=DocIngestion(docs_path='../RAG_Docs', chunk_size=chunk_size, chunk_overlap=chunk_overlap), collection=collection_name)

    def get_retriever(self, collection='Stat-RAG-200-100'):

        try:
            self.get_collection(collection)
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

            # Initialize the embeddings object
            embeddings = HuggingFaceEmbeddings(model_name=model_name)

            vector_stores = Chroma(
                persist_directory='./vector_db',
                embedding_function=embeddings,
                collection_name=collection,

            )

            chroma_retriever = vector_stores.as_retriever(
                search_type='similarity',
                search_kwargs={"k": 3}
            )

            data = vector_stores.get(include=['documents', 'metadatas'])
            all_docs = [
                Document(page_content=text, metadata=metadata) for text, metadata in zip(data['documents'], data['metadatas'])
            ]

            bm25_retriever = BM25Retriever.from_documents(documents=all_docs)
            bm25_retriever.k = 3

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, chroma_retriever],
                weights=[0.2, 0.8]
            )

            return ensemble_retriever
        except Exception as e:
            print(f"Error: {e}")

