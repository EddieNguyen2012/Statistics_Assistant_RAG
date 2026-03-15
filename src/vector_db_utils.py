import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_core.documents import Document
from dotenv import load_dotenv
from src.Ingestion import DocIngestion
from pathlib import Path

parent_dir = Path(__name__).parent.resolve()
default_db_path = parent_dir.parent / 'vector_db'
default_doc_path = parent_dir.parent / 'RAG_Docs'
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Default metadata from LangChain:
#   'producer', 'creator', 'creationdate', 'author', 'category', 'comments',
#   'company', 'keywords', 'moddate', 'sourcemodified', 'subject', 'title', 'source',
#   'total_pages', 'page', 'page_label'

def extract_metadata_by_page(chunks: list[Document]):
    """
    Extracts metadata from a list of document chunks by invoking an LLM chain for each chunk.
    Enriches each chunk with a heading and summary.

    Args:
        chunks (list[Document]): A list of Document objects to be enriched.

    Returns:
        tuple: A tuple containing a list of generated chunk IDs and the list of enriched Document objects.
    """
    chunk_ids = []
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        original_meta = chunk.metadata
        new_metadata = {
            "page": original_meta.get("page", 0) + 1,
            "title": original_meta.get("title", "Testing Statistical Assumptions"),
            "subject": original_meta.get("subject", ""),
            "author": original_meta.get("author", "Unknown")
        }

        chunk_id = f"stats_book_p{chunk.metadata['page']}_c{i}"
        chunk.metadata = new_metadata
        enriched_chunks.append(chunk)
        chunk_ids.append(chunk_id)

    return chunk_ids, enriched_chunks

def connect_db(path=default_db_path):
    """
    Connects to the ChromaDB persistent client at the specified path.

    Args:
        path (Path, optional): The path to the persistent database. Defaults to default_db_path.

    Returns:
        chromadb.PersistentClient: The initialized ChromaDB client.
    """
    client = chromadb.PersistentClient(path=path)
    return client

def insert_doc(conn, ingestor, collection, file):
    """
    Populates the vector database with documents from the ingestor's directory.
    Chunks are processed individually and metadata is enriched page-by-page.

    Args:
        conn (Database): The database connection object.
        ingestor (DocIngestion): The document ingestion object containing the documents path.
        collection (str): The name of the collection where documents will be inserted.
    """

    # I am worried about in-place memory usage when batch chunking so I decided to chunk individually
    print(f"Processing {file}")
    chunks = ingestor.individual_ingest(os.path.join(ingestor.docs_path, file))
    current_page = 0
    current_chunks = []
    for i, chunk in enumerate(chunks):
        # print(f'Working on page {current_page} of {file}')
        if chunk.metadata['page'] == current_page:
            current_chunks.append(chunk)
        else:
            if len(current_chunks) > 0:
                ids, current_chunks = extract_metadata_by_page(current_chunks)
                conn.upsert_docs(ids=ids, docs=current_chunks, collection=collection)
            current_chunks = []
            current_page += 1

def populate_db(conn, ingestor, collection):
    """
    Populates the vector database with documents from the ingestor's directory.
    Chunks are processed individually and metadata is enriched page-by-page.

    Args:
        conn (Database): The database connection object.
        ingestor (DocIngestion): The document ingestion object containing the documents path.
        collection (str): The name of the collection where documents will be inserted.
    """
    files = [file for file in os.listdir(ingestor.docs_path) if not file.startswith('.')]

    # I am worried about in-place memory usage when batch chunking so I decided to chunk individually
    for file in files:
        print(f"Processing {file}")
        chunks = ingestor.individual_ingest(os.path.join(ingestor.docs_path, file))
        current_page = 0
        current_chunks = []
        for i, chunk in enumerate(chunks):
            # print(f'Working on page {current_page} of {file}')
            if chunk.metadata['page'] == current_page:
                current_chunks.append(chunk)
            else:
                if len(current_chunks) > 0:
                    ids, current_chunks = extract_metadata_by_page(current_chunks)
                    conn.insert_doc(ids=ids, docs=current_chunks, collection=collection)
                current_chunks = []
                current_page += 1

class Database:
    """
    A class to interact with the ChromaDB vector database.
    Provides methods for collection management, document insertion, retrieval, and updates.
    """
    def __init__(self, path=default_db_path):
        """
        Initializes the Database instance by connecting to the specified path.

        Args:
            path (Path, optional): The path to the persistent database. Defaults to default_db_path.
        """
        self.client = connect_db(path=path)

    def inspect(self):
        """
        Prints information about all collections in the database, including document counts and a preview of items.
        """
        collection_list = self.client.list_collections()
        for collection in collection_list:
            print(f'-----------Collection: {collection.name}\n\n '
                  f'Doc counts: {collection.count()}\n\n '
                  f'First 10 items: {collection.peek()}\n\n'
                  f'-----------')

    def get_collection(self, collection_name):
        """
        Retrieves an existing collection from the database by name.

        Args:
            collection_name (str): The name of the collection to retrieve.

        Returns:
            chromadb.Collection: The retrieved collection object.
        """
        return self.client.get_collection(name=collection_name)

    def create_collection(self, collection_name, embedding_function=SentenceTransformerEmbeddingFunction(token=hf_token)):
        """
        Creates a new collection in the database with the specified embedding function.

        Args:
            collection_name (str): The name of the collection to create.
            embedding_function (callable, optional): The function used to generate embeddings. 
                                                     Defaults to SentenceTransformerEmbeddingFunction.

        Returns:
            chromadb.Collection: The newly created collection object.
        """
        return self.client.create_collection(name=collection_name, embedding_function=embedding_function)

    def insert_doc(self, ids, docs: list[Document], collection):
        """
        Inserts document chunks into a specified collection.

        Args:
            ids (list[str]): A list of unique identifiers for the document chunks.
            docs (list[Document]): A list of Document objects containing page content and metadata.
            collection (str): The name of the collection where documents will be inserted.
        """
        collection = self.get_collection(collection)
        collection.add(ids=ids, documents=[doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])

    def insert_new_book(self, file):
        insert_doc(
            conn=self,
            ingestor=DocIngestion(docs_path=default_doc_path, chunk_size=200, chunk_overlap=100),
            collection='Stat-RAG-200-100',
            file=file
        )


    def get_docs_by_ids(self, doc_ids, collection, embedding: bool):
        """
        Retrieves documents from a collection by their unique identifiers.

        Args:
            doc_ids (list[str]): A list of document IDs to retrieve.
            collection (str): The name of the collection to search in.
            embedding (bool): Whether to include embeddings in the retrieved documents.

        Returns:
            dict: The retrieval results from ChromaDB.
        """
        collection = self.get_collection(collection)
        return collection.get(ids=doc_ids, include=["embeddings"]) if embedding else collection.get(ids=doc_ids)

    def get_docs_by_text(self, text, collection, n_results: int):
        """
        Queries a collection for documents similar to the provided input text.

        Args:
            text (str): The input text to query for.
            collection (str): The name of the collection to search in.
            n_results (int): The number of top results to return.

        Returns:
            dict: The query results from ChromaDB.
        """
        collection = self.get_collection(collection)
        return collection.query(query_texts=text, n_results=n_results)

    def update_docs(self, new_ids, new_docs, new_metadatas, collection):
        """
        Updates existing documents in a collection.

        Args:
            new_ids (list[str]): The IDs of the documents to update.
            new_docs (list[str]): The new page content for the documents.
            new_metadatas (list[dict]): The new metadata for the documents.
            collection (str): The name of the collection to update.
        """
        collection = self.get_collection(collection)
        collection.update(ids=new_ids, documents=new_docs, metadatas=new_metadatas)

    def upsert_docs(self, ids, docs: list[Document], collection):
        """
        Inserts or updates document chunks in a specified collection.

        Args:
            ids (list[str]): A list of unique identifiers for the document chunks.
            docs (list[Document]): A list of Document objects to upsert.
            collection (str): The name of the collection.
        """
        collection = self.get_collection(collection)
        collection.upsert(ids=ids, documents=[doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])

    def delete_docs_by_id(self, ids, collection):
        """
        Deletes documents from a collection by their unique identifiers.

        Args:
            ids (list[str]): A list of document IDs to delete.
            collection (str): The name of the collection.
        """
        collection = self.get_collection(collection)
        collection.delete(ids=ids)

    def delete_docs_by_text(self, metadata, collection):
        """
        Deletes documents from a collection based on metadata criteria.

        Args:
            metadata (dict): The metadata filters used for deletion.
            collection (str): The name of the collection.
        """
        collection = self.get_collection(collection)
        collection.delete(where=metadata)

    def update_metadatas(self, ids, new_metadata, collection):
        """
        Updates metadata for existing documents in a collection.

        Args:
            ids (list[str]): The IDs of the documents whose metadata will be updated.
            new_metadata (list[dict]): The new metadata list.
            collection (str): The name of the collection.
        """
        collection = self.get_collection(collection)
        collection.update(ids=ids, documents=new_metadata)

    def reset_db(self):
        """
        Resets the database by deleting all collections and data.
        """
        self.client.reset()

    def get_or_init_collection(self):
        """
        Retrieves the default collection or initializes it if it doesn't exist.
        If missing, it creates the collection and populates it with documents.

        Returns:
            Database: Returns the current instance of the Database object.
        """
        try:
            self.client.get_collection('Stat-RAG-200-100')
            return self
        except Exception as e:
            print(f"Error: No collection found in the database. Populating the database...")
            self.client.create_collection('Stat-RAG-200-100',
                                          embedding_function=SentenceTransformerEmbeddingFunction(token=hf_token))
            populate_db(conn=self, ingestor=DocIngestion(docs_path=default_doc_path, chunk_size=200, chunk_overlap=100),
                        collection='Stat-RAG-200-100')
            return self


    def adjust_ingestor(self, chunk_size=-1, chunk_overlap=-1):
        """
        Adjusts the ingestor parameters by creating a new collection with specified chunk size and overlap,
        then populates it.

        Args:
            chunk_size (int, optional): The new chunk size. Defaults to -1 (no change).
            chunk_overlap (int, optional): The new chunk overlap. Defaults to -1 (no change).
        """
        if chunk_size != -1 and chunk_overlap != -1:
            collection_name = f'Stat-RAG-{chunk_size}-{chunk_overlap}'
            self.client.create_collection(collection_name,
                                     embedding_function=SentenceTransformerEmbeddingFunction(token=hf_token))
            populate_db(self, ingestor=DocIngestion(docs_path=default_doc_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap), collection=collection_name)

    def get_retriever(self, collection='Stat-RAG-200-100'):
        """
        Initializes and returns an EnsembleRetriever combining Chroma and BM25 retrievers.
        The Chroma retriever uses HuggingFace embeddings.

        Args:
            collection (str, optional): The collection name to retrieve from. Defaults to 'Stat-RAG-200-100'.

        Returns:
            EnsembleRetriever: The configured ensemble retriever object.
        """
        try:
            self.get_collection(collection)
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

            # Initialize the embeddings object
            embeddings = HuggingFaceEmbeddings(model_name=model_name)

            vector_stores = Chroma(
                persist_directory=str(default_db_path),
                embedding_function=embeddings,
                collection_name=collection,

            )

            chroma_retriever = vector_stores.as_retriever(
                search_type='similarity',
                search_kwargs={"k": 5}
            )

            data = vector_stores.get(include=['documents', 'metadatas'])
            all_docs = [
                Document(page_content=text, metadata=metadata) for text, metadata in zip(data['documents'], data['metadatas'])
            ]

            bm25_retriever = BM25Retriever.from_documents(documents=all_docs)
            bm25_retriever.k = 5

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, chroma_retriever],
                weights=[0.7, 0.3]
            )

            return ensemble_retriever
        except Exception as e:
            print(f"Error: {e}")
            return None

