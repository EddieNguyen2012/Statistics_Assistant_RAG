from logging import raiseExceptions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from astrapy import DataAPIClient
from dotenv import load_dotenv
import re
import unicodedata
from collections import Counter
import numpy as np

load_dotenv()
doc_path = '../RAG_Docs'
ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

# Static Methods

def path_validate(path: str):
    if not os.path.exists(path):
        return False
    else:
        return True


# Dynamic Methods
class DBConnector:
    def __init__(self):
        client = DataAPIClient()
        self.db = client.get_database(ENDPOINT, token=TOKEN)

    def test_db_connection(self):
        if self.db is not None:
            print(f"Connected to Astra DB: {self.db.name()}")
            print(f"Collections: {self.db.list_collection_names()}")
        else:
            print("Not connected to Astra DB")

# Helped by ChatGPT to generalize cleaning steps

def clean_encoding(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    return text

def frequent(lines, pages, freq_threshold=0.7):
    counts = Counter(l.strip().lower() for l in lines if l.strip())
    return {
        l for l, c in counts.items()
        if c / len(pages) >= freq_threshold
    }

def strip_headers_footers(docs, top_n=2, bottom_n=2, freq_threshold=0.7):
    pages = [d.page_content.splitlines() for d in docs]

    top_lines, bottom_lines = [], []
    for lines in pages:
        top_lines.extend(lines[:top_n])
        bottom_lines.extend(lines[-bottom_n:])

    bad_top = frequent(top_lines, pages=pages, freq_threshold=freq_threshold)
    bad_bottom = frequent(bottom_lines, pages=pages, freq_threshold=freq_threshold)

    for d in docs:
        lines = d.page_content.splitlines()
        lines = [
            l for l in lines
            if l.strip().lower() not in bad_top
            and l.strip().lower() not in bad_bottom
        ]
        d.page_content = "\n".join(lines)

    return docs

def normalize_whitespace(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?:\n\s*){3,}', '\n\n', text)
    text = re.sub(r'\.{4,}', '.', text) # Avoiding ... case
    return text.strip()

def preprocess_docs(docs):
    for d in docs:
        d.page_content = clean_encoding(d.page_content)

    docs = strip_headers_footers(docs)

    for d in docs:
        d.page_content = normalize_whitespace(d.page_content)

    return docs

# My work


class DocIngestion:
    def __init__(self, docs_path: str, chunk_size=100, chunk_overlap=20):
        self.docs_path = docs_path if path_validate(docs_path) else raiseExceptions
        self.chunk_size = -1
        self.chunk_overlap = -1
        self.splitter = None
        self.update_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def batch_ingest(self):
        for file in [f for f in os.listdir(self.docs_path) if not f.startswith('.')]: # Get all files name for ingestion
            res = self.individual_ingest(file)
            return res
        return None

    # Return True if the file is loaded successfully, False otherwise
    def individual_ingest(self, filename: str):
        if path_validate(os.path.join(self.docs_path, filename)):
            loader = PyPDFLoader(os.path.join(self.docs_path, filename))
            document = loader.load()
            cleaned_doc = preprocess_docs(document)
            return self.chunking(cleaned_doc)
        else:
            return None

    def chunking(self, doc):
        chunks = self.splitter.split_documents(doc)
        valid_chunks = [doc for doc in chunks if len(doc.page_content) > 50]
        print(f"Found total {len(chunks)} chunks")
        print(f"Found {len(valid_chunks)} valid (len > 50) chunks")
        print(f"Average chunk size: {np.mean([len(doc.page_content) for doc in valid_chunks])}")
        return valid_chunks

    def update_splitter(self, chunk_size=-1, chunk_overlap=-1):

        if chunk_size == -1 and chunk_overlap == -1:
            print("Nothing to update")
            return

        new_size = chunk_size if chunk_size != -1 else self.chunk_size
        new_overlap = chunk_overlap if chunk_overlap != -1 else self.chunk_overlap

        if new_size <= new_overlap:
            print(f"Error: Chunk size ({new_size}) must be greater than chunk overlap ({new_overlap}). Update aborted.")
            return

        self.chunk_size = new_size
        self.chunk_overlap = new_overlap

        try:
            self.splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " ", ""],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            print(f"Current splitter: name={self.splitter.__class__}, chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        except Exception as e:
            print(f"Unexpected error updating splitter: {e}")




