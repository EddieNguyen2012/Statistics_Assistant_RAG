from logging import raiseExceptions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import re
import unicodedata
from collections import Counter
import numpy as np
from pathlib import Path

parent_dir = Path(__name__).parent.resolve()
default_db_path = parent_dir.parent / 'vector_db'
doc_path = parent_dir.parent / 'RAG_Docs'
load_dotenv()


# Static Methods

def path_validate(path: str):
    """
    Validates if a given path exists.

    Args:
        path (str): The path to validate.

    Returns:
        bool: True if the path exists, False otherwise.
    """
    if not os.path.exists(path):
        return False
    else:
        return True

# Helped by ChatGPT to generalize cleaning steps

def clean_encoding(text):
    """
    Normalizes text encoding to NFKC and removes non-printable control characters.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    return text

def frequent(lines, pages, freq_threshold=0.7):
    """
    Identifies lines that appear frequently across multiple pages.

    Args:
        lines (list[str]): A list of lines to analyze.
        pages (list[list[str]]): A list of pages, where each page is a list of lines.
        freq_threshold (float, optional): The frequency threshold (0 to 1). Defaults to 0.7.

    Returns:
        set: A set of frequent lines.
    """
    counts = Counter(l.strip().lower() for l in lines if l.strip())
    return {
        l for l, c in counts.items()
        if c / len(pages) >= freq_threshold
    }

def strip_headers_footers(docs, top_n=2, bottom_n=2, freq_threshold=0.7):
    """
    Removes headers and footers from documents based on frequent line occurrences at the top and bottom of pages.

    Args:
        docs (list[Document]): A list of Document objects.
        top_n (int, optional): Number of lines at the top to consider as potential headers. Defaults to 2.
        bottom_n (int, optional): Number of lines at the bottom to consider as potential footers. Defaults to 2.
        freq_threshold (float, optional): Frequency threshold for identifying headers/footers. Defaults to 0.7.

    Returns:
        list[Document]: The documents with headers and footers removed.
    """
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
    """
    Normalizes whitespace in text by collapsing multiple spaces, tabs, and newlines.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: The text with normalized whitespace.
    """
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?:\n\s*){3,}', '\n\n', text)
    text = re.sub(r'\.{4,}', '.', text) # Avoiding ... case
    return text.strip()

def preprocess_docs(docs):
    """
    Performs a full preprocessing pipeline on a list of documents:
    1. Clean encoding.
    2. Strip headers and footers.
    3. Normalize whitespace.

    Args:
        docs (list[Document]): A list of Document objects.

    Returns:
        list[Document]: The preprocessed documents.
    """
    for d in docs:
        d.page_content = clean_encoding(d.page_content)

    docs = strip_headers_footers(docs)

    for d in docs:
        d.page_content = normalize_whitespace(d.page_content)

    return docs

# My work




class DocIngestion:
    """
    Handles document ingestion, including loading PDFs, preprocessing, and chunking.
    """
    def __init__(self, docs_path: str | Path, chunk_size=100, chunk_overlap=20):
        """
        Initializes the DocIngestion instance.

        Args:
            docs_path (str | Path): The directory path containing documents.
            chunk_size (int, optional): The size of each text chunk. Defaults to 100.
            chunk_overlap (int, optional): The overlap between consecutive chunks. Defaults to 20.
        """
        self.docs_path = docs_path if path_validate(docs_path) else raiseExceptions
        self.chunk_size = -1
        self.chunk_overlap = -1
        self.splitter = None
        self.update_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.page_map = None

    def batch_ingest(self):
        """
        Ingests all non-hidden files in the docs directory.

        Returns:
            list: A list of chunks retrieved from all files.
        """
        chunks = []
        for file in [f for f in os.listdir(self.docs_path) if not f.startswith('.')]: # Get all files name for ingestion
            res = self.individual_ingest(file)
            chunks.append(res)
        return chunks

    # Return True if the file is loaded successfully, False otherwise
    def individual_ingest(self, filename: str):
        """
        Ingests a single file: loads, preprocesses, and chunks the document.

        Args:
            filename (str): The name of the file to ingest.

        Returns:
            list[Document] | None: A list of document chunks if successful, None otherwise.
        """
        if path_validate(os.path.join(self.docs_path, filename)):
            loader = PyPDFLoader(os.path.join(self.docs_path, filename))
            document = loader.load()
            cleaned_doc = preprocess_docs(document)
            return self.chunking(cleaned_doc)
        else:
            return None

    def chunking(self, doc):
        """
        Splits a document into chunks using the configured text splitter and filters out small chunks.

        Args:
            doc (list[Document]): The document (list of pages) to chunk.

        Returns:
            list[Document]: A list of valid document chunks.
        """
        chunks = self.splitter.split_documents(doc)
        valid_chunks = [doc for doc in chunks if len(doc.page_content) > 50]
        print(f"Found total {len(chunks)} chunks")
        print(f"Found {len(valid_chunks)} valid (len > 50) chunks")
        print(f"Average chunk size: {np.mean([len(doc.page_content) for doc in valid_chunks])}")
        return valid_chunks

    def update_splitter(self, chunk_size=-1, chunk_overlap=-1):
        """
        Updates the RecursiveCharacterTextSplitter with new chunk size and overlap values.

        Args:
            chunk_size (int, optional): The new chunk size. Defaults to -1 (no change).
            chunk_overlap (int, optional): The new chunk overlap. Defaults to -1 (no change).
        """

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





