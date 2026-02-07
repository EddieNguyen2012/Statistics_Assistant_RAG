from logging import raiseExceptions
from langchain_community.document_loaders import PyPDFLoader
import os
from astrapy import DataAPIClient
from dotenv import load_dotenv

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



class DocIngestion:
    def __init__(self, path: str):
        self.path = path if path_validate(path) else raiseExceptions
        self.db = DBConnector()

    def ingest(self):
        for files in [f for f in os.listdir(self.path) if not f.startswith('.')]:
            loader = PyPDFLoader(os.path.join(self.path, files))
            document = loader.load()

