from src.Ingestion import DocIngestion
from src.vector_db_utils import Database
import os
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()



## From Gemini for summarizing page
# 1. Define the structure you want the LLM to return
class ChunkEnrichment(BaseModel):
    heading: str = Field(description="The specific section or chapter title this text belongs to.")
    summary: str = Field(description="A 1-sentence summary of the statistical concepts discussed.")

# 2. Initialize Ollama
# We set temperature to 0 for consistent, factual metadata extraction
llm = ChatOllama(model="llama3.1", format="json", temperature=0)
parser = JsonOutputParser(pydantic_object=ChunkEnrichment)

# 3. Create the Enrichment Chain
prompt = ChatPromptTemplate.from_template(
    "You are a statistical assistant. Analyze the following text chunk from a book.\n"
    "Extract a concise heading and a 1-sentence summary.\n"
    "{format_instructions}\n"
    "Text: {context}"
)
chain = llm | parser | prompt
##

# Default metadata from LangChain:
#   'producer', 'creator', 'creationdate', 'author', 'category', 'comments',
#   'company', 'keywords', 'moddate', 'sourcemodified', 'subject', 'title', 'source',
#   'total_pages', 'page', 'page_label'

def extract_metadata_by_page(chunks: list[Document]):
    chunk_ids = []
    enriched_chunks = []
    metadata = chunks[0].metadata
    print(f"Extracting topics from page {metadata['page']}/{metadata['total_pages']}.")

    for i, chunk in enumerate(chunks):
        original_meta = chunk.metadata
        try:
            prediction = chain.invoke({
                "context": chunk.page_content[:1000],  # Send first 1k chars to save tokens
                "format_instructions": parser.get_format_instructions()
            })
        except Exception as e:
            prediction = {"heading": "General Statistics", "summary": "Discussion on assumptions."}

        new_metadata = {
            "heading": prediction.get("heading"),
            "summary": prediction.get("summary"),
            "page": original_meta.get("page", 0) + 1,
            "title": original_meta.get("title", "Testing Statistical Assumptions"),
            "subject": original_meta.get("subject", ""),
            "author": original_meta.get("author", "Unknown")
        }

        chunk_id = f"stats_book_p{chunk.metadata['page']}_c{i}"
        chunk.metadata = new_metadata
        enriched_chunks.append(chunk)
        chunk_ids.append(chunk_id)
        if i % 10 == 0: print(f"Processed {i}/{len(chunks)} chunks...")

    return chunk_ids, enriched_chunks

if __name__=="__main__":
    db = Database()
    ingestor = DocIngestion(docs_path='../RAG_Docs', chunk_size=100, chunk_overlap=20)
    files = [file for file in os.listdir(ingestor.docs_path) if not file.startswith('.')]
    model = SentenceTransformer("all-MiniLM-L6-v2", token=os.environ["HF_TOKEN"])

    # I am worried about in-place memory usage when batch chunking so I decided to chunk individually
    for file in files:
        print(f"Processing {file}")
        chunks = ingestor.individual_ingest(os.path.join(ingestor.docs_path, file))
        current_page = 0
        current_chunks = []
        for i, chunk in enumerate(chunks):
            if chunk.metadata['page'] == current_page:
                current_chunks.append(chunk)
            else:
                if len(current_chunks) > 0:
                    ids, current_chunks = extract_metadata_by_page(current_chunks)
                    db.insert_doc(ids=ids, docs=current_chunks, collection='Stat-RAG')
                current_chunks = []
                current_page += 1
