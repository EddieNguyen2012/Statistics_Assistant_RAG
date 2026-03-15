
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from src.vector_db_utils import Database
from pydantic import BaseModel, Field
from typing import List
from langchain_core.runnables import RunnablePassthrough

class Citation(BaseModel):
    author: str = Field(description="The author of the textbook or source.")
    title: str = Field(description="The title of the textbook or document.")
    page: str = Field(description="The specific page number or range.")
    year: int = Field(description="The publication year.")

    def to_str(self):
        return f'{self.author}, {self.title}, {self.page}, {self.year}'

class RAGResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    citations: List[Citation] = Field(description="A list of sources cited.")

    def to_str(self):
        return f'{self.answer}'


load_dotenv()

def format_docs(docs):
    """
    Formats a list of Document objects into a single string.
    Each document is formatted with its metadata (title, author, subject, page, heading) and summary.

    Args:
        docs (list[Document]): A list of Document objects to format.

    Returns:
        str: A formatted string containing the combined content and metadata of the documents.
    """
    blocks = []
    for d in docs:
        md = d.metadata or {}

        # Build a compact header line from your metadata fields
        header_parts = []
        if md.get("title"):   header_parts.append(f"Title: {md['title']}")
        if md.get("author"):  header_parts.append(f"Author: {md['author']}")
        if md.get("subject"): header_parts.append(f"Subject: {md['subject']}")
        if md.get("page") is not None: header_parts.append(f"Page: {md['page']}")
        if md.get("heading"): header_parts.append(f"Heading: {md['heading']}")

        header = " | ".join(header_parts)

        # Put summary above the excerpt (optional, but useful)
        summary = md.get("summary")
        summary_line = f"Summary: {summary}" if summary else ""

        block = "\n".join([x for x in [header, summary_line, "Excerpt:", d.page_content] if x])
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)

def pipeline(model: ChatOllama, collection='Stat-RAG-200-100') -> str:
    """
    Initializes the RAG pipeline by setting up the database, retriever, and prompt template.

    Args:
        model: The LLM model to be used in the pipeline.
        collection (str, optional): The name of the collection to use. Defaults to 'Stat-RAG-200-100'.

    Returns:
        EnsembleRetriever: The retriever configured for the pipeline.
    """

    structured_llm = model.with_structured_output(RAGResponse)

    db = Database()
    db = db.get_or_init_collection()
    ret = db.get_retriever(collection)
    llm = model

    load_dotenv()

    system_template = """
    ### Role
    You are a helpful Statistics Graduate Assistant.

    ### Instructions
    1. **Prioritize the Context:** Use the provided snippets to answer the user's question first.
    2. **Supplement if Needed:** If the context is missing specific details, says from the supplement documents, you can't answer the question.
    3. **Be Concise:** Get straight to the point but answer all the questions and requests.
    4. **Be Credible** Provide in-text citations (MLA format) of the materials you used.
    5. **Consistent** Provide in the structure: answer then citations at the end following MLA format
    ### Context
    {context}

    ### Question
    {question}
    """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{question}")
    ])

    # return ret
    chain = (
            {'context': ret | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | model
            | StrOutputParser()
    )

    return chain


def pipeline_enforced(model):
    # 2. Wrap the model to enforce JSON output
    structured_llm = model.with_structured_output(RAGResponse)
    db = Database()
    db = db.get_or_init_collection()
    ret = db.get_retriever()
    system_template = """
        ### Role
        You are a helpful Statistics Graduate Assistant.

        ### Instructions
        1. **Prioritize the Context:** Use the provided snippets to answer the user's question first.
        2. **Supplement if Needed:** If the context is missing specific details, says from the supplement documents, you can't answer the question.
        3. **Be Concise:** Get straight to the point but answer all the questions and requests.
        4. **Be Credible** Provide in-text citations (MLA format) of the materials you used.
        5. **Consistent** Provide in the structure: answer then citations at the end following MLA format
        ### Context
        {context}

        ### Question
        {question}
        """

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{question}")
    ])

    # 3. Build the chain with ONLY the structured_llm
    chain = (
            {'context': ret | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | structured_llm  # <-- Make sure this is the ONLY model in the chain
    )

    return chain