import os
import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from typing import List, Optional
from config import CONFIG
from constants import HF_TOKEN, GROQ_API_KEY, QDRANT_API_KEY, QDRANT_URL, OPENAI_API_KEY

import torch

import datasets
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool, ManagedAgent
from smolagents import tool, LiteLLMModel, TransformersModel, Tool
from smolagents.agents import ToolCallingAgent
from litellm import completion

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS, Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

## Setting the HF token
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['GROQ_API_KEY'] = GROQ_API_KEY

## Device
device = "cuda" if torch.cuda.is_available() else "cpu"

## model
#model = LiteLLMModel(model_id = "groq/llama-3.3-70b-versatile", api = GROQ_API_KEY)
model = HfApiModel()

## RAG

## Load the knowledge base
knowledge_base = datasets.load_dataset("ariondas/research_papers")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

pdf_file_path = CONFIG["pdf_file_path"]
print(pdf_file_path)
pdf_loader = PyPDFLoader(pdf_file_path)
pdf_docs = pdf_loader.load()


## text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = CONFIG["chunk_size"],
    chunk_overlap = CONFIG["chunk_overlap"],
    add_start_index = CONFIG["add_start_index"],
    strip_whitespace = CONFIG["strip_whitespace"],
    separators = CONFIG["separators"],
)

processed_docs = text_splitter.split_documents(pdf_docs)


## Embeddings
embedding_model_name = CONFIG["embedding_model_name"]
model_kwargs = {"device": f"{device}"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceEmbeddings(
    model_name = embedding_model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs,
)

## Vector store
db = Chroma.from_documents(
    processed_docs,
    embeddings,
)


## Retriever Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = f"Read the paper provided by following the steps:\
                    1) Gather the motivation and intuition behind the paper by reading the introduction and literature review.\
                    2) Understand the essence of the paper by reading the methodology and results thoroughly like a senior researcher.\
                    3) Understand how this paper is better than the existing methods from your analysis.\
                    Provide a report on the paper by following all the above steps."
    
    inputs = {
        "query": {
            "type": "string",
            "description": f"The query to search for in the document."
        }
    }
    output_type = "string"
    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retriever = db.as_retriever(search_kwargs = {"k": 4})
        
    def forward(self, query: str) -> str:
        assert isinstance(query, str),  "Query must be a string."
        
        docs = self.retriever.invoke(
            query,
        )
        
        return "\nRetrieved documents:\n" + "".join(
            f"\n\n====== Document {str(i)} ======\n" + doc.page_content for i, doc in enumerate(docs)
        )
        
retriever_tool = RetrieverTool()


## RAG Agent
rag_agent = CodeAgent(tools=[retriever_tool], model=model, max_steps=2)

managed_rag_agent = ManagedAgent(
    agent = rag_agent,
    name = "rag-agent",
    description = "This agent is a RAG agent that can answer questions based on the document provided.",
)


## Web search tools
@tool
def visit_website(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url (str): URL of the webpage to visit.

    Returns:
        str: The content of the webpage converted to Markdown.
    """
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n", markdown_content)
        return markdown_content

    except RequestException as e:
        return f"Failed to fetch content from {url}: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
## Websearch Agent
web_agent = CodeAgent(
    tools = [DuckDuckGoSearchTool(), visit_website],
    model = model,
    max_steps = 3,
)

managed_web_agent = ManagedAgent(
    agent = web_agent,
    name = "web-agent",
    description = "This agent can search the web and provide information based on the query."
)


## Final Manager Agent
manager_agent = CodeAgent(
    tools = [],
    model = model,
    managed_agents = [managed_rag_agent, managed_web_agent],
    additional_authorized_imports=['numpy', 'pandas', 'bs4', 'requests', 'markdownify'],
    max_steps = 10,
)

manager_agent.run("Tell me about the document.")