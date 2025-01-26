import streamlit as st

import torch
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
Assume, you are a Senior Applied Scientist with specialization in Generative AI, NLP, LLM research.\
You will be given a research paper to read and summarize.\
First understand accurately what the paper is about.\
I want you to follow these steps to come up with your response:\
    1) Read the entire paper carefully, multple times if needed.\
    2) Read the introduction, related work parts to understand the motivation and context behind the paper.\
    3) Read the methodology and results sections to understand the experiments and findings.\
    4) Include all intricate details including mathematical references in your response.\
    5) Finally, read the conclusion and limitations part to understand the shortcomings of the paper.\
Please follow these steps to provide a response to the user's query.\
Make sure to align your response with the user's context provided.\
    
Query: {query}
Context: {context}
Answer:
"""

pdf_directory = "papers/"

## Embeddings & Vector Store
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)

## Model
model = OllamaLLM(model="deepseek-r1:1.5b")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Upload PDF
def upload_pdf(file):
    with open(pdf_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


## Load PDF
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    
    return documents


## Split text
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        add_start_index = True,
    )
    
    return text_splitter.split_documents(documents)


## Embed text
def index_docs(docs):
    vector_store.add_documents(docs)
    

## Retrieve docs
def retrieve_docs(query):
    return vector_store.similarity_search(query)


## Answer query
def answer_query(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    return chain.invoke({"query": query, "context": context})



uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False,
)

if uploaded_file:
    upload_pdf(uploaded_file)
    docs = load_pdf(pdf_directory + uploaded_file.name)
    chunked_docs = split_text(docs)
    indexed_docs = index_docs(chunked_docs)
    
    query = st.chat_input("Ask a question...")
    
    if query:
        st.chat_message("user").write(query)
        related_docs = retrieve_docs(query)
        
        response = answer_query(query, related_docs)
        st.chat_message("assistant").write(response)