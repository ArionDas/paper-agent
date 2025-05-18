import os
import sys
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate

from pdf_loader import load_pdf
from embedding_model import get_embeddings_model
from vector_store import get_vector_store
from llm import get_llm
from chunking import split_recursive_text
from prompt import get_prompt
from reranker import get_Cohere_Reranker


def main():
    
    pdf_path = "paper.pdf"
    
    ## Load pdf
    text = split_recursive_text(pdf_path)
    print(text)
    
    ## Embedding Model
    embedding_model = get_embeddings_model()
    
    ## Vector Store
    vector_store, retriever = get_vector_store(text, embedding_model)
    
    ## Reranker
    compression_retriever = get_Cohere_Reranker(retriever)
    
    ## LLM
    llm = get_llm()
    
    ## Queries
    queries = [
        "What is the main contribution of this paper?",
        "What are the main experimental results in this paper?",
        "What is the evaluation metric used in this paper?",
        "What are the methodological innovations used in this paper?",
        "What are the potential limitations of this paper?",
    ]
    
    ## Run RAG chain
    for query in queries:
        
        prompt = get_prompt(text, query, retriever, compression_retriever)
        response = llm.complete(prompt)
        
        print("Question: ", query)
        print("Answer: ", response.text)
        

if __name__ == "__main__":
    main()