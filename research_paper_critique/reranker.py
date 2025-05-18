import os
from dotenv import load_dotenv
import cohere
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def get_Cohere_Reranker(ensemble_retriever):
    
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )
    
    return compression_retriever