import torch
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## Google
    # embedding_model = VertexAIEmbeddings(model="text-embedding-004")
    
    ## HF
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": False}
    
    embedding_model = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs,
    )
    
    return embedding_model