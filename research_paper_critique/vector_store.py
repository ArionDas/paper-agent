import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

def get_vector_store(text, embedding_model):
    
    # index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
    # vector_store = FAISS(
    #     embedding_function = embedding_model,
    #     index = index,
    #     docstore = InMemoryDocstore(),
    #     index_to_docstore_id = {}
    # )
    
    # vector_store.add_documents(text)
    
    vector_store = FAISS.from_documents(
        text,
        embedding_model,
    )
    
    faiss_retriever = vector_store.as_retriever(
        search_type = "similarity", ## mmr search type is more expensive : O(n) + O(k^2) vs O(n)
        search_kwargs = {"k": 10}
    )
    
    bm25_sparse_retriever = BM25Retriever.from_documents(text)
    bm25_sparse_retriever.k = 5
    
    ensemble_retriever = EnsembleRetriever(
        retrievers = [faiss_retriever, bm25_sparse_retriever],
        weights = [0.5, 0.5]
    )

    return vector_store, ensemble_retriever