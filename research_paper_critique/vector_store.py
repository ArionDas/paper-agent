import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

def get_vector_store(text, embedding_model):
    
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function = embedding_model,
        index = index,
        docstore = InMemoryDocstore(),
        index_to_docstore_id = {}
    )
    
    vector_store.add_documents(text)
    
    retriever = vector_store.as_retriever(
        search_type = "similarity", ## mmr search type is more expensive : O(n) + O(k^2) vs O(n)
        search_kwargs = {"k": 5}
    )

    return vector_store, retriever