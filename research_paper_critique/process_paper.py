from chunking import split_recursive_text
from embedding_model import get_embeddings_model
from vector_store import get_vector_store
from llm import get_llm
from prompt import get_prompt
from reranker import get_Cohere_Reranker

def process_paper(pdf_path) -> dict:
    """
    Processes the PDF and returns question-answer mapping.
    """
    texts = split_recursive_text(pdf_path)
    if not texts:
        raise ValueError("Failed to extract text from the PDF.")

    embedding_model = get_embeddings_model()
    vector_store, retriever = get_vector_store(texts, embedding_model)
    reranker = get_Cohere_Reranker(retriever)
    llm = get_llm()

    queries = [
        "What is the main contribution of the paper?",
        "What are the main experimental results in this paper?",
        "What is the evaluation metric used in this paper?",
        "What are the methodological innovations used in this paper?",
        "What are the potential limitations of this paper?",
    ]

    results = {}
    for q in queries:
        try:
            prompt = get_prompt(texts, q, retriever, reranker)
            resp = llm.complete(prompt)
            results[q] = resp.text
        except Exception as e:
            results[q] = f"Error: {e}"
    return results