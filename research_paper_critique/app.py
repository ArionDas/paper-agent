import os
import fitz
import getpass
import streamlit as st
from werkzeug.utils import secure_filename
import pdfplumber
from PIL import Image
import pytesseract

# Configuration
UPLOAD_FOLDER = 'pdf_uploads'
IMAGE_FOLDER = 'extracted_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# --- Helper functions ---

def validate_file(file) -> bool:
    """
    Validates that the uploaded file is a PDF.
    """
    return bool(file and hasattr(file, 'name') and file.name.lower().endswith('.pdf'))


def process_paper(pdf_path):
    """
    Processes the PDF and returns analysis results and required components.
    """
    from chunking import split_recursive_text
    from embedding_model import get_embeddings_model
    from vector_store import get_vector_store
    from llm import get_llm
    from prompt import get_prompt
    from reranker import get_Cohere_Reranker

    # Extract full-text chunks
    texts = split_recursive_text(pdf_path)

    if not texts:
        raise ValueError("Failed to extract any text, tables, or image text from the PDF.")

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
            results[q] = f"Error during inference: {e}"

    return results, reranker, embedding_model, vector_store, retriever, llm


def user_chat_application(file_path, reranker, embedding_model, vector_store, retriever, llm):
    from prompt import get_prompt
    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = []

    user_input = st.text_input("Ask a question about the paper:", key="chat_input")
    response = ""
    if user_input:
        try:
            prompt = get_prompt([user_input], user_input, retriever, reranker)
            response = llm.complete(prompt).text
            st.session_state['conversation_history'].append((user_input, response))
        except Exception as e:
            st.error(f"Error during chat: {e}")

    st.markdown(f"**Question:** {user_input}")
    st.markdown(f"**Response:** {response}")

    if st.button("End Chat", key="end_chat"):
        st.session_state['conversation_history'] = []
        st.success("Chat session ended.")

# --- Streamlit UI ---

def main():
    st.title("Research Paper Analyzer")
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

    if uploaded_file:
        if not validate_file(uploaded_file):
            st.error("Invalid file type. Please upload a PDF.")
            return

        filename = secure_filename(uploaded_file.name)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {filename}")

        # Stacked buttons: Analyze then Chat
        if st.button("Analyze PDF", key="analyze_btn"):
            with st.spinner('Analyzing... this may take a while'):
                try:
                    results, reranker, embedding_model, vector_store, retriever, llm = process_paper(file_path)
                    st.session_state['chat_data'] = (file_path, reranker, embedding_model, vector_store, retriever, llm)
                    if results:
                        for question, answer in results.items():
                            st.subheader(question)
                            st.write(answer)
                        st.success("Analysis complete!")
                    else:
                        st.warning("No results returned.")
                except Exception as e:
                    st.error(f"Error processing the PDF: {e}")

        if st.button("Chat with the Paper", key="chat_btn"):
            with st.spinner('Initializing chat...'):
                try:
                    if 'chat_data' not in st.session_state:
                        _, reranker, embedding_model, vector_store, retriever, llm = process_paper(file_path)
                        st.session_state['chat_data'] = (file_path, reranker, embedding_model, vector_store, retriever, llm)
                    st.session_state['ready_for_chat'] = True
                except Exception as e:
                    st.error(f"Error initializing chat: {e}")

    if st.session_state.get('ready_for_chat'):
        st.header("Chat with the Paper")
        file_path, reranker, embedding_model, vector_store, retriever, llm = st.session_state['chat_data']
        user_chat_application(file_path, reranker, embedding_model, vector_store, retriever, llm)

if __name__ == "__main__":
    main()
