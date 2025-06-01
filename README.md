<h1 align='center'> ✨ PAPER AGENT ✨ </h1>
<h3 align='center'> An advanced RAG application which reads and analyzes research papers for you.</h3>

<p align="center">
<img src="https://github.com/user-attachments/assets/7364125e-4c97-49b4-916d-f6f28d03c7fe"/>
</p>

## Live Link
[Paper Agent](https://arion-research.streamlit.app/) <br><br>
<strong> NOTE: </strong> <br>
This is a free of cost application, so expect redundancy and a lot of latency during the startup. Once the chat application starts up, it's quite fast. So, please bear with it during the initialization. Thank you. Feel free to put in your feedback in the [FORM](https://forms.gle/xa8UTbmciU2kJTn8A).

## Frameworks & Tools Used
Langchain <br>
Faiss (vector store) <br>
Cohere (reranker) <br>
Huggingface (embedding model) <br>
Groq <br>
Streamlit <br>

## Models Used
Embedding Model : all-MiniLM-L6-v2 <br>
Reranker Model : rerank-english-v3.0 <br>
Inference Model : llama3-70b-8192

## Code Overview
1) User provides API keys & uploads a pdf

```python
with st.sidebar:
    st.header("Configuration")
    co_api_key = st.text_input("Cohere API Key", type='password')
    groq_api_key = st.text_input("Groq API Key", type='password')
    if not (co_api_key and groq_api_key):
        st.warning("Please enter both API keys.")
        
os.environ['COHERE_API_KEY'] = co_api_key
os.environ['GROQ_API_KEY'] = groq_api_key

st.title("Research Paper Analyzer")
uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
```

2) Text from the pdf is chunked

```python
def split_recursive_text(resume_path):
    loader = PyPDFLoader(resume_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size = 1000,
        chunk_overlap=500,
    )

    texts = text_splitter.split_documents(documents)
    
    # Appending text from tables
    texts += extract_tables(resume_path)

    # Extracting images and OCR text
    image_texts = extract_images_and_ocr(resume_path)
    texts += image_texts

    return texts
```

3) Loading the Embedding Model

```python
def get_embeddings_model():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## HF
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    model_kwargs = {"device": f"{device}"}
    encode_kwargs = {"normalize_embeddings": False}
    
    embedding_model = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs,
    )
    
    return embedding_model
```

4) Hybrid Retriever using Faiss vector store

```python
def get_vector_store(text, embedding_model):
    
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
```

5) Cohere Reranker to enhance accuracy

```python
def get_Cohere_Reranker(ensemble_retriever):
    
    compressor = CohereRerank(model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )
    
    return compression_retriever
```

6) LLM for query inference

```python
def get_llm():
    
    model_name = "llama3-70b-8192"
    llm = Groq(model=model_name, api_key=GROQ_API_KEY)
    return llm
```

7) Initial paper analysis

```python
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
"""
```

8) User Chat Application

```python
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
```

## Live Demo
[Video](https://youtu.be/8qBiHGUsJgg)
