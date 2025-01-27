<h1 align='center'> ✨ PAPER AGENT ✨ </h1>
<h3 align='center'> A lightweight AI agent which reads and summarizes research papers for you.</h2>

<p align="center">
<img src="https://github.com/user-attachments/assets/7fa62d21-ec47-4c02-be2b-38269c5401d7" width="800" height="500" />
</p>

## Frameworks & Tools Used
Langchain <br>
Ollama <br>
Streamlit <br>

## Models Used
Embedding Model : DeepSeek-R1-1.5B <br>
Inference Model : DeepSeek-R1-1.5B

## Code Overview
1) User uploads a pdf

```python
## Upload PDF
def upload_pdf(file):
    with open(pdf_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


## Load PDF
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents
```

2) Text from the pdf is chunked

```python
## Split text
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        add_start_index = True,
    )
    return text_splitter.split_documents(documents)
```

3) Embeddings of the text is stored in the vector database

```python
## Embed text
def index_docs(docs):
    vector_store.add_documents(docs)

## Retrieve docs
def retrieve_docs(query):
    return vector_store.similarity_search(query)
```

4) Model is invoked to get the response to the user query

```python
## Answer query
def answer_query(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    
    return chain.invoke({"query": query, "context": context})
```

5) Simple lightweight Streamlit interface

```python
## Answer query
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
```

6) Prompt template used

```python
## Answer query
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
Just answer the user query with the context provided, no need to add any extra text.\
    
Query: {query}
Context: {context}
Answer:
"""
```

## Live Demo
https://youtu.be/5fMuMmOguSg
