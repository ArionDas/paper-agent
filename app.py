import os

from phi.agent import Agent
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader

from phi.vectordb.qdrant import Qdrant
from phi.model.groq import Groq
import streamlit as st
import tempfile
from constants import GROQ_API_KEY, QDRANT_API_KEY, QDRANT_URL, OPENAI_API_KEY
from openai import OpenAI

client = OpenAI(
    api_key="fake-key"
)

## doesn't work
os.environ['OPENAI_API_KEY'] = "fake-key"

## Agent
def paper_agent(user: str = "Arion", query: str = "Summarize the document", uploaded_file=None):
    
    collection_name = "paper-agent"
    vector_db = Qdrant(
        url = QDRANT_URL,
        api_key = QDRANT_API_KEY,
        collection = collection_name,
    )

    ## knowledge base
    knowledge_base = PDFKnowledgeBase(
        path=uploaded_file.name,
        ## QDrant is our vector database
        vector_db = vector_db,
        reader = PDFReader(chunk=True),
    )

    knowledge_base.load(recreate=True, upsert=True)
    
    agent = Agent(
        model = Groq(id="llama-3.3-70b-versatile"),
        markdown = True,
        knowledge = knowledge_base,
        show_tool_calls = True,
        user_id = user,
    )
    
    return agent.print_response(query)


def main():
    
    st.title("Paper Agent")
    
    # File uploader for PDF
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Display the uploaded file name
        st.write("Uploaded File:", uploaded_file.name)
    
        # Save the file to the current directory
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())


    query = st.text_input("Ask a question about your document:")

    if query:
        with st.spinner("Indexing & Searching from the document..."):
            response = paper_agent(user="Arion", query=query, uploaded_file=uploaded_file)
            st.spinner("Done!")
            st.write(response)
            
            
if __name__ == "__main__":
    main()
    
