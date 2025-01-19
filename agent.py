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
    api_key=OPENAI_API_KEY
)

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

## Agent
def paper_agent(user: str = "Arion", query: str = "Summarize the document"):
    
    collection_name = "paper-agent"
    vector_db = Qdrant(
        url = QDRANT_URL,
        api_key = QDRANT_API_KEY,
        collection = collection_name,
    )

    ## knowledge base
    knowledge_base = PDFKnowledgeBase(
        path="data/",
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
    
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if pdf_file:
        with open(os.path.join("data", "paper"), "wb") as f:
            f.write(pdf_file.read())
        st.success("PDF file received")


    query = st.text_input("Ask a question about your document:")

    if query:
        with st.spinner("Indexing & Searching from the document..."):
            response = paper_agent(user="Arion", query=query)
            st.spinner("Done!")
            st.write(response)
            
            
if __name__ == "__main__":
    main()
    
