import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm():
    
    model_name = "llama3-70b-8192"
    llm = Groq(model=model_name, api_key=GROQ_API_KEY)
    return llm