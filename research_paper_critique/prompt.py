from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough


def get_prompt(query, retriever):
    
    context = retriever.get_relevant_documents(query)
    # print(context)
    
    return f"""
    Analyze the research paper sections below to answer the question.
    Question: {query}
    Provide a detailed technical analysis focusing on the question provided.
    Answer in academic English. Make sure to add all the necessary details from the paper.
    
    If you are unsure about something, you can mention that in your answer, no need to make up incorrect answers.
    But, try to get the answer within the context of the paper.
    
    Context: {context}
    """

# InferencePrompt = PromptTemplate.from_template(
#     """Analyze the research paper sections below to answer the question.
    
#     Paper excerpts:
#     {context}
    
#     Question: {question}
    
#     Provide a detailed technical analysis focusing on the question provided.
    
#     Answer in academic English:"""
# )

# def rag_chain(retriever, llm):
#     return RunnablePassthrough(
#         {"context": retriever, "question": RunnablePassthrough()}
#         | InferencePrompt
#         | llm
#     ), InferencePrompt