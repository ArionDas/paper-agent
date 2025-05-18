import cohere
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough


def get_prompt(texts, query, retriever, compression_retriever):
    
    ## without using the reranker
    # context = retriever.invoke(query)
    
    ## using the reranker
    # context = compression_retriever.invoke(query)
    context = compression_retriever.invoke(
        f"{query}",
    )
    
    # print(context)
    
    return f"""
    Analyze the research paper sections below to answer the question.
    Question: {query}
    Provide a detailed technical analysis focusing on the question provided.
    Answer in academic English. Make sure to add all the necessary details from the paper.
    
    If you are unsure about something, you can mention that in your answer, no need to make up incorrect answers.
    But, try to get the answer within the context of the paper.
    
    Context: {context}
    
    Don't make the answer too long, but also don't make it too short. Just mention the answer exact to the point. No unnecessary details required.
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