__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import ollama
import streamlit as st

from sentence_transformers import CrossEncoder
from db_tool.db_function import process_document, query_collection, add_to_vector_collection, get_retriever
from graph import build_graph
import uuid
from langsmith import utils
utils.tracing_is_enabled()
# Load api key
LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="pr-warmhearted-mining-56"

system_prompt = """
You are an Vulnerability researcher tasked with providing detailed answers based solely on the given context. 
Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question. Don't repeat.

context will be passed as "Context:"
user question will be passed as "Question:"

"""



def call_llm(context: str, prompt: str):
    """Calls the language model with context and prompt to generate a response.

    Uses Ollama to stream responses from a language model by providing context and a
    question prompt. The model uses a system prompt to format and ground its responses appropriately.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's question

    Yields:
        String chunks of the generated response as they become available from the model

    Raises:
        OllamaError: If there are issues communicating with the Ollama API
    """
    response = ollama.chat(
        model="Offensive-Qwen",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

    Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
    their relevance to the query prompt. Returns the concatenated text of the top 3 most
    relevant documents along with their indices.

    Args:
        documents: List of document strings to be re-ranked.

    Returns:
        tuple: A tuple containing:
            - relevant_text (str): Concatenated text from the top 3 ranked documents
            - relevant_text_ids (list[int]): List of indices for the top ranked documents

    Raises:
        ValueError: If documents list is empty
        RuntimeError: If cross-encoder model fails to load or rank documents
    """
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids

def predict_custom_agent_local_answer(example: dict):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state_dict = custom_graph.invoke(
        {"question": example["input"], "steps": []}, config
        )
        return {"response": state_dict["generation"], "steps": state_dict["steps"]}

if __name__ == "__main__":
    # Document Upload Area
    #print('##############Document')
    #retriever = get_retriever()
    #print(retriever.invoke("Math"))
    custom_graph = build_graph()
    with st.sidebar:
       st.set_page_config(page_title="RAG Question Answer")
       uploaded_file = st.file_uploader(
           "**📑 Upload PDF files for QnA**", type=["pdf", "docx", "pptx", "xlsx"], accept_multiple_files=True
       )

       process = st.button(
           "⚡️ Process",
       )
       ##
       #print(uploaded_file)
       #print('####################################################')
       if uploaded_file and process:
           # Because we accept multiple file, uploaded_file is list
           for file in uploaded_file:
               normalize_uploaded_file_name = file.name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
               print('####################################################')
               print(file)
               all_splits = process_document(file)
               add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    # Question and Answer Area
    st.header("🗣️ RAG Question Answer")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button(
       "🔥 Ask",
    )

    if ask and prompt:
       question = {"input": prompt}
       print("############Debug")
       print(question)
       print(question['input'])
       response = predict_custom_agent_local_answer(question)
       with st.container():
           st.write("Here is the response:")
           st.write(response["response"])