from typing import List
from typing_extensions import TypedDict
from IPython.display import Image, display
from langchain.schema import Document
from langgraph.graph import START, END, StateGraph
from db_tool.db_function import process_document, query_collection, get_vector_collection, get_retriever
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv

#################
llm_grader = ChatOllama(model="Offensive-Qwen:latest", format="json", temperature=0)

# Prompt
prompt = PromptTemplate(
    template="""You are a teacher grading a quiz. You will be given: 
    1/ a QUESTION
    2/ A FACT provided by the student
    
    You are grading RELEVANCE RECALL:
    A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
    A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
    1 is the highest (best) score. 0 is the lowest score you can give. 
    
    Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 
    
    Avoid simply stating the correct answer at the outset.
    
    Question: {question} \n
    Fact: \n\n {documents} \n\n
    
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """,
    input_variables=["question", "documents"],
)

retrieval_grader = prompt | llm_grader | JsonOutputParser()
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    retriever = get_retriever()
    question = state["question"]
    documents = retriever.invoke(question)
    steps = state["steps"]
    steps.append("retrieve_documents")
    return {"documents": documents, "question": question, "steps": steps}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    #################
    prompt = PromptTemplate(
        template="""
        Use the following documents to answer the question.
        Question: {question} 
        Documents: {documents} 
        Answer: 
        """,
        input_variables=["question", "documents"],
    )

    # LLM
    llm = ChatOllama(model="Offensive-Qwen:latest")

    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"documents": '.'.join([document.page_content for document in documents]), "question": question})
    print('############generate')
    print(generation)
    steps = state["steps"]
    steps.append("generate_answer")
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "steps": steps,
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_document_retrieval")
    filtered_docs = []
    search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "documents": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            filtered_docs.append(d)
        else:
            search = "Yes"
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "search": search,
        "steps": steps,
    }


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    # Load environment variables from .env file
    load_dotenv()

    # Now you can access the API keys like this:
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    web_search_tool = TavilySearchResults(tavily_api_key=tavily_api_key, k=3)
    question = state["question"]
    documents = state.get("documents", [])
    steps = state["steps"]
    steps.append("web_search")
    try:
        web_results = web_search_tool.invoke({"query": question})
        print('###############Debug2')
        print(web_results)

        # Check if web_results indicates an error (assuming it has a specific structure)
        if isinstance(web_results, dict) and web_results.get("error") == 400:
            print("Received 400 HTTP error from web search. Skipping document processing.")
            return {"documents": [], "question": question, "steps": steps}  # Return immediately

        documents.extend(
            [
                Document(page_content=d["content"], metadata={"url": d["url"]})
                for d in web_results
            ]
        )
        print(documents)
        return {"documents": documents, "question": question, "steps": steps}
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"documents": [], "question": question, "steps": steps}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    search = state["search"]
    if search == "Yes":
        return "search"
    else:
        return "generate"


# Graph
def build_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("web_search", web_search)  # web search

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    custom_graph = workflow.compile()
    return custom_graph