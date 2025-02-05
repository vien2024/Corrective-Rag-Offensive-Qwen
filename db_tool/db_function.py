__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import tempfile

import chromadb
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes a list of uploaded PDF documents by converting them to text chunks.

    Args:
        uploaded_files: A list of Streamlit UploadedFile objects containing the PDF documents

    Returns:
        A list of Document objects containing the chunked text from the documents

    Raises:
        IOError: If there are issues reading/writing the temporary file
    """
    all_docs = []  # To hold all processed documents
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,            
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    print(uploaded_file)
    # Ensure the file is a PDF
    if uploaded_file.name.endswith('.pdf'):
        # Store uploaded file as a temp file
        temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()  # Close the temp file to ensure it's saved properly
        # Process PDF files
        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()
        os.unlink(temp_file.name)  # Delete temp file
        all_docs.extend(text_splitter.split_documents(docs))  # Add the chunks to the overall list
    # Process Excel files
    elif uploaded_file.name.endswith('.xlxs'):
        temp_file = tempfile.NamedTemporaryFile("wb", suffix=".xlxs", delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()  # Close the temp file to ensure it's saved properly
        loader = UnstructuredExcelLoader(temp_file.name)
        docs = loader.load()
        os.unlink(temp_file.name)
        all_docs.extend(text_splitter.split_documents(docs))  # Add the chunks to the overall list
    # Process docx files
    elif uploaded_file.name.endswith('.docx'):
        temp_file = tempfile.NamedTemporaryFile("wb", suffix=".docx", delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()  # Close the temp file to ensure it's saved properly
        loader = Docx2txtLoader(temp_file.name)
        docs = loader.load()
        os.unlink(temp_file.name)
        all_docs.extend(text_splitter.split_documents(docs)) 
    # Process powerpoint files
    elif uploaded_file.name.endswith('.pptx'):
        temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pptx", delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()  # Close the temp file to ensure it's saved properly
        # Process excel files
        loader = UnstructuredPowerPointLoader(temp_file.name)
        docs = loader.load()
        os.unlink(temp_file.name)
        all_docs.extend(text_splitter.split_documents(docs)) 
    print(all_docs)
    return all_docs

def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the nomic-embed-text model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a vector collection for semantic search.

    Takes a list of document splits and adds them to a ChromaDB vector collection
    along with their metadata and unique IDs based on the filename.

    Args:
        all_splits: List of Document objects containing text chunks and metadata
        file_name: String identifier used to generate unique IDs for the chunks

    Returns:
        None. Displays a success message via Streamlit when complete.

    Raises:
        ChromaDBError: If there are issues upserting documents to the collection
    """
    collection = get_vector_collection()
    ######
    print('DEBUG################')
    print(all_splits)
    print(collection)
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")


def get_retriever():

    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    db = Chroma(client=chroma_client, collection_name="rag_app",embedding_function=embeddings)

    retriever = db.as_retriever(search_type="mmr",search_kwargs={"k": 1, "fetch_k": 5})
    return retriever

def query_collection(prompt: str, n_results: int = 10):
    """Queries the vector collection with a given prompt to retrieve relevant documents.

    Args:
        prompt: The search query text to find relevant documents.
        n_results: Maximum number of results to return. Defaults to 10.

    Returns:
        dict: Query results containing documents, distances and metadata from the collection.

    Raises:
        ChromaDBError: If there are issues querying the collection.
    """
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

