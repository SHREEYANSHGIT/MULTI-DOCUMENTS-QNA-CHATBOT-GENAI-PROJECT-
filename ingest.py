import os
import uuid
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_files(files, manual_text):
    documents = []

    for file in files:
        path = os.path.join(os.getcwd(), file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        if path.lower().endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())

        elif path.lower().endswith(".txt"):
            documents.extend(TextLoader(path, encoding="utf-8").load())

        elif path.lower().endswith(".xlsx"):
            df = pd.read_excel(path)
            if not df.empty:
                documents.append(
                    Document(
                        page_content=df.to_string(index=False),
                        metadata={"source": file.name}
                    )
                )

    if manual_text.strip():
        documents.append(
            Document(
                page_content=manual_text,
                metadata={"source": "manual_input"}
            )
        )

    return documents


def create_vector_db(documents):
    # 🚨 CRITICAL SAFETY CHECK
    if not documents:
        raise ValueError("No valid documents found to embed.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document splitting resulted in empty chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 🔥 UNIQUE COLLECTION PER RUN (STREAMLIT SAFE)
    collection_name = f"collection_{uuid.uuid4().hex}"

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )

    vectordb.persist()
