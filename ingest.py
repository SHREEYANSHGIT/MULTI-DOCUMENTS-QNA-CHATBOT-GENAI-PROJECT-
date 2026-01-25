import os
import shutil
import uuid

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


UPLOAD_DIR = "uploaded_files"
CHROMA_DIR = "chroma_db"


def load_files(files, manual_text):
    documents = []
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for file in files:
        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        if path.endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())

        elif path.endswith(".txt"):
            documents.extend(TextLoader(path, encoding="utf-8").load())

    if manual_text.strip():
        documents.append(
            Document(
                page_content=manual_text,
                metadata={"source": "manual_input"}
            )
        )

    return documents


def create_vector_db(documents):
    if not documents:
        raise ValueError("No valid content found to embed.")

    # 🔥 DELETE OLD VECTOR DB
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks generated.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    collection_name = f"collection_{uuid.uuid4().hex}"

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )

    vectordb.persist()

    # 🔥 DELETE UPLOADED FILES AFTER PROCESSING
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
