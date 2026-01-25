import os
import uuid

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


UPLOAD_DIR = "uploaded_files"


def load_files(files, manual_text):
    documents = []
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for file in files:
        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        if path.lower().endswith(".pdf"):
            documents.extend(PyPDFLoader(path).load())

        elif path.lower().endswith(".txt"):
            documents.extend(TextLoader(path, encoding="utf-8").load())

        # 🔥 delete file after reading
        os.remove(path)

    if manual_text.strip():
        documents.append(
            Document(page_content=manual_text, metadata={"source": "manual_input"})
        )

    return documents


def create_vector_db(documents):
    if not documents:
        raise ValueError("No valid documents to embed")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Text splitting failed")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_dir = os.path.abspath("chroma_db")
    collection_name = f"collection_{uuid.uuid4().hex}"

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir
    )

    vectordb.persist()
