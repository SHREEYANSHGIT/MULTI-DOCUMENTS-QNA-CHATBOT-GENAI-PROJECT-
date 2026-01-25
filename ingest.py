import os
import uuid
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_files(files, manual_text):
    documents = []

    # Use temp directory → auto cleaned → no file collision
    with tempfile.TemporaryDirectory() as tmpdir:
        for file in files:
            path = os.path.join(tmpdir, file.name)

            with open(path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".pdf"):
                documents.extend(PyPDFLoader(path).load())

            elif file.name.endswith(".txt"):
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ IN-MEMORY CHROMA (NO PERSISTENCE)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"rag_{uuid.uuid4().hex}"
    )

    return vectordb
