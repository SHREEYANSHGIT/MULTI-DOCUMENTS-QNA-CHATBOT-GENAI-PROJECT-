import streamlit as st
import os
import shutil
import uuid
import gc
import time   # ✅ REQUIRED

from ingest import load_files, create_vector_db
from llm_router import get_llm

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------- CONSTANTS ----------
UPLOAD_DIR = "uploaded_files"
CHROMA_DIR = "chroma_db"


# ---------- SESSION STATE ----------
if "vectordb_ready" not in st.session_state:
    st.session_state.vectordb_ready = False

if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------- PAGE ----------
st.set_page_config(page_title="Multi-File RAG Chatbot", layout="wide")
st.title("📂 Multi-File RAG Chatbot 🤖")
st.markdown("Developed by **Shreeyansh Asati**")


# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("📤 Upload Data")

    files = st.file_uploader(
        "Upload TXT / PDF",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    manual_text = st.text_area("✍️ Paste text (optional)")

    llm_choice = st.selectbox(
        "Choose LLM",
        [
            "llama-3.1-8b-instant",
            "qwen3-32b",
            "gpt-oss-120b",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash"
        ]
    )

    # ---------- PROCESS BUTTON ----------
    if st.button("🚀 Process Data"):

        if not files and not manual_text.strip():
            st.warning("Upload files or paste text.")
        else:
            with st.spinner("🔥 Resetting old data and creating fresh embeddings..."):

                # ---------- RESET SESSION ----------
                st.session_state.vectordb_ready = False
                st.session_state.chat_history = []
                gc.collect()

                # ---------- DELETE OLD UPLOADS ----------
                if os.path.exists(UPLOAD_DIR):
                    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)

                os.makedirs(UPLOAD_DIR, exist_ok=True)

                # ---------- DELETE OLD CHROMA DB (CRITICAL FIX) ----------
                if os.path.exists(CHROMA_DIR):
                    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                    time.sleep(0.5)   # ✅ LINUX FILE LOCK SAFETY

                os.makedirs(CHROMA_DIR, exist_ok=True)

                # ---------- NEW UNIQUE COLLECTION ----------
                st.session_state.collection_name = f"rag_{uuid.uuid4().hex}"

                # ---------- INGEST + EMBEDDING ----------
                docs = load_files(files, manual_text, UPLOAD_DIR)
                create_vector_db(
                    documents=docs,
                    chroma_dir=CHROMA_DIR,
                    collection_name=st.session_state.collection_name
                )

                st.session_state.vectordb_ready = True
                st.success("✅ Fresh data indexed. Old embeddings fully removed.")


# ---------- CHAT ----------
if st.session_state.vectordb_ready:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        collection_name=st.session_state.collection_name,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(llm_choice)

    query = st.chat_input("Ask your question")

    if query:
        docs = retriever.invoke(query)

        if not docs:
            answer = "❌ No such context found in the uploaded files."
        else:
            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = f"""
Answer ONLY from the context below.
If the answer is not present, say:
"No such context found in the uploaded files."

Context:
{context}

Question:
{query}
"""
            response = llm.invoke(prompt)
            answer = response.content

        st.chat_message("user").write(query)
        st.chat_message("assistant").write(answer)

else:
    st.info("⬅ Upload files and click **Process Data**")
