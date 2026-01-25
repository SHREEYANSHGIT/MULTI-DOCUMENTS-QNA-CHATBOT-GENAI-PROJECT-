import streamlit as st
from ingest import load_files, create_vector_db
from llm_router import get_llm

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------- SESSION STATE ----------
if "ready" not in st.session_state:
    st.session_state.ready = False


# ---------- PAGE ----------
st.set_page_config(page_title="Multi-File RAG Chatbot", layout="wide")
st.title("📂 Multi-File RAG Chatbot 🤖")
st.caption("Developed by Shreeyansh Asati")


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

    if st.button("🚀 Process Data"):
        if not files and not manual_text.strip():
            st.warning("Upload files or paste text.")
        else:
            with st.spinner("Processing documents..."):
                try:
                    docs = load_files(files, manual_text)
                    create_vector_db(docs)
                    st.session_state.ready = True
                    st.success("✅ New data indexed. Old data removed.")
                except Exception as e:
                    st.error(str(e))
                    st.stop()


# ---------- CHAT ----------
if st.session_state.ready:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(llm_choice)

    query = st.chat_input("Ask your question")

    if query:
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are a helpful assistant.
Answer ONLY from the context below.

Context:
{context}

Question:
{query}
"""

        answer = llm.invoke(prompt).content
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(answer)

else:
    st.info("⬅ Upload files and click **Process Data**")
