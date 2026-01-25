import streamlit as st
import uuid

from ingest import load_files, create_vector_db
from llm_router import get_llm


# ---------- SESSION STATE ----------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

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

    if st.button("🚀 Process Data"):

        if not files and not manual_text.strip():
            st.warning("Upload files or paste text.")
        else:
            with st.spinner("Creating fresh embeddings..."):

                st.session_state.chat_history = []

                docs = load_files(files, manual_text)
                vectordb = create_vector_db(docs)   # 👈 in-memory only

                st.session_state.vectordb = vectordb
                st.success("✅ Documents processed (fresh session)")


# ---------- CHAT ----------
if st.session_state.vectordb:

    retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(llm_choice)

    query = st.chat_input("Ask your question")

    if query:
        docs = retriever.invoke(query)

        if not docs:
            answer = "❌ No such context found in the uploaded files."
        else:
            context = "\n\n".join(d.page_content for d in docs)

            prompt = f"""
Answer ONLY from the context.
If the answer is not present, say:
"No such context found in the uploaded files."

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
