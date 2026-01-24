import streamlit as st
import os
import shutil

from ingest import load_files, create_vector_db
from llm_router import get_llm

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ✅ Health check (KEEP THIS AT TOP)
st.markdown("<!-- health-check -->")


# ---------- SESSION STATE ----------
if "vectordb_ready" not in st.session_state:
    st.session_state.vectordb_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------- PAGE ----------
st.set_page_config(page_title="Multi-File RAG Chatbot", layout="wide")
st.title("📂 Multi-File RAG Chatbot 🤖")
st.markdown("Developed by : Shreeyansh Asati")


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
            with st.spinner("Creating new embeddings..."):

                # 🔥 FULLY DELETE OLD EMBEDDINGS
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db", ignore_errors=True)

                docs = load_files(files, manual_text)
                create_vector_db(docs)

                st.session_state.chat_history = []
                st.session_state.vectordb_ready = True

                st.success("✅ Document processed, now you can ask questions.")


# ---------- CHAT ----------
if st.session_state.vectordb_ready:

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

        response = llm.invoke(prompt)
        answer = response.content

        st.session_state.chat_history.append((query, answer))
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(answer)

else:
    st.info("⬅ Upload data and click **Process Data**")



# ---------- FOOTER ----------
st.markdown(
    """
    <style>
    /* Add bottom padding so footer doesn't overlap chat input */
    .block-container {
        padding-bottom: 80px;
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(0,0,0,0);
        color: #B3B3B3;
        text-align: center;
        font-size: 16px;
        padding: 10px;
        z-index: 100;
    }

    /* Blue links */
    .footer a {
        color: #1DA1F2;   /* Blue */
        text-decoration: none;
        margin: 0 8px;
        font-weight: 500;
    }

    .footer a:hover {
        color: #0A66C2;   /* Darker blue on hover */
        text-decoration: underline;
    }
    </style>

    <div class="footer">
        © 2026 <b>Developed by Shreeyansh Asati</b> |
        <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">
            🔗 LinkedIn
        </a> |
        <a href="https://github.com/SHREEYANSHGIT" target="_blank">
            💻 GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True
)




