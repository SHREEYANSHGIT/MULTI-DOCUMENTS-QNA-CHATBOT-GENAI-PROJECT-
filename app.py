import streamlit as st
import os
import shutil
from ingest import load_files, create_vector_db
from llm_router import get_llm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Multi-File RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

/* ---------- Main Background ---------- */
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* ---------- Title ---------- */
.main-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(to right, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.sub-text {
    text-align: center;
    color: #cbd5e1;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ---------- Upload Box ---------- */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ---------- Text Area ---------- */
textarea {
    background-color: rgba(255,255,255,0.05) !important;
    color: white !important;
    border-radius: 12px !important;
}

/* ---------- Select Box ---------- */
div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1rem;
    font-size: 1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 0px 20px rgba(124, 58, 237, 0.5);
}

/* ---------- Chat Messages ---------- */
.stChatMessage {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 12px;
    margin-bottom: 10px;
    backdrop-filter: blur(10px);
}

/* ---------- Chat Input ---------- */
[data-testid="stChatInput"] {
    border-radius: 18px;
}

/* ---------- Footer ---------- */
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 12px;
    color: #cbd5e1;
    font-size: 15px;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(10px);
    border-top: 1px solid rgba(255,255,255,0.08);
    z-index: 999;
}

.footer a {
    color: #38bdf8;
    text-decoration: none;
    margin: 0 10px;
    font-weight: 600;
}

.footer a:hover {
    color: #818cf8;
}

/* ---------- Padding ---------- */
.block-container {
    padding-top: 2rem;
    padding-bottom: 6rem;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "vectordb_ready" not in st.session_state:
    st.session_state.vectordb_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- HEADER ----------
st.markdown(
    '<div class="main-title">📂 Multi-File RAG Chatbot 🤖</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">Developed by <b>Shreeyansh Asati</b></div>',
    unsafe_allow_html=True
)

# ---------- SIDEBAR ----------
with st.sidebar:

    st.markdown("## ⚙️ Configuration")

    files = st.file_uploader(
        "📤 Upload TXT / PDF Files",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    manual_text = st.text_area(
        "✍️ Paste Additional Text",
        height=150
    )

    llm_choice = st.selectbox(
        "🧠 Choose LLM",
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
            with st.spinner("Creating embeddings..."):

                # Delete old embeddings
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db", ignore_errors=True)

                docs = load_files(files, manual_text)

                create_vector_db(docs)

                st.session_state.chat_history = []
                st.session_state.vectordb_ready = True

                st.success("✅ Documents processed successfully!")

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

    # Display old chat history
    for q, a in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(q)

        with st.chat_message("assistant"):
            st.write(a)

    # Chat input
    query = st.chat_input("💬 Ask anything about your uploaded files...")

    if query:

        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Thinking..."):

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

        with st.chat_message("assistant"):
            st.write(answer)

else:
    st.info("⬅ Upload files and click 'Process Data' to begin.")

# ---------- FOOTER ----------
st.markdown(
    """
    <div class="footer">
        © 2026 <b>Shreeyansh Asati</b> |
        <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">
            🔗 LinkedIn
        </a>
        |
        <a href="https://github.com/SHREEYANSHGIT" target="_blank">
            💻 GitHub
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
