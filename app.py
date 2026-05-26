import streamlit as st
import os
import shutil
from ingest import load_files, create_vector_db
from llm_router import get_llm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Multi-File RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
    color: white;
}

/* Main Container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 6rem;
}

/* Title */
.main-title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(to right, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 2rem;
    font-size: 1.1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(17, 25, 40, 0.75);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Upload Box */
.upload-box {
    background: rgba(255,255,255,0.05);
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3rem;
    border: none;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-size: 16px;
    font-weight: 600;
    transition: 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0px 0px 20px rgba(99,102,241,0.5);
}

/* Chat Messages */
.chat-user {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    padding: 14px;
    border-radius: 16px;
    margin-bottom: 10px;
    color: white;
}

.chat-assistant {
    background: rgba(255,255,255,0.08);
    padding: 14px;
    border-radius: 16px;
    margin-bottom: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    color: #f1f5f9;
}

/* Input box */
.stChatInput input {
    border-radius: 15px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background-color: rgba(255,255,255,0.05) !important;
    color: white !important;
}

/* Footer */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    text-align: center;
    padding: 12px;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(10px);
    color: #cbd5e1;
    font-size: 14px;
    border-top: 1px solid rgba(255,255,255,0.08);
}

.footer a {
    color: #60a5fa;
    text-decoration: none;
    margin: 0 10px;
}

.footer a:hover {
    color: #93c5fd;
}

/* Selectbox */
.stSelectbox div[data-baseweb="select"] {
    background-color: rgba(255,255,255,0.05);
    border-radius: 12px;
}

/* Text Area */
textarea {
    border-radius: 12px !important;
    background-color: rgba(255,255,255,0.05) !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "vectordb_ready" not in st.session_state:
    st.session_state.vectordb_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">📂 Multi-File RAG Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-powered document chatbot using RAG + LLMs</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.markdown("## ⚙️ Control Panel")

    st.markdown('<div class="upload-box">', unsafe_allow_html=True)

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

    process = st.button("🚀 Process Data")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ### 📌 Features
    - Multi-file RAG
    - PDF + TXT support
    - Semantic Search
    - Multiple LLM Routing
    - Fast Retrieval
    """)

# ---------------- PROCESS DATA ----------------
if process:

    if not files and not manual_text.strip():
        st.warning("⚠️ Upload files or paste text.")
    else:
        with st.spinner("🔄 Creating embeddings and vector database..."):

            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db", ignore_errors=True)

            docs = load_files(files, manual_text)
            create_vector_db(docs)

            st.session_state.chat_history = []
            st.session_state.vectordb_ready = True

        st.success("✅ Documents processed successfully!")

# ---------------- CHAT SECTION ----------------
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

    # Show previous chats
    for q, a in st.session_state.chat_history:
        st.markdown(
            f'<div class="chat-user">🧑 {q}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="chat-assistant">🤖 {a}</div>',
            unsafe_allow_html=True
        )

    query = st.chat_input("💬 Ask something about your documents...")

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

Context:
{context}

Question:
{query}
"""

        with st.spinner("🤖 Thinking..."):

            response = llm.invoke(prompt)
            answer = response.content

        st.session_state.chat_history.append((query, answer))

        st.rerun()

else:
    st.info("⬅ Upload documents and click 'Process Data' to start chatting.")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    © 2026 <b>Developed by Shreeyansh Asati</b> |
    <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">
        LinkedIn
    </a>
    |
    <a href="https://github.com/SHREEYANSHGIT" target="_blank">
        GitHub
    </a>
</div>
""", unsafe_allow_html=True)
