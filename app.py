import streamlit as st
import os
import shutil

from ingest import load_files, create_vector_db
from llm_router import get_llm

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RAG AI · Shreeyansh",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================
if "vectordb_ready" not in st.session_state:
    st.session_state.vectordb_ready = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "active_model" not in st.session_state:
    st.session_state.active_model = ""

if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0


# =========================================================
# GLOBAL CSS
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg: #050816;
    --surface: #0f172a;
    --surface-light: #111827;
    --border: rgba(255,255,255,0.08);
    --gold: #f5a623;
    --gold-soft: rgba(245,166,35,0.12);
    --cyan: #4ecdc4;
    --cyan-soft: rgba(78,205,196,0.12);
    --text: #f8fafc;
    --muted: #94a3b8;
    --radius: 18px;
    --radius-lg: 28px;
}

html, body, .stApp {
    background:
        radial-gradient(circle at top left, #172554 0%, transparent 25%),
        radial-gradient(circle at bottom right, #581c87 0%, transparent 25%),
        linear-gradient(135deg, #050816 0%, #0b1120 100%) !important;
    background-attachment: fixed;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

#MainMenu, header, footer { visibility: hidden; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 999px; }

[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.78) !important;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.2rem; }

.glass-card {
    background: rgba(17,24,39,0.55);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border-radius: 24px;
    padding: 1.4rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
}

.topbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(5,8,22,0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 14px 22px;
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.logo {
    font-family: 'Space Mono', monospace;
    color: var(--gold);
    font-size: 1rem;
    letter-spacing: 0.18em;
    font-weight: 700;
}
.top-sub { color: var(--muted); font-size: 0.9rem; }

.main .block-container { padding: 2rem 3rem 120px; max-width: 1100px; }

.hero { padding: 1rem 0 2rem; }
.hero-badge {
    display: inline-block;
    padding: 8px 14px;
    background: rgba(245,166,35,0.10);
    border: 1px solid rgba(245,166,35,0.25);
    border-radius: 999px;
    color: var(--gold);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    font-family: 'Space Mono', monospace;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-size: 3.4rem;
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: 1rem;
    background: linear-gradient(90deg, #ffffff, #f5a623);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { max-width: 720px; color: var(--muted); line-height: 1.8; font-size: 1rem; }

.metric-card {
    background: rgba(17,24,39,0.52);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 22px;
    padding: 1.2rem;
    backdrop-filter: blur(14px);
}
.metric-title { color: var(--muted); font-size: 0.78rem; margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { font-size: 1.5rem; font-weight: 700; }

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1.5px dashed rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 0.4rem;
}
[data-testid="stFileUploader"]:hover { border-color: var(--gold); }

textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    color: white !important;
    font-size: 0.92rem !important;
}
textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,0.15) !important;
}

[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
}

[data-testid="stButton"] > button {
    width: 100%;
    border: none !important;
    border-radius: 18px !important;
    padding: 0.85rem 1rem !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace !important;
    color: #050816 !important;
    background: linear-gradient(135deg, #f5a623, #ffb703) !important;
    box-shadow: 0 0 15px rgba(245,166,35,0.35), 0 0 40px rgba(245,166,35,0.12);
    transition: all 0.25s ease !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(245,166,35,0.55), 0 0 55px rgba(245,166,35,0.18);
}

[data-testid="stChatMessage"] {
    border-radius: 24px !important;
    padding: 1.2rem 1.3rem !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    backdrop-filter: blur(16px);
    margin-bottom: 1rem !important;
    transition: all 0.2s ease;
}
[data-testid="stChatMessage"]:hover { transform: translateY(-2px); }
[data-testid="stChatMessage"][data-testid*="user"] {
    background: linear-gradient(135deg, rgba(245,166,35,0.14), rgba(245,166,35,0.04)) !important;
}
[data-testid="stChatMessage"][data-testid*="assistant"] {
    background: rgba(255,255,255,0.03) !important;
}

[data-testid="stChatInput"] {
    background: rgba(17,24,39,0.65) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 22px !important;
    backdrop-filter: blur(14px);
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,0.12);
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    border-radius: 999px;
    background: rgba(107,203,119,0.12);
    border: 1px solid rgba(107,203,119,0.22);
    color: #6bcb77;
    font-size: 0.82rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
}
.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #6bcb77;
    box-shadow: 0 0 12px #6bcb77;
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0%   { transform: scale(1);   opacity: 1;   }
    50%  { transform: scale(1.3); opacity: 0.6; }
    100% { transform: scale(1);   opacity: 1;   }
}

.empty-state { text-align: center; padding: 5rem 2rem; }
.empty-icon {
    font-size: 5rem;
    margin-bottom: 1rem;
    animation: floaty 4s ease-in-out infinite;
}
@keyframes floaty {
    0%   { transform: translateY(0px);   }
    50%  { transform: translateY(-12px); }
    100% { transform: translateY(0px);   }
}
.empty-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.6rem; }
.empty-sub { color: var(--muted); line-height: 1.8; }

.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    text-align: center;
    padding: 14px;
    color: #64748b;
    font-size: 0.82rem;
    background: linear-gradient(to top, rgba(5,8,22,1), rgba(5,8,22,0));
    backdrop-filter: blur(8px);
}
.footer a { color: var(--gold); text-decoration: none; margin: 0 8px; }
.footer a:hover { color: var(--cyan); }

</style>
""", unsafe_allow_html=True)


# =========================================================
# TOPBAR
# =========================================================
st.markdown("""
<div class="topbar">
    <div class="logo">🧠 RAG AI</div>
    <div class="top-sub">Multi-File Document Intelligence</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("""
    <h2 style="margin-top:0; font-size:1.2rem; margin-bottom:1.2rem;">
        📂 Knowledge Upload
    </h2>
    """, unsafe_allow_html=True)

    files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    manual_text = st.text_area(
        "Paste text",
        placeholder="Paste raw text here...",
        height=140,
        label_visibility="collapsed",
    )

    llm_choice = st.selectbox(
        "Choose LLM",
        [
            "llama-3.1-8b-instant",
            "qwen3-32b",
            "gpt-oss-120b",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚡ Process & Embed"):
        if not files and not manual_text.strip():
            st.warning("Upload files or paste text first.")
        else:
            with st.spinner("Building vector database..."):
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db", ignore_errors=True)

                docs = load_files(files, manual_text)
                create_vector_db(docs)

                st.session_state.vectordb_ready = True
                st.session_state.chat_history = []
                st.session_state.active_model = llm_choice
                st.session_state.doc_count = len(docs)

                st.success("✅ Vector database ready!")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        text-align: center;
        color: #475569;
        font-size: 0.72rem;
        font-family: 'Space Mono', monospace;
        letter-spacing: 0.08em;
    ">
        POWERED BY LANGCHAIN · CHROMA · STREAMLIT
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# HERO SECTION
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI DOCUMENT INTELLIGENCE</div>
    <h1>Multi-File RAG Chatbot</h1>
    <p>
        Upload PDFs, TXT files, or raw text and chat with your documents
        using semantic search and powerful LLMs.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# METRICS
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Documents</div>
        <div class="metric-value">{len(files) if files else 0}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Chunks</div>
        <div class="metric-value">{st.session_state.doc_count}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Model</div>
        <div class="metric-value" style="font-size:1rem;">{llm_choice}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# CHAT SECTION
# =========================================================
if st.session_state.vectordb_ready:

    st.markdown("""
    <div class="status-badge">
        <div class="status-dot"></div>
        INDEX READY
    </div>
    """, unsafe_allow_html=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(llm_choice)

    # Replay chat history
    for user_msg, bot_msg in st.session_state.chat_history:
        st.chat_message("user", avatar="👤").write(user_msg)
        st.chat_message("assistant", avatar="🧠").write(bot_msg)

    # Chat input
    query = st.chat_input("Ask a question about your documents...")

    if query:
        st.chat_message("user", avatar="👤").write(query)

        docs_retrieved = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs_retrieved)

        prompt = f"""You are a helpful AI assistant.

Answer ONLY from the provided context.
If the answer is not present in the context, say clearly that the information is unavailable.

Context:
{context}

Question:
{query}
"""

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)
            answer = response.content

        st.chat_message("assistant", avatar="🧠").write(answer)
        st.session_state.chat_history.append((query, answer))

else:

    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📂</div>
        <div class="empty-title">No Documents Indexed</div>
        <div class="empty-sub">
            Upload PDF/TXT files or paste raw text in the sidebar.<br>
            Then click <strong>⚡ Process &amp; Embed</strong> to start chatting with your data.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    © 2026 <strong>Shreeyansh Asati</strong>
    ·
    <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">LinkedIn</a>
    ·
    <a href="https://github.com/SHREEYANSHGIT" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
