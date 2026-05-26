import streamlit as st
import os
import shutil
from ingest import load_files, create_vector_db
from llm_router import get_llm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- PAGE CONFIG (must be first) ----------
st.set_page_config(
    page_title="RAG Chatbot · Shreeyansh",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- GLOBAL STYLES ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --surface2:  #1a2235;
    --border:    #1f2d42;
    --gold:      #f5a623;
    --gold-soft: #f5a62322;
    --teal:      #4ecdc4;
    --teal-soft: #4ecdc415;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --danger:    #ff6b6b;
    --success:   #6bcb77;
    --radius:    12px;
    --radius-lg: 20px;
}

/* ── Base reset ── */
html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.2rem !important;
}

/* ── Sidebar header ── */
.sidebar-brand {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.3rem;
}
.sidebar-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.35;
    margin-bottom: 1.8rem;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.4rem 0 0.5rem;
    font-family: 'Space Mono', monospace;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--gold) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--muted) !important;
    font-size: 0.85rem !important;
}

/* ── Text area ── */
textarea {
    background: var(--surface2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s !important;
}
textarea:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px var(--gold-soft) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--gold) !important;
}

/* ── Process button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--gold) 0%, #e8941a 100%) !important;
    color: #0b0f1a !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.65rem 1.2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 18px #f5a62340 !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Main area ── */
.main .block-container {
    padding: 2.5rem 3rem 100px !important;
    max-width: 900px !important;
}

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 2.5rem;
    padding-bottom: 1.8rem;
    border-bottom: 1px solid var(--border);
}
.page-header-icon {
    font-size: 2.4rem;
    line-height: 1;
    filter: drop-shadow(0 0 10px #f5a62360);
}
.page-header-text {}
.page-header-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.25rem;
}
.page-header-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.2;
}
.page-header-sub {
    font-size: 0.88rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.85rem;
    border-radius: 99px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
}
.status-badge.ready {
    background: #6bcb7722;
    color: var(--success);
    border: 1px solid #6bcb7744;
}
.status-badge.idle {
    background: #f5a62318;
    color: var(--gold);
    border: 1px solid #f5a62340;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.3rem !important;
    margin-bottom: 0.9rem !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: var(--gold-soft) !important;
    border-color: #f5a62335 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px var(--gold-soft) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Info / warning / success boxes ── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    border-left-width: 3px !important;
    font-size: 0.88rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--gold) !important;
}

/* ── Model tag pill ── */
.model-pill {
    display: inline-block;
    background: var(--teal-soft);
    color: var(--teal);
    border: 1px solid #4ecdc430;
    border-radius: 99px;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    padding: 0.2rem 0.7rem;
    letter-spacing: 0.06em;
    margin-bottom: 1rem;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Footer ── */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: linear-gradient(to top, var(--bg) 70%, transparent);
    color: var(--muted);
    text-align: center;
    font-size: 0.78rem;
    padding: 14px 0 10px;
    z-index: 100;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.04em;
}
.footer a {
    color: var(--gold);
    text-decoration: none;
    margin: 0 10px;
    transition: color 0.2s;
}
.footer a:hover { color: var(--teal); }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--muted);
}
.empty-state-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    filter: grayscale(0.3);
}
.empty-state-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: var(--text);
    margin-bottom: 0.5rem;
}
.empty-state-sub {
    font-size: 0.85rem;
    line-height: 1.6;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "vectordb_ready" not in st.session_state:
    st.session_state.vectordb_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_model" not in st.session_state:
    st.session_state.active_model = ""

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">RAG · Document Intelligence</div>
        <div class="sidebar-title">Knowledge<br>Ingestion</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">📎 Documents</div>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Upload TXT / PDF",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">✍️ Paste Text</div>', unsafe_allow_html=True)
    manual_text = st.text_area(
        "Paste text",
        placeholder="Or paste raw text here...",
        height=120,
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-label">🤖 Model</div>', unsafe_allow_html=True)
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
            with st.spinner("Building vector index..."):
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db", ignore_errors=True)
                docs = load_files(files, manual_text)
                create_vector_db(docs)
                st.session_state.chat_history = []
                st.session_state.vectordb_ready = True
                st.session_state.active_model = llm_choice
                st.success("✅ Index ready — ask away!")

    # Sidebar footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.68rem;color:#2d3f55;font-family:Space Mono,monospace;text-align:center;letter-spacing:0.05em;">'
        'POWERED BY LANGCHAIN + CHROMA'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------- MAIN ----------
st.markdown("""
<div class="page-header">
    <div class="page-header-icon">🧠</div>
    <div class="page-header-text">
        <div class="page-header-eyebrow">Retrieval-Augmented Generation</div>
        <div class="page-header-title">Multi-File RAG Chatbot</div>
        <div class="page-header-sub">Upload documents · Ask anything · Get grounded answers</div>
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.vectordb_ready:
    model_label = st.session_state.active_model or llm_choice
    st.markdown(
        f'<div class="status-badge ready">● Index Ready</div>'
        f'&nbsp;&nbsp;<span class="model-pill">{model_label}</span>',
        unsafe_allow_html=True,
    )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(llm_choice)

    # Replay history
    for user_msg, bot_msg in st.session_state.chat_history:
        st.chat_message("user", avatar="👤").write(user_msg)
        st.chat_message("assistant", avatar="🧠").write(bot_msg)

    query = st.chat_input("Ask a question about your documents...")
    if query:
        docs_retrieved = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs_retrieved)
        prompt = f"""You are a helpful assistant.
Answer ONLY from the context below. If the answer is not in the context, say so clearly.

Context:
{context}

Question:
{query}"""

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)
            answer = response.content

        st.session_state.chat_history.append((query, answer))
        st.chat_message("user", avatar="👤").write(query)
        st.chat_message("assistant", avatar="🧠").write(answer)

else:
    st.markdown(
        '<div class="status-badge idle">○ Awaiting Documents</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📂</div>
        <div class="empty-state-title">No documents indexed yet</div>
        <div class="empty-state-sub">
            Upload PDF or TXT files (or paste text) in the sidebar,<br>
            choose a model, then click <strong>⚡ Process &amp; Embed</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    © 2026 &nbsp;<strong>Shreeyansh Asati</strong>&nbsp; · &nbsp;
    <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">LinkedIn</a>
    <a href="https://github.com/SHREEYANSHGIT" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
