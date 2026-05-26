import streamlit as st
import os
import shutil
from ingest import load_files, create_vector_db
from llm_router import get_llm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- SESSION STATE ----------
if "vectordb_ready" not in st.session_state:
    st.session_state.vectordb_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="RAG Chatbot · Shreeyansh Asati",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- GLOBAL STYLES ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & Base ─────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Gradient noise background */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(99,102,241,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 80% 90%, rgba(16,185,129,0.10) 0%, transparent 60%),
        #0a0b10;
    min-height: 100vh;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ───────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
    backdrop-filter: blur(20px);
}

[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem 6rem 1.5rem !important;
}

/* Sidebar section headers */
.sidebar-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(99,102,241,0.8);
    margin: 1.6rem 0 0.6rem 0;
}

/* ── File uploader ─────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(99,102,241,0.35) !important;
    border-radius: 12px !important;
    background: rgba(99,102,241,0.04) !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.65) !important;
}

/* ── Selectbox ─────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
}

/* ── Text area ─────────────────────────────────── */
textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
textarea:focus {
    border-color: rgba(99,102,241,0.55) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* ── Process button ────────────────────────────── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #4338ca 100%) !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.30) !important;
    margin-top: 0.5rem !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.45) !important;
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 50%, #4f46e5 100%) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Main area padding ─────────────────────────── */
.block-container {
    padding: 2rem 3rem 120px 3rem !important;
    max-width: 900px !important;
}

/* ── Page title ────────────────────────────────── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800;
    background: linear-gradient(135deg, #e8e8f0 0%, #a5b4fc 50%, #6ee7b7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.38);
    font-weight: 300;
    letter-spacing: 0.01em;
    margin-bottom: 2rem;
}

/* ── Status pill ───────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.status-ready {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.30);
    color: #6ee7b7;
}
.status-waiting {
    background: rgba(245,158,11,0.10);
    border: 1px solid rgba(245,158,11,0.25);
    color: #fcd34d;
}
.dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    animation: pulse 1.8s ease-in-out infinite;
}
.dot-green { background: #10b981; }
.dot-amber { background: #f59e0b; }
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.75); }
}

/* ── Chat divider ──────────────────────────────── */
.chat-divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1.5rem 0;
}
.chat-divider span {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.20);
    white-space: nowrap;
}
.chat-divider::before, .chat-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
}

/* ── Chat messages ─────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(99,102,241,0.10) !important;
    border-color: rgba(99,102,241,0.20) !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(255,255,255,0.03) !important;
}

/* ── Chat input ────────────────────────────────── */
[data-testid="stChatInput"] {
    border: 1.5px solid rgba(99,102,241,0.30) !important;
    border-radius: 16px !important;
    background: rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.60) !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: #e8e8f0 !important;
    font-size: 0.9rem !important;
}

/* ── Info / success / warning alerts ──────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-width: 1px !important;
    font-size: 0.85rem !important;
}

/* ── Scrollbar ─────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.35); border-radius: 99px; }

/* ── Footer ────────────────────────────────────── */
.footer {
    position: fixed;
    left: 0; bottom: 0;
    width: 100%;
    padding: 12px 0;
    text-align: center;
    font-size: 0.75rem;
    font-weight: 400;
    color: rgba(255,255,255,0.22);
    background: linear-gradient(to top, rgba(10,11,16,0.95) 60%, transparent);
    z-index: 100;
    letter-spacing: 0.02em;
}
.footer a {
    color: rgba(99,102,241,0.75);
    text-decoration: none;
    margin: 0 6px;
    font-weight: 500;
    transition: color 0.2s;
}
.footer a:hover { color: #818cf8; }
.footer strong { color: rgba(255,255,255,0.45); }
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    # Sidebar brand
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                    background:linear-gradient(135deg,#a5b4fc,#6ee7b7);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;letter-spacing:-0.02em;">
            RAG Chatbot
        </div>
        <div style="font-size:0.68rem;color:rgba(255,255,255,0.25);
                    font-weight:300;letter-spacing:0.06em;margin-top:2px;">
            INTELLIGENT DOCUMENT Q&A
        </div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0 0 1rem 0;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">📎 Upload Documents</div>', unsafe_allow_html=True)
    files = st.file_uploader(
        "Drop TXT or PDF files here",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">✏️ Paste Text</div>', unsafe_allow_html=True)
    manual_text = st.text_area(
        "Optional text input",
        placeholder="Paste any text you'd like to query...",
        height=110,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">🤖 Language Model</div>', unsafe_allow_html=True)
    llm_choice = st.selectbox(
        "Select model",
        [
            "llama-3.1-8b-instant",
            "qwen3-32b",
            "gpt-oss-120b",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    if st.button("🚀  Process & Embed Data"):
        if not files and not manual_text.strip():
            st.warning("Please upload files or paste some text first.")
        else:
            with st.spinner("Chunking · Embedding · Indexing…"):
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db", ignore_errors=True)
                docs = load_files(files, manual_text)
                create_vector_db(docs)
                st.session_state.chat_history = []
                st.session_state.vectordb_ready = True
            st.success("✅ Knowledge base ready — start chatting!")

    # Sidebar footer
    st.markdown("""
    <div style="position:absolute;bottom:1.5rem;left:1.5rem;right:1.5rem;
                font-size:0.68rem;color:rgba(255,255,255,0.18);text-align:center;
                border-top:1px solid rgba(255,255,255,0.06);padding-top:0.75rem;">
        Built with LangChain · ChromaDB · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ---------- MAIN ----------
st.markdown('<h1 class="hero-title">Multi-File RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload documents · Select a model · Ask anything</p>', unsafe_allow_html=True)

if st.session_state.vectordb_ready:
    st.markdown("""
    <div class="status-pill status-ready">
        <div class="dot dot-green"></div> Knowledge base active
    </div>
    """, unsafe_allow_html=True)

    # Load vector DB & LLM
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = get_llm(llm_choice)

    # Replay history
    if st.session_state.chat_history:
        st.markdown('<div class="chat-divider"><span>Conversation</span></div>', unsafe_allow_html=True)
        for q, a in st.session_state.chat_history:
            st.chat_message("user").write(q)
            st.chat_message("assistant").write(a)

    # New query
    query = st.chat_input("Ask a question about your documents…")
    if query:
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"""You are a helpful assistant.
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
    st.markdown("""
    <div class="status-pill status-waiting">
        <div class="dot dot-amber"></div> Awaiting data
    </div>
    """, unsafe_allow_html=True)

    # Empty-state card
    st.markdown("""
    <div style="margin-top:1rem;padding:2.5rem 2rem;
                background:rgba(255,255,255,0.025);
                border:1.5px dashed rgba(255,255,255,0.09);
                border-radius:20px;text-align:center;">
        <div style="font-size:2.5rem;margin-bottom:1rem;">📂</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.05rem;
                    font-weight:700;color:rgba(255,255,255,0.55);margin-bottom:0.5rem;">
            No documents loaded yet
        </div>
        <div style="font-size:0.82rem;color:rgba(255,255,255,0.25);max-width:320px;
                    margin:0 auto;line-height:1.7;">
            Upload TXT or PDF files in the sidebar, pick your preferred language model,
            then click <strong style="color:rgba(99,102,241,0.8)">Process & Embed Data</strong>
            to build your knowledge base.
        </div>
        <div style="margin-top:1.75rem;display:flex;justify-content:center;gap:1.5rem;
                    flex-wrap:wrap;">
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.20);
                        display:flex;align-items:center;gap:0.4rem;">
                <span style="color:#a5b4fc;">01</span> Upload files
            </div>
            <div style="color:rgba(255,255,255,0.10);font-size:0.75rem;">→</div>
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.20);
                        display:flex;align-items:center;gap:0.4rem;">
                <span style="color:#a5b4fc;">02</span> Choose model
            </div>
            <div style="color:rgba(255,255,255,0.10);font-size:0.75rem;">→</div>
            <div style="font-size:0.75rem;color:rgba(255,255,255,0.20);
                        display:flex;align-items:center;gap:0.4rem;">
                <span style="color:#a5b4fc;">03</span> Process & chat
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    © 2026 <strong>Shreeyansh Asati</strong> &nbsp;·&nbsp;
    <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">🔗 LinkedIn</a>
    <a href="https://github.com/SHREEYANSHGIT" target="_blank">💻 GitHub</a>
</div>
""", unsafe_allow_html=True)
