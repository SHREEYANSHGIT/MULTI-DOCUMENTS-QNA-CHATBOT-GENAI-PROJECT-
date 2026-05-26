import streamlit as st
import os
import shutil
from ingest import load_files, create_vector_db
from llm_router import get_llm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- PAGE CONFIGURATION ----------
st.set_page_config(
    page_title="RAG Chatbot · Shreeyansh",
    page_icon="🎉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ─── GLOBAL ─────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: #06060E;
    font-family: 'DM Sans', sans-serif;
    color: #E2E8F0;
}

/* Animated noise-grain overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.6;
}

/* Radial glow orbs */
.stApp::after {
    content: '';
    position: fixed;
    top: -20%;
    left: -10%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: drift 12s ease-in-out infinite alternate;
}

@keyframes drift {
    from { transform: translate(0, 0) scale(1); }
    to   { transform: translate(80px, 40px) scale(1.1); }
}

/* ─── SIDEBAR ─────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0D0D1A !important;
    border-right: 1px solid rgba(99,102,241,0.15);
    padding-top: 0 !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0;
}

/* Sidebar accent bar at top */
section[data-testid="stSidebar"]::before {
    content: '';
    display: block;
    height: 3px;
    background: linear-gradient(90deg, #6366F1, #22D3EE, #A78BFA);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
    margin-bottom: 24px;
}

@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif !important;
    color: #E2E8F0 !important;
    letter-spacing: -0.02em;
}

/* ─── MAIN TITLE ──────────────────────────────── */
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.6rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    text-align: center;
    line-height: 1.1;
    margin: 48px 0 10px;
    background: linear-gradient(135deg, #E2E8F0 10%, #A5B4FC 45%, #22D3EE 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sub-title {
    text-align: center;
    color: #64748B;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 48px;
}

/* ─── DIVIDER ─────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 20px 0 !important;
}

/* ─── FILE UPLOADER ───────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(99,102,241,0.35);
    border-radius: 12px;
    padding: 12px;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.7);
}

/* ─── TEXT AREA ───────────────────────────────── */
textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #CBD5E1 !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.3s !important;
}
textarea:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* ─── SELECTBOX ───────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}

/* ─── BUTTONS ─────────────────────────────────── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    border: none !important;
    border-radius: 10px;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 0.03em;
    padding: 14px 20px;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3);
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.25s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.45);
}

.stButton > button:hover::before { opacity: 1; }

.stButton > button:active {
    transform: translateY(0);
}

/* ─── CHAT MESSAGES ───────────────────────────── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    backdrop-filter: blur(8px);
    animation: fadeSlideUp 0.35s ease both;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-color: rgba(99,102,241,0.2) !important;
    background: rgba(99,102,241,0.06) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-color: rgba(34,211,238,0.15) !important;
    background: rgba(34,211,238,0.04) !important;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ─── CHAT INPUT ──────────────────────────────── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(12px);
}

[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.10) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    border: none !important;
    color: #E2E8F0 !important;
}

/* ─── STATUS BOX ──────────────────────────────── */
[data-testid="stStatus"] {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 12px !important;
}

/* ─── INFO / ALERT BOXES ──────────────────────── */
[data-testid="stAlert"] {
    background: rgba(34,211,238,0.06) !important;
    border: 1px solid rgba(34,211,238,0.2) !important;
    border-radius: 12px !important;
    color: #94A3B8 !important;
}

/* ─── SPINNER ─────────────────────────────────── */
[data-testid="stSpinner"] {
    color: #6366F1 !important;
}

/* ─── SCROLLBAR ───────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.6); }

/* ─── FOOTER ──────────────────────────────────── */
.block-container { padding-bottom: 90px !important; }

.modern-footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: rgba(6, 6, 14, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-top: 1px solid rgba(99,102,241,0.15);
    color: #475569;
    text-align: center;
    font-size: 13px;
    padding: 10px 16px;
    z-index: 9999;
    letter-spacing: 0.02em;
}

.modern-footer b { color: #94A3B8; }

.modern-footer a {
    color: #818CF8;
    text-decoration: none;
    margin: 0 10px;
    font-weight: 600;
    transition: color 0.2s;
    letter-spacing: 0.03em;
}

.modern-footer a:hover { color: #22D3EE; }

/* ─── EMPTY STATE CARD ────────────────────────── */
.empty-state {
    text-align: center;
    padding: 60px 40px;
    border: 1px dashed rgba(99,102,241,0.2);
    border-radius: 20px;
    background: rgba(99,102,241,0.03);
    margin-top: 20px;
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 16px;
    display: block;
}

.empty-state h3 {
    font-family: 'Syne', sans-serif;
    color: #94A3B8;
    font-size: 1.2rem;
    margin-bottom: 8px;
}

.empty-state p {
    color: #475569;
    font-size: 0.9rem;
    max-width: 340px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ─── SIDEBAR BRAND BADGE ─────────────────────── */
.brand-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 16px;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 12px;
    margin-bottom: 20px;
}

.brand-badge-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22D3EE;
    box-shadow: 0 0 8px #22D3EE;
    animation: pulse 2s ease infinite;
    flex-shrink: 0;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

.brand-badge-text {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: #94A3B8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ─── LABEL STYLING ───────────────────────────── */
label, .stSelectbox label, .stFileUploader label, .stTextArea label {
    color: #64748B !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ─── TOAST ───────────────────────────────────── */
[data-testid="stToast"] {
    background: #1E1B4B !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 12px !important;
    color: #C7D2FE !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "vectordb_ready" not in st.session_state:
    st.session_state.vectordb_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- MAIN HEADER ----------
st.markdown('<h1 class="main-title">Multi-File RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Intelligent document analysis · Powered by advanced LLMs</p>', unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("""
    <div class="brand-badge">
        <div class="brand-badge-dot"></div>
        <span class="brand-badge-text">RAG Intelligence Engine</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Configuration")
    st.markdown("<p style='color:#475569;font-size:0.85rem;margin-bottom:16px;'>Upload your knowledge base to get started.</p>", unsafe_allow_html=True)

    with st.container():
        files = st.file_uploader(
            "Upload TXT / PDF",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="Select one or multiple files to analyze."
        )

        manual_text = st.text_area(
            "Or paste raw text",
            placeholder="Paste any additional context here...",
            height=120
        )

    st.divider()

    llm_choice = st.selectbox(
        "Intelligence Model",
        [
            "llama-3.1-8b-instant",
            "qwen3-32b",
            "gpt-oss-120b",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash"
        ],
        help="Choose the underlying LLM to power your responses."
    )

    st.divider()

    if st.button("✦ Process Data & Initialize"):
        if not files and not manual_text.strip():
            st.error("⚠️ Please upload files or paste text to proceed.")
        else:
            with st.status("Initializing Knowledge Base...", expanded=True) as status:
                st.write("🗑️ Cleaning previous database...")
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db", ignore_errors=True)

                st.write("📚 Loading and chunking documents...")
                docs = load_files(files, manual_text)

                st.write("🧩 Generating vector embeddings...")
                create_vector_db(docs)

                st.session_state.chat_history = []
                st.session_state.vectordb_ready = True
                status.update(label="✅ Database Ready!", state="complete", expanded=False)
            st.toast("Knowledge base ready — start asking questions!", icon="🎉")

# ---------- CHAT INTERFACE ----------
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

    if not st.session_state.chat_history:
        st.info("✦ **Documents loaded.** Ask anything — your knowledge base is ready.")

    for q, a in st.session_state.chat_history:
        with st.chat_message("user", avatar="👤"):
            st.write(q)
        with st.chat_message("assistant", avatar="✦"):
            st.write(a)

    query = st.chat_input("Ask anything about your documents...")
    if query:
        with st.chat_message("user", avatar="👤"):
            st.write(query)

        with st.chat_message("assistant", avatar="✦"):
            with st.spinner("Retrieving context & generating response..."):
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
                st.write(answer)

        st.session_state.chat_history.append((query, answer))

else:
    # Empty State
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-state-icon">✦</span>
            <h3>Awaiting Knowledge Base</h3>
            <p>Upload your documents in the sidebar and click <strong style="color:#818CF8">Process Data & Initialize</strong> to begin a conversation.</p>
        </div>
        """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(
    """
    <div class="modern-footer">
        © 2026 <b>Developed by Shreeyansh Asati</b> &nbsp;·&nbsp;
        <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">🔗 LinkedIn</a>
        <a href="https://github.com/SHREEYANSHGIT" target="_blank">💻 GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
