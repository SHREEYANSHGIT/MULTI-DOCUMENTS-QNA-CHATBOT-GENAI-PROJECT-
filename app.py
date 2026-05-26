import streamlit as st
import os
import shutil
from ingest import load_files, create_vector_db
from llm_router import get_llm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Multi-File RAG Chatbot", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    /* Gradient Header */
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #1DA1F2, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        padding-bottom: 10px;
    }
    .sub-text {
        text-align: center;
        color: #888;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    /* Adjust footer to not overlap */
    .block-container {
        padding-bottom: 100px;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1E1E1E; 
        border-top: 1px solid #333;
        color: #E0E0E0;
        text-align: center;
        font-size: 14px;
        padding: 12px;
        z-index: 100;
    }
    .footer a {
        color: #1DA1F2;
        text-decoration: none;
        margin: 0 10px;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    .footer a:hover {
        color: #0A66C2;
        text-decoration: underline;
    }
    /* Stylish Button */
    div.stButton > button:first-child {
        background-color: #1DA1F2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background-color: #0A66C2;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "vectordb_ready" not in st.session_state:
    st.session_state.vectordb_ready = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- HEADER ----------
st.markdown('<div class="gradient-text">Multi-File RAG Chatbot 🤖</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Seamlessly interact with your documents • Developed by Shreeyansh Asati</div>', unsafe_allow_html=True)
st.divider()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Control Panel")
    st.markdown("Upload your knowledge base below to get started.")
    
    with st.container(border=True):
        st.subheader("📤 Upload Data")
        files = st.file_uploader(
            "Upload TXT or PDF files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="You can select multiple files at once."
        )
        
        with st.expander("✍️ Paste raw text instead"):
            manual_text = st.text_area("Enter text here...", height=150)
    
    with st.container(border=True):
        st.subheader("🧠 Model Selection")
        llm_choice = st.selectbox(
            "Choose your preferred LLM",
            [
                "llama-3.1-8b-instant",
                "qwen3-32b",
                "gpt-oss-120b",
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash"
            ],
            index=4 # Defaults to gemini-2.5-flash
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Process Data", type="primary"):
        if not files and not manual_text.strip():
            st.error("⚠️ Please upload files or paste text first.")
        else:
            with st.spinner("🔄 Processing documents & building vector database..."):
                # 🔥 FULLY DELETE OLD EMBEDDINGS
                if os.path.exists("chroma_db"):
                    shutil.rmtree("chroma_db", ignore_errors=True)
                docs = load_files(files, manual_text)
                create_vector_db(docs)
                st.session_state.chat_history = []
                st.session_state.vectordb_ready = True
                st.success("✅ Knowledge base ready! You can now start chatting.")

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
    
    # Render previous messages
    if not st.session_state.chat_history:
        st.info("👋 Welcome! Ask me anything about the documents you just uploaded.")
        
    for past_query, past_answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(past_query)
        with st.chat_message("assistant"):
            st.write(past_answer)
            
    # Handle new input
    query = st.chat_input("Ask a question about your documents...")
    if query:
        with st.chat_message("user"):
            st.write(query)
            
        with st.chat_message("assistant"):
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
                st.write(answer)
                
        # Save to history
        st.session_state.chat_history.append((query, answer))
else:
    # Empty State UI
    st.markdown(
        """
        <div style="text-align: center; padding: 50px; color: #888;">
            <h2>⬅️ Waiting for data...</h2>
            <p>Please upload your documents in the sidebar and click <b>Process Data</b>.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# ---------- FOOTER ----------
st.markdown(
    """
    <div class="footer">
        © 2026 <b>Developed by Shreeyansh Asati</b> |
        <a href="https://www.linkedin.com/in/shreeyansh-asati-18shreey/" target="_blank">🔗 LinkedIn</a> |
        <a href="https://github.com/SHREEYANSHGIT" target="_blank">💻 GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
