import os
import io
import uuid

import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader
from PIL import Image
import pytesseract
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="KushalRAG AI | Smart PDF Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM EXACT DASHBOARD STYLING (PURE CSS)
# =========================================================

st.markdown(
    """
    <style>
    /* Main Background */
    .stApp {
        background-color: #f4f6fb;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Sidebar Custom Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12092b 0%, #1a0f3d 100%) !important;
        color: #ffffff;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* Sidebar Footer Box */
    .sidebar-footer-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 13px;
    }

    /* Top Banner Header */
    .hero-banner {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #3b82f6 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.18);
        margin-bottom: 25px;
    }
    .hero-left {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .hero-icon-bg {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        width: 64px;
        height: 64px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 15px;
        opacity: 0.9;
        margin-top: 4px;
    }

    /* Info Description Card */
    .info-box {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 4px solid #6366f1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 24px;
    }
    .info-box-title {
        font-size: 16px;
        font-weight: 700;
        color: #1e1b4b;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }
    .info-box-desc {
        font-size: 13.5px;
        color: #64748b;
        margin: 0;
        line-height: 1.5;
    }

    /* System Information Cards */
    .sys-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 14px 18px;
        border: 1px solid #e2e8f0;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 14px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .sys-card-icon {
        background: #f1f5f9;
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        color: #4f46e5;
    }
    .sys-card-label {
        font-size: 12px;
        color: #64748b;
        font-weight: 600;
    }
    .sys-card-value {
        font-size: 13.5px;
        font-weight: 700;
        color: #1e293b;
    }

    /* Document Ready Status Box */
    .status-ready-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 20px;
    }
    .status-ready-title {
        color: #166534;
        font-weight: 700;
        font-size: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .status-ready-sub {
        color: #15803d;
        font-size: 12px;
        margin-top: 4px;
        margin-left: 24px;
    }

    /* AI Answer Card */
    .answer-box {
        background: #ffffff;
        border-radius: 14px;
        padding: 22px;
        border: 1px solid #e0e7ff;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.05);
        margin-top: 20px;
    }
    .answer-header {
        font-size: 16px;
        font-weight: 700;
        color: #312e81;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }
    .answer-text {
        font-size: 14.5px;
        color: #334155;
        line-height: 1.6;
    }

    /* 🎨 BEAUTIFUL AUTHOR & CREDITS FOOTER */
    .developer-footer {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-top: 50px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03);
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
    }
    .footer-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .developer-avatar {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        width: 52px;
        height: 52px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        font-weight: 800;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    .developer-info-title {
        font-size: 17px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .developer-info-sub {
        font-size: 13px;
        color: #64748b;
        margin-top: 2px;
    }
    .footer-tags {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .tech-pill {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #475569;
        font-size: 12px;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD ENVIRONMENT & INITIALIZE CLIENTS
# =========================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    st.error("⚠️ API Keys Missing! Please check `.env` file.")
    st.stop()

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("rag-qa-index")

groq_client = Groq(api_key=GROQ_API_KEY)

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "openai/gpt-oss-120b"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================================================
# SESSION STATE
# =========================================================

if "namespace" not in st.session_state:
    st.session_state.namespace = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def convert_question_for_search(question):
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You convert Hindi and Hinglish questions into clear English search queries for document retrieval. Return ONLY the English search query."
                },
                {"role": "user", "content": question}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return question


prompt_template = ChatPromptTemplate.from_template(
    """
You are a strict document question-answering assistant.

RULES:
1. Answer ONLY using information explicitly present in the provided context.
2. If context lacks answer, say: "I could not find this information in the document."
3. Answer in the same language and script as the user's question.

Context:
{context}

Question:
{question}

Answer:
"""
)


def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def process_pdf(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    all_chunks = []
    progress = st.progress(0)
    total_pages = len(reader.pages)

    for page_number, page in enumerate(reader.pages, start=1):
        text = ""
        try:
            extracted = page.extract_text()
            if extracted:
                text = extracted.strip()
        except Exception:
            text = ""

        if not text:
            try:
                page_text_parts = []
                for image_file in page.images:
                    try:
                        image = Image.open(io.BytesIO(image_file.data))
                        ocr_text = pytesseract.image_to_string(image, lang="eng+hin")
                        if ocr_text.strip():
                            page_text_parts.append(ocr_text.strip())
                    except Exception:
                        continue
                text = "\n".join(page_text_parts)
            except Exception:
                text = ""

        if text:
            chunks = split_text(text, chunk_size=1000, overlap=200)
            for chunk_number, chunk in enumerate(chunks, start=1):
                all_chunks.append({
                    "text": chunk,
                    "page": page_number,
                    "chunk": chunk_number
                })

        progress.progress(page_number / total_pages)

    progress.empty()
    return all_chunks


def upload_to_pinecone(chunks, filename):
    namespace = "pdf-" + str(uuid.uuid4())
    vectors = []

    for i, item in enumerate(chunks):
        embedding = embedding_model.encode(item["text"]).tolist()
        vectors.append({
            "id": f"vec-{i}-{uuid.uuid4()}",
            "values": embedding,
            "metadata": {
                "text": item["text"],
                "page": item["page"],
                "chunk": item["chunk"],
                "filename": filename
            }
        })

    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start:start + batch_size]
        index.upsert(vectors=batch, namespace=namespace)

    return namespace


def ask_question(question):
    search_question = convert_question_for_search(question)
    query_vector = embedding_model.encode(search_question).tolist()

    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True,
        namespace=st.session_state.namespace
    )

    context_parts = []
    sources = []

    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        text = metadata.get("text", "")
        page = metadata.get("page", "Unknown")
        score = match.get("score", 0)

        if text:
            context_parts.append(f"Page {page}:\n{text}")
            sources.append({
                "page": page,
                "score": score,
                "text": text
            })

    context = "\n\n".join(context_parts)

    if not context.strip():
        return "I could not find this information in the document.", sources

    prompt = prompt_template.format(context=context, question=question)

    response = groq_client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()
    return answer, sources


# =========================================================
# PURPLE SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0;'>🤖 AI PDF Q&A</h2>", unsafe_allow_html=True)
    st.caption("Intelligent Document Assistant")
    st.divider()

    st.markdown("#### ⚙️ AI Configuration")
    st.markdown(f"**LLM:**<br>`{LLM_MODEL_NAME}`", unsafe_allow_html=True)
    st.markdown(f"**Embedding Model:**<br>`{EMBEDDING_MODEL_NAME}`", unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 🛠️ Technology Stack")
    st.markdown("""
    - 📄 PDF + OCR
    - 🧠 Sentence Transformers
    - 📌 Pinecone Vector DB
    - 🔗 LangChain
    - ⚡ Groq LLM
    - 🎨 Streamlit
    """)
    st.divider()

    st.markdown(
        """
        <div class="sidebar-footer-box">
            🎓 <strong>B.Tech CSE Project</strong><br>
            RAG-Based Q&A System
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MAIN DASHBOARD AREA
# =========================================================

# Top Hero Banner
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-left">
            <div class="hero-icon-bg">🤖</div>
            <div>
                <div class="hero-title">KushalRAG AI System</div>
                <div class="hero-subtitle">Ask intelligent questions from your PDF documents using Retrieval-Augmented Generation</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Intelligent Document Assistant Info Box
st.markdown(
    """
    <div class="info-box">
        <div class="info-box-title">🚀 Intelligent Document Assistant</div>
        <div class="info-box-desc">
            Upload a PDF document and ask questions about its content. The system extracts the document, 
            creates embeddings, searches relevant information using Pinecone, and generates an answer using an LLM.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Layout into 2 Columns (Upload Left - System Status Right)
col_left, col_right = st.columns([1.6, 1], gap="large")

with col_left:
    st.markdown("### 📄 Upload Your PDF")
    st.caption("Upload a PDF document to start asking questions. Scanned documents are also supported using OCR.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=["pdf"], 
        help="Supports PDF files up to 200MB",
        key="main_pdf_uploader"
    )

    if uploaded_file is not None:
        if st.session_state.document_name != uploaded_file.name:
            with st.spinner("🔄 Extracting text, generating vectors & indexing into Pinecone..."):
                try:
                    chunks = process_pdf(uploaded_file)
                    if not chunks:
                        st.error("❌ PDF me koi readable text nahi mila.")
                    else:
                        namespace = upload_to_pinecone(chunks, uploaded_file.name)
                        st.session_state.namespace = namespace
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.document_ready = True
                        st.session_state.last_answer = None
                        st.session_state.last_sources = []
                        st.success(f"✅ Indexed {len(chunks)} chunks into vector store.")
                except Exception as e:
                    st.error(f"Error: {e}")

with col_right:
    if st.session_state.document_ready:
        st.markdown(
            f"""
            <div class="status-ready-box">
                <div class="status-ready-title">✅ Document Ready</div>
                <div class="status-ready-sub"><strong>{st.session_state.document_name}</strong> is indexed and ready for questions.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("### ⚙️ System Information")
    
    doc_display = st.session_state.document_name if st.session_state.document_name else "No document uploaded"
    
    st.markdown(
        f"""
        <div class="sys-card">
            <div class="sys-card-icon">📄</div>
            <div>
                <div class="sys-card-label">Document</div>
                <div class="sys-card-value">{doc_display}</div>
            </div>
        </div>
        <div class="sys-card">
            <div class="sys-card-icon">🧠</div>
            <div>
                <div class="sys-card-label">Embedding Model</div>
                <div class="sys-card-value">{EMBEDDING_MODEL_NAME}</div>
            </div>
        </div>
        <div class="sys-card">
            <div class="sys-card-icon">⚡</div>
            <div>
                <div class="sys-card-label">Language Model</div>
                <div class="sys-card-value">{LLM_MODEL_NAME}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# Question Section
st.markdown("### 💬 Ask Your Question")
st.caption("Ask anything related to the uploaded document. English, Hindi, and Hinglish questions are supported.")

ask_col1, ask_col2 = st.columns([4, 1])

with ask_col1:
    question = st.text_input(
        "Question",
        placeholder="Example: What is the main objective of the internship?",
        label_visibility="collapsed"
    )

with ask_col2:
    ask_button = st.button("✨ Ask AI", type="primary", use_container_width=True)

if ask_button:
    if not st.session_state.document_ready:
        st.warning("⚠️ Pehle PDF upload karein.")
    elif not question.strip():
        st.warning("⚠️ Kripya question enter karein.")
    else:
        with st.spinner("🤖 Searching context and generating answer..."):
            try:
                answer, sources = ask_question(question)
                st.session_state.last_answer = answer
                st.session_state.last_sources = sources
            except Exception as e:
                st.error(f"Error: {e}")

# Answer Output Section
if st.session_state.last_answer:
    st.markdown(
        f"""
        <div class="answer-box">
            <div class="answer-header">✨ AI Answer</div>
            <div class="answer-text">{st.session_state.last_answer}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.last_sources:
        with st.expander("📚 Retrieved Sources (View document context used for this answer)", expanded=False):
            for i, source in enumerate(st.session_state.last_sources, start=1):
                st.markdown(f"**Source {i} — Page {source['page']}** *(Score: `{source['score']:.3f}`)*")
                st.info(source["text"])

# 🎨 BEAUTIFUL DEVELOPER FOOTER
st.markdown(
    """
    <div class="developer-footer">
        <div class="footer-left">
            <div class="developer-avatar">KS</div>
            <div>
                <div class="developer-info-title">Designed & Developed by Kushal Salvi</div>
                <div class="developer-info-sub">B.Tech Computer Science & Engineering • Final Year Capstone Project</div>
            </div>
        </div>
        <div class="footer-tags">
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">Pinecone</span>
            <span class="tech-pill">LangChain</span>
            <span class="tech-pill">Groq Llama-3</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)