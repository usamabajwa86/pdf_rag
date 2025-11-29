import streamlit as st
import os
import pickle
import hashlib
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import glob
# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    JSONLoader,
    UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Load environment
load_dotenv()

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
NUM_RESULTS = 8

# Available Models
AVAILABLE_MODELS = {
    "DeepSeek": {
        "DeepSeek Chat": "deepseek-chat",
        "DeepSeek Coder": "deepseek-coder"
    },
    "ChatGPT (OpenAI)": {
        "GPT-4": "gpt-4",
        "GPT-4 Turbo": "gpt-4-turbo-preview",
        "GPT-3.5 Turbo": "gpt-3.5-turbo"
    },
    "Groq": {
        "Llama 3.1 70B": "llama-3.1-70b-versatile",
        "Llama 3.1 8B": "llama-3.1-8b-instant",
        "Mixtral 8x7B": "mixtral-8x7b-32768"
    }
}

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_available_ollama_models():
    """Get list of available Ollama models"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            return [model['name'] for model in models_data.get('models', [])]
        return []
    except:
        return []

def initialize_llm(model_provider, model_name, api_key=None):
    """Initialize the appropriate LLM based on provider"""
    try:
        if model_provider == "DeepSeek":
            if not api_key:
                st.error("❌ DEEPSEEK_API_KEY not found!")
                return None
            return ChatOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                model=model_name,
                temperature=0
            )
        
        elif model_provider == "ChatGPT (OpenAI)":
            if not api_key:
                st.error("❌ OPENAI_API_KEY not found!")
                return None
            return ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0
            )
        
        elif model_provider == "Groq":
            if not api_key:
                st.error("❌ GROQ_API_KEY not found!")
                return None
            from langchain_groq import ChatGroq
            return ChatGroq(
                groq_api_key=api_key,
                model_name=model_name,
                temperature=0
            )

        return None
    except Exception as e:
        st.error(f"❌ Error initializing LLM: {e}")
        return None

# Try to get API keys from secrets first, then env
try:
    deepseek_api_key = st.secrets.get('DEEPSEEK_API_KEY', os.environ.get('DEEPSEEK_API_KEY'))
    openai_api_key = st.secrets.get('OPENAI_API_KEY', os.environ.get('OPENAI_API_KEY'))
    groq_api_key = st.secrets.get('GROQ_API_KEY', os.environ.get('GROQ_API_KEY'))
except:
    deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY')
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    groq_api_key = os.environ.get('GROQ_API_KEY')

# Set default API keys if not found (fallback only, use secrets.toml or Streamlit Cloud Secrets)
if not deepseek_api_key:
    deepseek_api_key = os.environ.get('DEEPSEEK_KEY_FALLBACK', '')
if not openai_api_key:
    openai_api_key = os.environ.get('OPENAI_KEY_FALLBACK', '')
if not groq_api_key:
    groq_api_key = os.environ.get('GROQ_KEY_FALLBACK', '')

def load_document(file_path):
    """Load document based on file type"""
    file_ext = file_path.suffix.lower()
    
    try:
        if file_ext == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_ext == '.txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
        elif file_ext in ['.doc', '.docx']:
            loader = UnstructuredWordDocumentLoader(str(file_path))
        elif file_ext == '.csv':
            loader = CSVLoader(str(file_path))
        elif file_ext == '.json':
            loader = JSONLoader(str(file_path), jq_schema='.', text_content=False)
        elif file_ext in ['.md', '.markdown']:
            loader = UnstructuredMarkdownLoader(str(file_path))
        else:
            # Try as text file for unknown types
            loader = TextLoader(str(file_path), encoding='utf-8')
        
        return loader.load()
    except Exception as e:
        st.warning(f"⚠️ Could not load {file_path.name}: {str(e)}")
        return []

def clean_text(text):
    """Clean and validate text content"""
    if not text or not isinstance(text, str):
        return ""
    cleaned = text.encode('utf-8', errors='ignore').decode('utf-8').strip()
    return ' '.join(cleaned.split())

def get_all_documents(folder_path):
    """Get all supported documents from specified folder and subfolders"""
    supported_extensions = ['*.pdf', '*.txt', '*.docx', '*.doc', '*.csv', '*.json', '*.md', '*.markdown']
    all_files = []
    
    for ext in supported_extensions:
        patterns = [
            os.path.join(folder_path, f"**/{ext}"),
            os.path.join(folder_path, ext)
        ]
        for pattern in patterns:
            all_files.extend(glob.glob(pattern, recursive=True))
    
    return [Path(f) for f in set(all_files) if os.path.exists(f)]

def scan_folder_for_databases(folder_path):
    """Scan a folder for vector databases"""
    databases = []
    try:
        if not os.path.exists(folder_path):
            return databases

        vector_db_path = os.path.join(folder_path, "vector_databases")
        if os.path.exists(vector_db_path):
            for item in os.listdir(vector_db_path):
                if item.startswith('db_'):
                    db_path = os.path.join(vector_db_path, item)
                    metadata_path = os.path.join(db_path, 'metadata.json')
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            databases.append({
                                'name': f"{metadata.get('embedding_model', 'Unknown')} - {metadata.get('total_pdfs', 0)} PDFs",
                                'path': vector_db_path,
                                'metadata': metadata,
                                'db_folder': item
                            })
                        except:
                            continue

    except Exception:
        pass

    return databases

def load_existing_database(db_path, db_folder):
    """Load an existing vector database"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{EMBEDDING_MODEL}",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        base_path = os.path.join(db_path, db_folder)
        faiss_path = os.path.join(base_path, "faiss_index")
        bm25_path = os.path.join(base_path, "bm25_retriever.pkl")
        documents_path = os.path.join(base_path, "documents.pkl")
        metadata_path = os.path.join(base_path, "metadata.json")

        vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

        with open(bm25_path, 'rb') as f:
            bm25_retriever = pickle.load(f)

        with open(documents_path, 'rb') as f:
            documents = pickle.load(f)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return vector_store, bm25_retriever, documents, embeddings, metadata

    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

def create_new_database(document_files, save_folder):
    """Create new vector database from document files"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{EMBEDDING_MODEL}",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        with st.spinner(f"🔄 Processing {len(document_files)} documents..."):
            documents = []
            progress_bar = st.progress(0)

            for i, file_path in enumerate(document_files):
                try:
                    st.text(f"📄 Loading {file_path.name} ({i+1}/{len(document_files)})...")
                    file_docs = load_document(file_path)

                    valid_docs = []
                    for doc in file_docs:
                        cleaned_content = clean_text(doc.page_content)
                        if len(cleaned_content) > 20:
                            doc.page_content = cleaned_content
                            doc.metadata['source_file'] = file_path.name
                            doc.metadata['full_path'] = str(file_path)
                            valid_docs.append(doc)

                    documents.extend(valid_docs)
                    progress_bar.progress((i + 1) / len(document_files))

                except Exception as e:
                    st.warning(f"⚠️ Failed to load {file_path.name}: {str(e)}")

            st.text("✂️ Splitting documents...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            final_documents = text_splitter.split_documents(documents)

            st.text("🔢 Creating vector database...")
            vector_store = FAISS.from_documents(final_documents, embeddings)

            st.text("🔍 Creating keyword index...")
            bm25_retriever = BM25Retriever.from_documents(final_documents)
            bm25_retriever.k = NUM_RESULTS

            st.text("💾 Saving database...")
            config_str = f"{sorted([str(p) for p in document_files])}_{EMBEDDING_MODEL}_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]

            os.makedirs(save_folder, exist_ok=True)
            db_folder = f"db_{config_hash}"
            db_path = os.path.join(save_folder, db_folder)
            os.makedirs(db_path, exist_ok=True)

            vector_store.save_local(os.path.join(db_path, "faiss_index"))

            with open(os.path.join(db_path, "bm25_retriever.pkl"), 'wb') as f:
                pickle.dump(bm25_retriever, f)

            with open(os.path.join(db_path, "documents.pkl"), 'wb') as f:
                pickle.dump(final_documents, f)

            metadata = {
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'embedding_model': EMBEDDING_MODEL,
                'chunk_size': CHUNK_SIZE,
                'chunk_overlap': CHUNK_OVERLAP,
                'total_pdfs': len(document_files),
                'total_chunks': len(final_documents),
                'pdf_files': [str(p) for p in document_files],
                'config_hash': config_hash
            }

            with open(os.path.join(db_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            progress_bar.progress(1.0)
            st.success(f"✅ Database created successfully!")

            return vector_store, bm25_retriever, final_documents, embeddings, metadata

    except Exception as e:
        st.error(f"❌ Error creating database: {e}")
        return None

def process_answer_with_citations(answer, context_docs):
    """Add inline citations to the answer text"""
    import re

    citations = {}
    for i, doc in enumerate(context_docs):
        source_file = doc.metadata.get('source_file', 'Unknown')
        page_num = doc.metadata.get('page', 'Unknown')
        full_path = doc.metadata.get('full_path', '')

        citation_key = f"[{i+1}]"
        if full_path and os.path.exists(full_path):
            citation_link = f"""<sup><a href="file:///{full_path}" target="_blank" class="citation-link" title="Open {source_file} - Page {page_num}">[{i+1}]</a></sup>"""
        else:
            citation_link = f"""<sup class="citation-unavailable" title="{source_file} - Page {page_num}">[{i+1}]</sup>"""

        citations[citation_key] = citation_link

    processed_answer = answer

    patterns_to_cite = [
        r'(Item[‑\-]\s*No\.\s*[\d\-‑]+[\w\-‑]*)',
        r'(Rs\.?\s*[\d,]+\.?\d*)',
        r'(Chapter\s*[\d]+)',
        r'(\d+\.?\d*\s*%)',
        r'(Rule\s*[\d\.]+)',
        r'(\d+″\s*×\s*\d+″)',
        r'(Sq\s*[mft]+)',
        r'(MRS|CSR|Market[‑\-]Rate\s*System)',
    ]

    citation_usage = [False] * len(context_docs)

    for pattern in patterns_to_cite:
        matches = list(re.finditer(pattern, processed_answer, re.IGNORECASE))
        for match in reversed(matches):
            match_text = match.group(1).lower()
            best_citation_idx = None
            best_score = 0

            for i, doc in enumerate(context_docs):
                if citation_usage[i]:
                    continue

                doc_content = doc.page_content.lower()
                score = 0
                if 'item' in match_text and 'item' in doc_content:
                    score += 3
                if 'chapter' in match_text and 'chapter' in doc_content:
                    score += 3
                if 'rs' in match_text and 'rs' in doc_content:
                    score += 2
                if any(word in doc_content for word in match_text.split()):
                    score += 1

                if score > best_score:
                    best_score = score
                    best_citation_idx = i

            if best_citation_idx is not None and best_score > 0:
                citation_link = citations[f"[{best_citation_idx+1}]"]
                citation_usage[best_citation_idx] = True

                end_pos = match.end()
                processed_answer = (processed_answer[:end_pos] +
                                  citation_link +
                                  processed_answer[end_pos:])

    return processed_answer

def create_hybrid_retriever(vector_store, bm25_retriever):
    """Create hybrid retriever combining semantic and keyword search"""
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": NUM_RESULTS})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[SEMANTIC_WEIGHT, KEYWORD_WEIGHT]
    )

    return ensemble_retriever

# Page Configuration
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra Modern CSS with Complete Redesign
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Dark Modern Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
    }

    .main {
        background: transparent;
        padding: 0 !important;
    }

    .main > div {
        background: transparent;
        padding: 0 !important;
    }

    /* Modern Header Bar */
    .top-bar {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem 3rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }

    .logo-section {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .logo-icon {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .logo-text {
        color: white;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .status-badge {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1.25rem;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Container */
    .content-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 2rem 3rem 2rem;
    }

    /* Modern Cards */
    .modern-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }

    .modern-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 30px 80px rgba(99, 102, 241, 0.2);
    }

    .card-title {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .card-title-icon {
        font-size: 1.75rem;
    }

    /* Glassmorphism Panels */
    .glass-panel {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 0.5rem;
        gap: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.6) !important;
        font-weight: 600;
        padding: 0.875rem 1.5rem;
        border: none;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.9) !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
        color: white !important;
    }

    /* Text Colors */
    p, span, div, label {
        color: rgba(255, 255, 255, 0.9) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: 700;
    }

    /* Modern Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
        letter-spacing: 0.3px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, #a855f7, #6366f1);
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 0.875rem 1rem !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }

    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: rgba(255, 255, 255, 0.4) !important;
    }

    /* Select & Radio */
    .stSelectbox > div > div,
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    .stSelectbox label,
    .stRadio label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 500;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.5);
    }

    [data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.6) !important;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #6366f1 !important;
        font-weight: 800;
        font-size: 2rem;
    }

    /* Chat Messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .stChatMessage p,
    .stChatMessage div,
    .stChatMessage span {
        color: rgba(255, 255, 255, 0.9) !important;
    }

    /* Chat Input */
    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
    }

    [data-testid="stChatInput"] input {
        background: transparent !important;
        color: white !important;
        border: none !important;
    }

    /* Success/Info/Warning Boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border-left: 4px solid;
    }

    .stSuccess {
        border-color: #10b981;
    }

    .stInfo {
        border-color: #6366f1;
    }

    .stWarning {
        border-color: #f59e0b;
    }

    .stError {
        border-color: #ef4444;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600;
    }

    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 0 0 12px 12px !important;
    }

    /* Code Blocks */
    code, pre {
        background: rgba(0, 0, 0, 0.3) !important;
        color: #a855f7 !important;
        border-radius: 8px;
        padding: 0.25rem 0.5rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7);
    }

    /* Citations */
    .citation-link {
        background: rgba(99, 102, 241, 0.2);
        color: #a78bfa !important;
        padding: 2px 8px;
        border-radius: 6px;
        text-decoration: none !important;
        font-weight: 600;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .citation-link:hover {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white !important;
        transform: translateY(-1px);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #a855f7, #6366f1);
    }

    /* Answer Content */
    .answer-content {
        color: rgba(255, 255, 255, 0.9) !important;
        line-height: 1.8;
    }

    .answer-content h1,
    .answer-content h2,
    .answer-content h3 {
        color: #a78bfa !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .answer-content table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
    }

    .answer-content th,
    .answer-content td {
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.875rem 1rem;
        text-align: left;
        color: rgba(255, 255, 255, 0.9) !important;
    }

    .answer-content th {
        background: rgba(99, 102, 241, 0.2);
        font-weight: 600;
        color: white !important;
    }

    .answer-content ul,
    .answer-content ol {
        margin: 1rem 0;
        padding-left: 2rem;
    }

    .answer-content li {
        margin: 0.5rem 0;
        color: rgba(255, 255, 255, 0.9) !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Clear Streamlit cache on startup (force fresh scan)
st.cache_data.clear()
st.cache_resource.clear()

# Initialize Session State
if 'database_loaded' not in st.session_state:
    st.session_state.database_loaded = False
if 'llm_configured' not in st.session_state:
    st.session_state.llm_configured = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with All Controls
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
        <h2 style="color: white; margin: 0; font-size: 1.5rem;">🧠 DocuMind AI</h2>
        <p style="color: rgba(255,255,255,0.5); margin: 0.5rem 0 0 0; font-size: 0.875rem;">Document Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'scanned_databases' not in st.session_state:
        st.session_state.scanned_databases = []
    if 'found_pdfs' not in st.session_state:
        st.session_state.found_pdfs = []
    
    # Auto-scan for existing databases
    if not st.session_state.scanned_databases:
        databases = scan_folder_for_databases(".")
        st.session_state.scanned_databases = databases
    
    st.markdown("---")
    st.markdown("### 📂 Database Selection")
    
    # Show existing databases
    if st.session_state.scanned_databases:
        selected_db = st.selectbox(
            "Select Database",
            range(len(st.session_state.scanned_databases)),
            format_func=lambda x: st.session_state.scanned_databases[x]['name'],
            key="db_selector"
        )
        
        if st.button("🚀 Load Database", key="load_db", use_container_width=True):
            with st.spinner("Loading database..."):
                db_info = st.session_state.scanned_databases[selected_db]
                result = load_existing_database(db_info['path'], db_info['db_folder'])
                
                if result:
                    vector_store, bm25_retriever, documents, embeddings, metadata = result
                    st.session_state.vector_store = vector_store
                    st.session_state.bm25_retriever = bm25_retriever
                    st.session_state.documents = documents
                    st.session_state.embeddings = embeddings
                    st.session_state.metadata = metadata
                    st.session_state.database_loaded = True
                    
                    # Auto-initialize AI model with selected provider
                    selected_provider = st.session_state.get('selected_provider', 'DeepSeek')
                    selected_model_name = st.session_state.get('selected_model', 'deepseek-chat')
                    
                    api_key = None
                    if selected_provider == "DeepSeek":
                        api_key = deepseek_api_key
                    elif selected_provider == "ChatGPT (OpenAI)":
                        api_key = openai_api_key
                    elif selected_provider == "Groq":
                        api_key = groq_api_key
                    
                    if api_key:
                        llm = initialize_llm(selected_provider, selected_model_name, api_key)
                        if llm:
                            st.session_state.llm = llm
                            st.session_state.model_provider = selected_provider
                            st.session_state.selected_model = selected_model_name
                            st.session_state.llm_configured = True
                    
                    st.success("✅ Database & AI Ready!")
                    st.rerun()
    else:
        st.info("No existing databases found")
    
    st.markdown("---")
    st.markdown("### 🤖 AI Model Selection")
    
    model_provider = st.selectbox(
        "Select AI Provider",
        list(AVAILABLE_MODELS.keys()),
        key="model_provider_selector"
    )
    
    selected_model_display = st.selectbox(
        "Select Model",
        list(AVAILABLE_MODELS[model_provider].keys()),
        key="model_selector"
    )
    
    selected_model = AVAILABLE_MODELS[model_provider][selected_model_display]
    
    # Always update session state with current selection
    st.session_state.selected_provider = model_provider
    st.session_state.selected_model = selected_model
    
    # Initialize AI when model selection changes
    if st.session_state.database_loaded:
        # Get appropriate API key
        api_key = None
        if model_provider == "DeepSeek":
            api_key = deepseek_api_key
        elif model_provider == "ChatGPT (OpenAI)":
            api_key = openai_api_key
        elif model_provider == "Groq":
            api_key = groq_api_key
        
        # Initialize LLM with selected model
        if api_key:
            current_provider = st.session_state.get('model_provider')
            current_model = st.session_state.get('llm_model_name')
            
            # Only reinitialize if model changed
            if current_provider != model_provider or current_model != selected_model:
                with st.spinner(f"Initializing {model_provider}..."):
                    llm = initialize_llm(model_provider, selected_model, api_key)
                    if llm:
                        st.session_state.llm = llm
                        st.session_state.model_provider = model_provider
                        st.session_state.llm_model_name = selected_model
                        st.session_state.llm_configured = True
                        st.success(f"✅ {model_provider} Ready!")
                        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'txt', 'docx', 'doc', 'csv', 'json', 'md', 'markdown'],
        accept_multiple_files=True,
        key="doc_uploader",
        help="Supported: PDF, Word, Text, CSV, JSON, Markdown"
    )
    
    if uploaded_files:
        if st.button("🚀 Process & Load", key="process_load", use_container_width=True):
            with st.spinner("Processing files..."):
                import tempfile
                temp_dir = tempfile.mkdtemp()
                pdf_paths = []
                
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_paths.append(Path(temp_path))
                
                result = create_new_database(pdf_paths, "vector_databases")
                
                if result:
                    vector_store, bm25_retriever, documents, embeddings, metadata = result
                    st.session_state.vector_store = vector_store
                    st.session_state.bm25_retriever = bm25_retriever
                    st.session_state.documents = documents
                    st.session_state.embeddings = embeddings
                    st.session_state.metadata = metadata
                    st.session_state.database_loaded = True
                    st.session_state.scanned_databases = []
                    
                    # Auto-initialize AI model with selected provider
                    selected_provider = st.session_state.get('selected_provider', 'DeepSeek')
                    selected_model_name = st.session_state.get('selected_model', 'deepseek-chat')
                    
                    api_key = None
                    if selected_provider == "DeepSeek":
                        api_key = deepseek_api_key
                    elif selected_provider == "ChatGPT (OpenAI)":
                        api_key = openai_api_key
                    elif selected_provider == "Groq":
                        api_key = groq_api_key
                    
                    if api_key:
                        llm = initialize_llm(selected_provider, selected_model_name, api_key)
                        if llm:
                            st.session_state.llm = llm
                            st.session_state.model_provider = selected_provider
                            st.session_state.selected_model = selected_model_name
                            st.session_state.llm_configured = True
                    
                    st.success("✅ Files Processed & AI Ready!")
                    st.rerun()
    
    # Show loaded documents
    if st.session_state.database_loaded and 'metadata' in st.session_state:
        st.markdown("---")
        st.markdown("### 📄 Loaded Documents")
        
        metadata = st.session_state.metadata
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 12px; margin-top: 0.5rem;">
            <p style="color: rgba(255,255,255,0.6); margin: 0 0 0.5rem 0; font-size: 0.75rem;">DATABASE INFO</p>
            <p style="color: white; margin: 0; font-weight: 600;">📊 {metadata.get('total_pdfs', 0)} Documents</p>
            <p style="color: rgba(255,255,255,0.7); margin: 0.25rem 0 0 0; font-size: 0.875rem;">
                {metadata.get('total_chunks', 0):,} chunks indexed
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display files in expander
        pdf_files = metadata.get('pdf_files', [])
        with st.expander(f"📂 View Files ({len(pdf_files)})", expanded=False):
            for i, pdf_path in enumerate(pdf_files[:20], 1):
                file_name = os.path.basename(pdf_path)
                st.markdown(f"**{i}.** {file_name[:40]}{'...' if len(file_name) > 40 else ''}")
            if len(pdf_files) > 20:
                st.markdown(f"*+ {len(pdf_files) - 20} more files*")
    
    # Status indicators
    st.markdown("---")
    st.markdown("### ⚙️ Status")
    
    db_status = "🟢 Active" if st.session_state.database_loaded else "🔴 Not Loaded"
    
    if st.session_state.llm_configured:
        provider = st.session_state.get('model_provider', 'Unknown')
        ai_status = f"🟢 {provider}"
    else:
        ai_status = "🔴 Not Ready"
    
    st.markdown(f"""
    <div style="font-size: 0.875rem;">
        <p style="margin: 0.5rem 0; color: rgba(255,255,255,0.8);">
            <strong>Database:</strong> {db_status}
        </p>
        <p style="margin: 0.5rem 0; color: rgba(255,255,255,0.8);">
            <strong>AI Model:</strong> {ai_status}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Modern Header Bar
st.markdown("""
<div class="top-bar">
    <div class="logo-section">
        <div class="logo-icon">🧠</div>
        <div class="logo-text">DocuMind AI - Chat Interface</div>
    </div>
    <div class="status-badge">
        <span>●</span> System Ready
    </div>
</div>
""", unsafe_allow_html=True)

# Content Container
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# MAIN CHAT AREA
if not st.session_state.database_loaded or not st.session_state.llm_configured:
    st.markdown("""
    <div class="modern-card">
        <h3 style="color: white; text-align: center;">⚠️ Setup Required</h3>
        <p style="color: rgba(255,255,255,0.7); text-align: center; margin-top: 1rem;">
            👈 Please use the sidebar to load documents and configure the AI model.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="card-title"><span class="card-title-icon">💬</span>Chat with Your Documents</div>', unsafe_allow_html=True)

    # Create hybrid retriever
    hybrid_retriever = create_hybrid_retriever(
        st.session_state.vector_store,
        st.session_state.bm25_retriever
    )

    llm = st.session_state.llm

    prompt = ChatPromptTemplate.from_template("""
    You are an expert assistant analyzing documents. Answer the following question based only on the provided context.
    Be comprehensive, detailed, and include specific references.

    <context>
    {context}
    </context>

    Question: {input}

    Answer:
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(hybrid_retriever, document_chain)

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"], unsafe_allow_html=True)

                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Sources", expanded=False):
                        import pandas as pd
                        # Create simplified dataframe for display
                        display_sources = []
                        for s in message["sources"]:
                            display_sources.append({
                                "Citation": s.get("Citation", ""),
                                "Document": s.get("Document", "Unknown"),
                                "Page": str(s.get("Page", "Unknown"))
                            })
                        df = pd.DataFrame(display_sources)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                        # Show detailed source content
                        st.markdown("---")
                        st.markdown("**📖 Source Content Preview:**")
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**[{i+1}] {source.get('Document', 'Unknown')} - Page {source.get('Page', 'Unknown')}**")
                            if "content" in source:
                                preview = source["content"][:300] + "..." if len(source["content"]) > 300 else source["content"]
                                st.markdown(f'<div style="background: rgba(255,255,255,0.03); padding: 1rem; border-radius: 8px; margin: 0.5rem 0 1rem 0; color: rgba(255,255,255,0.7); line-height: 1.6;">{preview}</div>', unsafe_allow_html=True)
            else:
                st.markdown(message["content"])

    # Chat input at the bottom
    if prompt_input := st.chat_input("💭 Ask me anything about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt_input})

        # Generate response
        try:
            response = retrieval_chain.invoke({"input": prompt_input})
            answer = response["answer"]

            # Process answer with citations
            processed_answer = process_answer_with_citations(answer, response["context"])
            styled_answer = f'<div class="answer-content">{processed_answer}</div>'

            # Prepare source data
            import pandas as pd
            source_data = []
            for i, doc in enumerate(response["context"]):
                source_file = doc.metadata.get('source_file', 'Unknown')
                page_num = doc.metadata.get('page', 'Unknown')
                full_path = doc.metadata.get('full_path', '')
                source_data.append({
                    "Citation": f"[{i+1}]",
                    "Document": source_file,
                    "Page": str(page_num),
                    "content": doc.page_content,
                    "path": full_path
                })

            # Save to session with sources
            st.session_state.messages.append({
                "role": "assistant",
                "content": styled_answer,
                "sources": source_data
            })

            # Force rerun to display the new message
            st.rerun()

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.rerun()

    # Action buttons
    st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", key="reset_config"):
            st.session_state.database_loaded = False
            st.session_state.llm_configured = False
            st.session_state.messages = []
            st.session_state.scanned_databases = []
            st.session_state.found_pdfs = []
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Footer with Copyright
st.markdown("""
<div style="
    text-align: center;
    padding: 2rem 0 1rem 0;
    margin-top: 3rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);">
    <p style="color: rgba(255, 255, 255, 0.5); font-size: 0.875rem; margin: 0;">
        © 2025 DocuMind AI. All rights reserved. | Powered by Advanced RAG Technology
    </p>
</div>
""", unsafe_allow_html=True)
