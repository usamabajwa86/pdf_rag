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
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Load environment
load_dotenv()

# Fixed Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
NUM_RESULTS = 8

# Available Models
AVAILABLE_MODELS = {
    "Local Models (Ollama)": {
        "llama3.2:3b": "llama3.2:3b",
        "llama3.2:1b": "llama3.2:1b",
        "llama3.1:8b": "llama3.1:8b",
        "mistral:7b": "mistral:7b",
        "codellama:7b": "codellama:7b"
    },
    "API Models (Groq)": {
        "Llama 3.1 8B": "llama-3.1-8b-instant",
        "Llama 3.1 70B": "llama-3.1-70b-versatile",
        "Mixtral 8x7B": "mixtral-8x7b-32768",
        "Gemma 2 9B": "gemma2-9b-it"
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

def initialize_llm(model_provider, model_name, groq_api_key=None):
    """Initialize the appropriate LLM based on provider"""
    try:
        if model_provider == "Local Models (Ollama)":
            if not check_ollama_connection():
                st.error("❌ Ollama is not running! Please start Ollama service.")
                return None
            
            # Check if model is available
            available_models = get_available_ollama_models()
            if model_name not in available_models:
                st.warning(f"⚠️ Model '{model_name}' not found in Ollama. Available models: {available_models}")
                st.info("To install the model, run in terminal: `ollama pull " + model_name + "`")
                return None
            
            return Ollama(
                model=model_name,
                base_url="http://localhost:11434",
                temperature=0.1
            )
        
        elif model_provider == "API Models (Groq)":
            if not groq_api_key:
                st.error("❌ GROQ_API_KEY not found! Please add it to your .env file for API models.")
                return None
            
            return ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=0
            )
        
        return None
    except Exception as e:
        st.error(f"❌ Error initializing LLM: {e}")
        return None

# Check for API key (optional now since we support local models)
groq_api_key = os.environ.get('GROQ_API_KEY')

def clean_text(text):
    """Clean and validate text content"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove problematic characters and normalize
    cleaned = text.encode('utf-8', errors='ignore').decode('utf-8')
    cleaned = cleaned.strip()
    
    # Remove excessive whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def get_common_folders():
    """Get list of common folders for user selection"""
    common_folders = ["Data"]  # Default
    
    try:
        # Add user's common folders if they exist
        user_folders = [
            ("Documents", os.path.expanduser("~/Documents")),
            ("Downloads", os.path.expanduser("~/Downloads")), 
            ("Desktop", os.path.expanduser("~/Desktop")),
            ("Pictures", os.path.expanduser("~/Pictures"))
        ]
        
        for name, path in user_folders:
            if os.path.exists(path):
                common_folders.append(f"{name} ({path})")
        
        # Add current drive common locations
        current_drive = os.path.splitdrive(os.getcwd())[0]
        drive_folders = [
            f"{current_drive}/Data",
            f"{current_drive}/Documents", 
            f"{current_drive}/PDFs"
        ]
        
        for folder in drive_folders:
            if os.path.exists(folder):
                common_folders.append(folder)
                
    except Exception:
        pass  # If any error, just use defaults
    
    common_folders.append("Custom Path...")
    return common_folders

def get_available_databases():
    """Get list of available vector databases"""
    databases = []
    
    # Check current directory for vector_databases folder
    current_db_folder = "vector_databases"
    if os.path.exists(current_db_folder):
        for item in os.listdir(current_db_folder):
            if item.startswith('db_'):
                db_path = os.path.join(current_db_folder, item)
                metadata_path = os.path.join(db_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        databases.append({
                            'name': f"Local DB - {metadata.get('embedding_model', 'Unknown')} ({metadata.get('total_pdfs', 0)} PDFs)",
                            'path': current_db_folder,
                            'metadata': metadata,
                            'db_folder': item
                        })
                    except:
                        continue
    
    return databases

def scan_folder_for_databases(folder_path):
    """Scan a folder for vector databases"""
    databases = []
    try:
        if not os.path.exists(folder_path):
            return databases
            
        # Look for vector_databases subfolder
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
                                'name': f"{metadata.get('embedding_model', 'Unknown')} - {metadata.get('total_pdfs', 0)} PDFs - Created: {metadata.get('created_at', 'Unknown')}",
                                'path': vector_db_path,
                                'metadata': metadata,
                                'db_folder': item
                            })
                        except:
                            continue
        
        # Also look for direct db_ folders in the selected folder
        for item in os.listdir(folder_path):
            if item.startswith('db_'):
                db_path = os.path.join(folder_path, item)
                metadata_path = os.path.join(db_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        databases.append({
                            'name': f"Direct DB - {metadata.get('embedding_model', 'Unknown')} - {metadata.get('total_pdfs', 0)} PDFs",
                            'path': folder_path,
                            'metadata': metadata,
                            'db_folder': item
                        })
                    except:
                        continue
                        
    except Exception as e:
        st.error(f"Error scanning folder: {e}")
    
    return databases

def load_existing_database(db_path, db_folder):
    """Load an existing vector database"""
    try:
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{EMBEDDING_MODEL}",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Construct paths
        base_path = os.path.join(db_path, db_folder)
        faiss_path = os.path.join(base_path, "faiss_index")
        bm25_path = os.path.join(base_path, "bm25_retriever.pkl")
        documents_path = os.path.join(base_path, "documents.pkl")
        metadata_path = os.path.join(base_path, "metadata.json")
        
        # Load FAISS vector store
        vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        
        # Load BM25 retriever
        with open(bm25_path, 'rb') as f:
            bm25_retriever = pickle.load(f)
        
        # Load documents
        with open(documents_path, 'rb') as f:
            documents = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return vector_store, bm25_retriever, documents, embeddings, metadata
        
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

def get_all_pdfs(folder_path):
    """Get all PDFs from specified folder and subfolders"""
    pdf_patterns = [
        os.path.join(folder_path, "**/*.pdf"),
        os.path.join(folder_path, "*.pdf")
    ]
    
    all_pdfs = []
    for pattern in pdf_patterns:
        all_pdfs.extend(glob.glob(pattern, recursive=True))
    
    return [Path(pdf) for pdf in set(all_pdfs) if os.path.exists(pdf)]

def create_new_database(pdf_files, save_folder):
    """Create new vector database from PDF files"""
    try:
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=f"sentence-transformers/{EMBEDDING_MODEL}",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Process PDFs
        with st.spinner(f"🔄 Processing {len(pdf_files)} PDFs..."):
            documents = []
            progress_bar = st.progress(0)
            
            # Load all PDFs
            for i, pdf_path in enumerate(pdf_files):
                try:
                    st.text(f"📄 Loading {pdf_path.name} ({i+1}/{len(pdf_files)})...")
                    loader = PyPDFLoader(str(pdf_path))
                    pdf_docs = loader.load()
                    
                    # Clean and validate document content
                    valid_docs = []
                    for doc in pdf_docs:
                        cleaned_content = clean_text(doc.page_content)
                        if len(cleaned_content) > 20:
                            doc.page_content = cleaned_content
                            doc.metadata['source_file'] = pdf_path.name
                            doc.metadata['full_path'] = str(pdf_path)
                            valid_docs.append(doc)
                    
                    documents.extend(valid_docs)
                    progress_bar.progress((i + 1) / len(pdf_files))
                    
                except Exception as e:
                    st.warning(f"⚠️ Failed to load {pdf_path.name}: {str(e)}")
            
            # Split documents
            st.text("✂️ Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            
            final_documents = text_splitter.split_documents(documents)
            
            # Create vector database
            st.text("🔢 Creating vector database...")
            vector_store = FAISS.from_documents(final_documents, embeddings)
            
            # Create BM25 retriever
            st.text("🔍 Creating keyword search index...")
            bm25_retriever = BM25Retriever.from_documents(final_documents)
            bm25_retriever.k = NUM_RESULTS
            
            # Save database
            st.text("💾 Saving database...")
            config_str = f"{sorted([str(p) for p in pdf_files])}_{EMBEDDING_MODEL}_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
            config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
            
            # Create save directory
            os.makedirs(save_folder, exist_ok=True)
            db_folder = f"db_{config_hash}"
            db_path = os.path.join(save_folder, db_folder)
            os.makedirs(db_path, exist_ok=True)
            
            # Save files
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
                'total_pdfs': len(pdf_files),
                'total_chunks': len(final_documents),
                'pdf_files': [str(p) for p in pdf_files],
                'config_hash': config_hash
            }
            
            with open(os.path.join(db_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            progress_bar.progress(1.0)
            st.success(f"✅ Database created successfully! Saved to {db_path}")
            
            return vector_store, bm25_retriever, final_documents, embeddings, metadata
            
    except Exception as e:
        st.error(f"❌ Error creating database: {e}")
        return None

def create_hybrid_retriever(vector_store, bm25_retriever):
    """Create hybrid retriever combining semantic and keyword search"""
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": NUM_RESULTS})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[SEMANTIC_WEIGHT, KEYWORD_WEIGHT]
    )
    
    return ensemble_retriever

# Streamlit App
st.set_page_config(
    page_title="Advanced Agentic RAG with Hybrid Search Model",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optimized Robot Theme CSS
st.markdown("""
<style>
    .main > div { padding: 2rem 0; background: linear-gradient(135deg, #f8f9fa 0%, #e8f5e8 100%); }
    .main-header { background: linear-gradient(135deg, #4CAF50 0%, #20B2AA 50%, #00CED1 100%); padding: 2rem 1rem; border-radius: 25px; margin-bottom: 2rem; box-shadow: 0 15px 50px rgba(76, 175, 80, 0.4); text-align: center; border: 4px solid #FF8C00; position: relative; overflow: hidden; }
    .main-header::before { content: ''; position: absolute; top: -2px; left: -2px; right: -2px; bottom: -2px; background: linear-gradient(45deg, #FF8C00, #FFA726, #00CED1, #20B2AA); border-radius: 25px; z-index: -1; animation: borderGlow 3s ease-in-out infinite alternate; }
    @keyframes borderGlow { 0% { filter: blur(5px); opacity: 0.8; } 100% { filter: blur(10px); opacity: 1; } }
    .main-title { color: white; font-size: 3.2rem; font-weight: 800; margin-bottom: 0.5rem; text-shadow: 3px 3px 6px rgba(0,0,0,0.4); letter-spacing: 1px; }
    .sub-title { color: #f8f9fa; font-size: 1.6rem; font-weight: 500; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .description { color: #e9ecef; font-size: 1.2rem; font-style: italic; opacity: 0.95; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }
    .option-card { background: linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%); padding: 2.5rem; border-radius: 25px; box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15); border: 3px solid #B0BEC5; transition: all 0.4s ease; margin-bottom: 1.5rem; position: relative; overflow: hidden; }
    .option-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 6px; background: linear-gradient(90deg, #FF8C00, #FFA726, #00CED1); border-radius: 25px 25px 0 0; }
    .option-card:hover { transform: translateY(-10px) scale(1.02); box-shadow: 0 20px 45px rgba(76, 175, 80, 0.25); border-color: #4CAF50; }
    .card-header { color: #4CAF50; font-size: 1.8rem; font-weight: 700; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.8rem; }
    .card-description { color: #757575; font-size: 1.1rem; margin-bottom: 1.5rem; line-height: 1.7; }
    .success-box, .info-box, .warning-box { color: white; padding: 1.5rem; border-radius: 20px; margin: 1.5rem 0; font-weight: 500; border-left: 6px solid #FF8C00; }
    .success-box { background: linear-gradient(135deg, #4CAF50, #66BB6A); box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4); }
    .info-box { background: linear-gradient(135deg, #00CED1, #20B2AA); box-shadow: 0 8px 25px rgba(32, 178, 170, 0.4); }
    .warning-box { background: linear-gradient(135deg, #FF8C00, #FFA726); box-shadow: 0 8px 25px rgba(255, 140, 0, 0.4); border-left-color: #4CAF50; }
    .stButton > button { background: linear-gradient(135deg, #4CAF50 0%, #66BB6A 50%, #20B2AA 100%); color: white; border: 3px solid #FF8C00; border-radius: 20px; padding: 1rem 3rem; font-weight: 700; transition: all 0.3s ease; box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4); text-transform: uppercase; letter-spacing: 1.5px; }
    .stButton > button:hover { transform: translateY(-4px); box-shadow: 0 12px 35px rgba(76, 175, 80, 0.6); border-color: #FFA726; background: linear-gradient(135deg, #66BB6A 0%, #4CAF50 50%, #00CED1 100%); }
    [data-testid="metric-container"] { background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); border: 3px solid #B0BEC5; padding: 1.5rem; border-radius: 20px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12); position: relative; transition: all 0.3s ease; }
    [data-testid="metric-container"]:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); }
    [data-testid="metric-container"]::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, #4CAF50, #00CED1, #FF8C00); border-radius: 20px 20px 0 0; }
    .stTextInput > div > div > input { border: 3px solid #B0BEC5; border-radius: 15px; transition: all 0.3s ease; padding: 0.8rem; }
    .stTextInput > div > div > input:focus { border-color: #4CAF50; box-shadow: 0 0 15px rgba(76, 175, 80, 0.4); }
    .stRadio > div, .stSelectbox > div > div, .row-widget.stRadio > div { background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); border: 2px solid #B0BEC5; border-radius: 15px; transition: all 0.3s ease; padding: 1rem; }
    .stRadio > div:hover, .stSelectbox > div > div:hover { border-color: #4CAF50; box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2); }
    .stRadio label, .stRadio > div > label > div, .stRadio div[role="radiogroup"] label, .stSelectbox label, .stSelectbox > div > div > div { color: #2c3e50 !important; font-weight: 500 !important; }
    .stChatMessage { background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); border-radius: 25px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12); border-left: 5px solid #4CAF50; }
    .robot-eyes { animation: robotGlow 2s ease-in-out infinite alternate; }
    @keyframes robotGlow { from { filter: drop-shadow(0 0 10px #00CED1); transform: scale(1); } to { filter: drop-shadow(0 0 20px #20B2AA); transform: scale(1.05); } }
</style>
""", unsafe_allow_html=True)

# Beautiful Header with Robot Logo
logo_path = "assets/company_logo.png"
if os.path.exists(logo_path):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.image(logo_path, width=200)
        st.markdown("""
            <div class="main-title">Advanced Agentic RAG</div>
            <div class="sub-title">with Hybrid Search Model</div>
            <div class="description">Intelligent Multi-Database RAG System</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="main-header">
        <div class="robot-eyes" style="font-size: 5rem; margin-bottom: 1rem;">🤖</div>
        <div class="main-title">Advanced Agentic RAG</div>
        <div class="sub-title">with Hybrid Search Model</div>
        <div class="description">Intelligent Multi-Database RAG System</div>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'mode_selected' not in st.session_state:
    st.session_state.mode_selected = False
if 'database_loaded' not in st.session_state:
    st.session_state.database_loaded = False
if 'llm_configured' not in st.session_state:
    st.session_state.llm_configured = False

# Mode selection screen
if not st.session_state.mode_selected:
    st.markdown('<div style="text-align: center; color: #4CAF50; font-size: 2.5rem; font-weight: 700; margin: 2rem 0;">🚀 Choose Your Operation Mode</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="option-card">
            <div class="card-header">
                📂 Load Existing Database
            </div>
            <div class="card-description">
                Instantly access your pre-built vector databases for lightning-fast search and retrieval. Perfect for continuing previous work or switching between different document collections.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Database Location Selection
        st.write("**🔍 Database Search Method:**")
        db_search_option = st.radio(
            "Choose how to find databases:",
            ["Current Directory", "Browse Locations", "Custom Path"],
            horizontal=True
        )
        
        external_folder = None
        all_found_dbs = []
        
        if db_search_option == "Current Directory":
            available_dbs = get_available_databases()
            if available_dbs:
                st.markdown(f"""
                <div class="success-box">
                    <strong>🎉 Found {len(available_dbs)} databases in current directory!</strong><br>
                    Ready to load and start searching immediately.
                </div>
                """, unsafe_allow_html=True)
                for db in available_dbs:
                    st.write(f"• {db['name']}")
                all_found_dbs = available_dbs
            else:
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ No databases found in current directory</strong><br>
                    Try browsing other locations or create a new database.
                </div>
                """, unsafe_allow_html=True)
        
        elif db_search_option == "Browse Locations":
            st.write("**📁 Select from common locations:**")
            common_db_locations = [
                "./vector_databases",
                "../vector_databases", 
                "E:/Updated-Langchain-main/Advance RAG_Hybrid Search/vector_databases",
                "E:/Updated-Langchain-main/PDF_Search_Clean/vector_databases",
                "Custom Path..."
            ]
            
            selected_location = st.selectbox("Choose a location to scan:", common_db_locations)
            
            if selected_location == "Custom Path...":
                external_folder = st.text_input("📁 Enter custom database folder path:", placeholder="E:/path/to/your/database/folder")
            else:
                external_folder = selected_location
                
        else:  # Custom Path
            external_folder = st.text_input("📁 Database folder path:", placeholder="E:/path/to/your/database/folder")
        
        # Scan external folder if specified
        if external_folder:
            if os.path.exists(external_folder):
                external_dbs = scan_folder_for_databases(external_folder)
                if external_dbs:
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>🎉 Found {len(external_dbs)} databases in '{external_folder}'!</strong><br>
                        Multiple AI-powered databases ready for deployment.
                    </div>
                    """, unsafe_allow_html=True)
                    for db in external_dbs:
                        st.write(f"• {db['name']}")
                    all_found_dbs.extend(external_dbs)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                        <strong>⚠️ No databases found in '{external_folder}'</strong><br>
                        Please check the path or try a different location.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f44336, #ef5350); color: white; padding: 1.5rem; border-radius: 20px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(244, 67, 54, 0.4);">
                    <strong>❌ Folder '{external_folder}' does not exist</strong><br>
                    Please verify the path and try again.
                </div>
                """, unsafe_allow_html=True)
        
        # Show total databases found
        if all_found_dbs:
            st.markdown(f"""
            <div class="info-box">
                <strong>📊 Total: {len(all_found_dbs)} databases available for loading</strong><br>
                AI-powered knowledge bases ready for intelligent search.
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🔄 Load Existing Database", type="primary", use_container_width=True):
            if all_found_dbs:
                st.session_state.mode = "load_existing"
                st.session_state.mode_selected = True
                st.session_state.external_folder = external_folder if external_folder else None
                st.session_state.all_databases = all_found_dbs
                st.rerun()
            else:
                st.error("No databases found. Please check your paths or create a new database.")
    
    with col2:
        st.markdown("""
        <div class="option-card">
            <div class="card-header">
                📄 Create New Database
            </div>
            <div class="card-description">
                Transform your PDF documents into a powerful searchable knowledge base using advanced AI embeddings and hybrid search technology.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # PDF Folder Selection
        st.write("**📁 Select PDF Source Folder:**")
        pdf_folder_option = st.radio(
            "Choose PDF folder selection method:",
            ["Manual Path", "Browse Common Locations"],
            horizontal=True
        )
        
        if pdf_folder_option == "Manual Path":
            pdf_folder = st.text_input("📁 PDF folder path:", placeholder="Data", value="Data")
        else:
            # Get dynamic common folder options
            common_folders = get_common_folders()
            selected_common = st.selectbox("Choose a common folder:", common_folders)
            
            if selected_common == "Custom Path...":
                pdf_folder = st.text_input("📁 Enter custom PDF folder path:", placeholder="E:/Your/PDF/Folder")
            elif "(" in selected_common and ")" in selected_common:
                # Extract path from "Name (path)" format
                pdf_folder = selected_common.split("(")[1].rstrip(")")
            else:
                pdf_folder = selected_common
        
        # Save Location Selection
        st.write("**💾 Select Database Save Location:**")
        save_folder_option = st.radio(
            "Choose save location method:",
            ["Default Location", "Custom Location"],
            horizontal=True
        )
        
        if save_folder_option == "Default Location":
            save_folder = "vector_databases"
        else:
            save_folder = st.text_input("💾 Custom save path:", placeholder="E:/Your/Database/Folder", value="vector_databases")
        
        # Display folder info with beautiful styling
        if pdf_folder:
            if os.path.exists(pdf_folder):
                pdf_files = get_all_pdfs(pdf_folder)
                if pdf_files:
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>🎉 Found {len(pdf_files)} PDF files in '{pdf_folder}'!</strong><br>
                        Ready to process into an intelligent knowledge base.
                    </div>
                    """, unsafe_allow_html=True)
                    with st.expander("📄 View PDF Collection", expanded=False):
                        st.markdown("**📚 Your Document Library:**")
                        for pdf in pdf_files[:10]:  # Show first 10
                            st.write(f"📄 {pdf.name}")
                        if len(pdf_files) > 10:
                            st.info(f"... and {len(pdf_files) - 10} more documents")
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                        <strong>⚠️ No PDF files found in '{pdf_folder}'</strong><br>
                        Please check the folder path or select a different location.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f44336, #ef5350); color: white; padding: 1.5rem; border-radius: 20px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(244, 67, 54, 0.4);">
                    <strong>❌ Folder '{pdf_folder}' does not exist</strong><br>
                    Please verify the path and try again.
                </div>
                """, unsafe_allow_html=True)
        
        if st.button("🆕 Create New Database", type="primary", use_container_width=True):
            if pdf_folder and save_folder and os.path.exists(pdf_folder):
                pdf_files = get_all_pdfs(pdf_folder)
                if pdf_files:
                    st.session_state.mode = "create_new"
                    st.session_state.mode_selected = True
                    st.session_state.pdf_folder = pdf_folder
                    st.session_state.save_folder = save_folder
                    st.rerun()
                else:
                    st.error("No PDF files found in the selected folder")
            else:
                st.error("Please specify valid PDF folder and save location")

# Database loading/creation screen
elif st.session_state.mode_selected and not st.session_state.database_loaded:
    
    if st.session_state.mode == "load_existing":
        st.markdown("""
        <div style="text-align: center; color: #4CAF50; font-size: 2.8rem; font-weight: 700; margin: 2rem 0;">
            📂 Load Existing Database
        </div>
        """, unsafe_allow_html=True)
        
        # Use pre-found databases from session state
        all_databases = st.session_state.get('all_databases', [])
        
        if all_databases:
            st.markdown(f"""
            <div class="success-box">
                <strong>🎯 {len(all_databases)} AI-powered databases available for loading!</strong><br>
                Select your preferred knowledge base to begin intelligent search.
            </div>
            """, unsafe_allow_html=True)
            
            # Show database details in expandable sections
            selected_db = st.selectbox(
                "Select database to load:",
                range(len(all_databases)),
                format_func=lambda x: all_databases[x]['name']
            )
            
            # Show detailed info about selected database
            if selected_db is not None:
                db_info = all_databases[selected_db]
                metadata = db_info['metadata']
                
                with st.expander("📋 Database Details", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 PDF Files", metadata.get('total_pdfs', 'Unknown'), help="Total documents in database")
                    with col2:
                        st.metric("📝 Text Chunks", f"{metadata.get('total_chunks', 'Unknown'):,}", help="Searchable text segments")
                    with col3:
                        st.metric("🔤 AI Model", metadata.get('embedding_model', 'Unknown'), help="Embedding model used")
                    
                    st.write(f"**📅 Created:** {metadata.get('created_at', 'Unknown')}")
                    st.write(f"**📁 Location:** {db_info['path']}")
                    
                    if 'pdf_files' in metadata:
                        st.write(f"**📚 Document Collection ({len(metadata['pdf_files'])}):**")
                        for i, pdf_file in enumerate(metadata['pdf_files'][:5]):
                            st.write(f"📄 {os.path.basename(pdf_file)}")
                        if len(metadata['pdf_files']) > 5:
                            st.info(f"... and {len(metadata['pdf_files']) - 5} more documents")
            
            if st.button("🔄 Load Selected Database", type="primary", use_container_width=True):
                with st.spinner("🤖 Activating AI Assistant..."):
                    db_info = all_databases[selected_db]
                    result = load_existing_database(db_info['path'], db_info['db_folder'])
                    
                    if result:
                        vector_store, bm25_retriever, documents, embeddings, metadata = result
                        
                        # Store in session state
                        st.session_state.vector_store = vector_store
                        st.session_state.bm25_retriever = bm25_retriever
                        st.session_state.documents = documents
                        st.session_state.embeddings = embeddings
                        st.session_state.metadata = metadata
                        st.session_state.database_loaded = True
                        
                        st.success(f"✅ AI Assistant activated successfully!")
                        st.info(f"📊 Loaded: {metadata['total_pdfs']} PDFs, {metadata['total_chunks']:,} chunks")
                        time.sleep(1)  # Brief pause to show success message
                        st.rerun()
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ No databases found</strong><br>
                Please go back and check your paths or create a new database.
            </div>
            """, unsafe_allow_html=True)
    
    elif st.session_state.mode == "create_new":
        st.markdown("""
        <div style="text-align: center; color: #4CAF50; font-size: 2.8rem; font-weight: 700; margin: 2rem 0;">
            🆕 Create New Database
        </div>
        """, unsafe_allow_html=True)
        
        pdf_files = get_all_pdfs(st.session_state.pdf_folder)
        
        if pdf_files:
            st.markdown(f"""
            <div class="info-box">
                <strong>🚀 Ready to process {len(pdf_files)} PDF files</strong><br>
                Transform your documents into an intelligent knowledge base.
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Start AI Processing", type="primary", use_container_width=True):
                result = create_new_database(pdf_files, st.session_state.save_folder)
                
                if result:
                    vector_store, bm25_retriever, documents, embeddings, metadata = result
                    
                    # Store in session state
                    st.session_state.vector_store = vector_store
                    st.session_state.bm25_retriever = bm25_retriever
                    st.session_state.documents = documents
                    st.session_state.embeddings = embeddings
                    st.session_state.metadata = metadata
                    st.session_state.database_loaded = True
                    
                    st.rerun()
        else:
            st.error(f"No PDF files found in '{st.session_state.pdf_folder}'")
    
    # Back button
    if st.button("⬅️ Back to Mode Selection", use_container_width=True):
        st.session_state.mode_selected = False
        st.session_state.database_loaded = False
        st.rerun()

# Main application screen
elif st.session_state.database_loaded and not st.session_state.llm_configured:
    
    st.markdown("""
    <div style="text-align: center; color: #4CAF50; font-size: 2.8rem; font-weight: 700; margin: 2rem 0;">
        🤖 Choose Your AI Model
    </div>
    """, unsafe_allow_html=True)
    
    # Display database info
    metadata = st.session_state.metadata
    st.markdown(f"""
    <div class="info-box">
        <strong>📊 Database Loaded Successfully!</strong><br>
        {metadata['total_pdfs']} PDFs • {metadata['total_chunks']:,} chunks • {metadata['embedding_model']} embeddings
    </div>
    """, unsafe_allow_html=True)
    
    # Model Provider Selection
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="option-card">
            <div class="card-header">
                🏠 Local Models (Ollama)
            </div>
            <div class="card-description">
                Use local AI models running on your machine with Ollama. Completely private, no internet required for inference, and no API costs.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Check Ollama status
        ollama_running = check_ollama_connection()
        if ollama_running:
            available_ollama_models = get_available_ollama_models()
            if available_ollama_models:
                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ Ollama is running!</strong><br>
                    Found {len(available_ollama_models)} models available locally.
                </div>
                """, unsafe_allow_html=True)
                
                # Show available models
                st.write("**🤖 Available Local Models:**")
                for model in available_ollama_models:
                    model_info = "🔥 Recommended" if model == "llama3.2:3b" else ""
                    st.write(f"• {model} {model_info}")
            else:
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Ollama is running but no models found</strong><br>
                    Install models using: <code>ollama pull llama3.2:3b</code>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f44336, #ef5350); color: white; padding: 1.5rem; border-radius: 20px; margin: 1rem 0; box-shadow: 0 8px 25px rgba(244, 67, 54, 0.4);">
                <strong>❌ Ollama is not running</strong><br>
                Please start Ollama service and install models:<br>
                <code>ollama serve</code><br>
                <code>ollama pull llama3.2:3b</code>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="option-card">
            <div class="card-header">
                ☁️ API Models (Groq)
            </div>
            <div class="card-description">
                Use powerful cloud-based AI models via Groq API. Fast inference with latest models, but requires internet and API key.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Check API key
        if groq_api_key:
            st.markdown("""
            <div class="success-box">
                <strong>✅ Groq API Key found!</strong><br>
                Ready to use cloud-based models with fast inference.
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**☁️ Available API Models:**")
            for model_name in AVAILABLE_MODELS["API Models (Groq)"]:
                st.write(f"• {model_name}")
        else:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Groq API Key not found</strong><br>
                Add <code>GROQ_API_KEY</code> to your .env file to use API models.
            </div>
            """, unsafe_allow_html=True)
    
    # Model Selection Form
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4CAF50 0%, #20B2AA 50%, #00CED1 100%); padding: 1.5rem; border-radius: 25px; margin: 2rem 0; text-align: center; border: 3px solid #FF8C00; box-shadow: 0 10px 30px rgba(76, 175, 80, 0.4);">
        <h3 style="color: white; margin: 0; font-weight: 700;">🎯 Select Your AI Model</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Provider selection
    model_provider = st.selectbox(
        "**🏭 Choose Model Provider:**",
        ["Local Models (Ollama)", "API Models (Groq)"],
        help="Local models run on your machine, API models run in the cloud"
    )
    
    # Model selection based on provider
    if model_provider == "Local Models (Ollama)":
        if ollama_running and available_ollama_models:
            # Show only available models
            available_model_options = {}
            for model in available_ollama_models:
                display_name = f"{model}"
                if model == "llama3.2:3b":
                    display_name += " 🔥 (Recommended)"
                available_model_options[display_name] = model
            
            selected_model_display = st.selectbox(
                "**🤖 Select Local Model:**",
                list(available_model_options.keys()),
                help="Choose from your installed Ollama models"
            )
            selected_model = available_model_options[selected_model_display]
        else:
            st.error("❌ No local models available. Please install Ollama and pull models.")
            selected_model = None
    
    else:  # API Models
        if groq_api_key:
            selected_model_display = st.selectbox(
                "**☁️ Select API Model:**",
                list(AVAILABLE_MODELS["API Models (Groq)"].keys()),
                help="Choose from available Groq API models"
            )
            selected_model = AVAILABLE_MODELS["API Models (Groq)"][selected_model_display]
        else:
            st.error("❌ No API key found. Please add GROQ_API_KEY to your .env file.")
            selected_model = None
    
    # Model configuration info
    if selected_model:
        if model_provider == "Local Models (Ollama)":
            st.markdown(f"""
            <div class="success-box">
                <strong>🏠 Local Model Selected: {selected_model}</strong><br>
                ✅ Private • ✅ No API costs • ✅ Works offline
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-box">
                <strong>☁️ API Model Selected: {selected_model_display}</strong><br>
                ⚡ Fast inference • 🌐 Requires internet • 💳 Uses API credits
            </div>
            """, unsafe_allow_html=True)
        
        # Initialize button
        if st.button("🚀 Initialize AI Assistant", type="primary", use_container_width=True):
            with st.spinner("🤖 Initializing AI Assistant..."):
                llm = initialize_llm(model_provider, selected_model, groq_api_key)
                
                if llm:
                    st.session_state.llm = llm
                    st.session_state.model_provider = model_provider
                    st.session_state.selected_model = selected_model
                    st.session_state.llm_configured = True
                    
                    st.success(f"✅ AI Assistant initialized successfully with {selected_model}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize AI Assistant. Please check your configuration.")
    
    # Back button
    if st.button("⬅️ Back to Database Selection", use_container_width=True):
        st.session_state.database_loaded = False
        st.rerun()

# Chat interface screen
elif st.session_state.database_loaded and st.session_state.llm_configured:
    
    # Create hybrid retriever
    hybrid_retriever = create_hybrid_retriever(
        st.session_state.vector_store, 
        st.session_state.bm25_retriever
    )
    
    # Use the configured LLM from session state
    llm = st.session_state.llm
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context. Be comprehensive and detailed.

    <context>
    {context}
    </context>

    Question: {input}

    Answer:
    """)
    
    # Create chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(hybrid_retriever, document_chain)
    
    # Display database info with beautiful metrics
    metadata = st.session_state.metadata
    
    st.markdown("""
    <div style="text-align: center; color: #4CAF50; font-size: 2.5rem; font-weight: 700; margin: 1rem 0;">
        🤖 Your AI Assistant is Ready!
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 PDF Files", metadata['total_pdfs'], help="Total documents in database")
    with col2:
        st.metric("📝 Text Chunks", f"{metadata['total_chunks']:,}", help="Searchable text segments")
    with col3:
        st.metric("🔤 AI Model", metadata['embedding_model'].split('-')[1], help="Embedding model used")
    with col4:
        st.metric("🔍 Search Type", "Hybrid", help="Semantic + Keyword search")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4CAF50 0%, #20B2AA 50%, #00CED1 100%); padding: 1.5rem; border-radius: 25px; margin: 2rem 0 1rem 0; text-align: center; border: 3px solid #FF8C00; box-shadow: 0 10px 30px rgba(76, 175, 80, 0.4);">
        <h2 style="color: white; margin: 0; font-weight: 700; font-size: 2rem;">💬 Ask Your Questions</h2>
        <p style="color: #f8f9fa; margin: 0.5rem 0 0 0; opacity: 0.95; font-size: 1.1rem;">Start an intelligent conversation with your documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt_input := st.chat_input("Ask a question about your PDFs..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤖 AI Processing your question..."):
                try:
                    response = retrieval_chain.invoke({"input": prompt_input})
                    answer = response["answer"]
                    
                    st.markdown(answer)
                    
                    # Show sources
                    with st.expander("📚 Sources Used"):
                        st.info(f"🔍 **Found {len(response['context'])} relevant sections**")
                        for i, doc in enumerate(response["context"]):
                            source_file = doc.metadata.get('source_file', 'Unknown')
                            st.write(f"**📄 Source {i+1}:** {source_file}")
                            st.write(f"**📝 Content:** {doc.page_content[:300]}...")
                            if i < len(response['context']) - 1:
                                st.divider()
                    
                    # Add assistant message
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #00CED1 100%); padding: 1.5rem; border-radius: 25px; margin-bottom: 2rem; text-align: center; border: 3px solid #FF8C00; box-shadow: 0 10px 30px rgba(76, 175, 80, 0.4);">
            <h2 style="color: white; margin: 0; font-weight: 700;">🔧 Control Panel</h2>
            <p style="color: #f8f9fa; margin: 0.5rem 0 0 0; opacity: 0.95;">Manage AI assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); padding: 1.5rem; border-radius: 25px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12); margin-bottom: 1.5rem; border: 3px solid #B0BEC5; position: relative;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 5px; background: linear-gradient(90deg, #4CAF50, #00CED1, #FF8C00); border-radius: 25px 25px 0 0;"></div>
            <h4 style="color: #4CAF50; margin-bottom: 1rem; margin-top: 0.5rem;">🤖 AI Model</h4>
            <div style="color: #757575; line-height: 2;">
                <strong style="color: #4CAF50;">🏭 Provider:</strong> {"🏠 Local" if st.session_state.model_provider == "Local Models (Ollama)" else "☁️ API"}<br>
                <strong style="color: #00CED1;">🤖 Model:</strong> {st.session_state.selected_model}<br>
                <strong style="color: #FF8C00;">⚡ Status:</strong> Active
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%); padding: 1.5rem; border-radius: 25px; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12); margin-bottom: 1.5rem; border: 3px solid #B0BEC5; position: relative;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 5px; background: linear-gradient(90deg, #4CAF50, #00CED1, #FF8C00); border-radius: 25px 25px 0 0;"></div>
            <h4 style="color: #4CAF50; margin-bottom: 1rem; margin-top: 0.5rem;">📊 Database</h4>
            <div style="color: #757575; line-height: 2;">
                <strong style="color: #4CAF50;">📄 PDFs:</strong> {metadata['total_pdfs']}<br>
                <strong style="color: #00CED1;">📝 Chunks:</strong> {metadata['total_chunks']:,}<br>
                <strong style="color: #FF8C00;">🔤 Model:</strong> {metadata['embedding_model']}<br>
                <strong style="color: #90A4AE;">📅 Created:</strong> {metadata['created_at']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Switch Database", use_container_width=True):
            st.session_state.mode_selected = False
            st.session_state.database_loaded = False
            st.session_state.llm_configured = False
            if "messages" in st.session_state:
                del st.session_state.messages
            st.rerun()
        
        if st.button("🤖 Change AI Model", use_container_width=True):
            st.session_state.llm_configured = False
            if "messages" in st.session_state:
                del st.session_state.messages
            st.rerun()
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("""
        <div style="background: linear-gradient(145deg, #e8f5e8 0%, #f0f8f0 100%); padding: 1.5rem; border-radius: 20px; margin-top: 2rem; border: 3px solid #B0BEC5; position: relative; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, #FF8C00, #FFA726); border-radius: 20px 20px 0 0;"></div>
            <h5 style="color: #4CAF50; margin-bottom: 1rem; margin-top: 0.5rem;">🤖 Tips</h5>
            <ul style="color: #757575; font-size: 0.95rem; margin: 0; line-height: 1.8;">
                <li>Ask specific questions</li>
                <li>Reference document names</li>
                <li>Use follow-up questions</li>
                <li>Try different phrasings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)