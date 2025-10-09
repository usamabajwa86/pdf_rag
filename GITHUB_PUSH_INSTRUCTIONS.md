# GitHub Push Instructions for Advanced RAG Application

## 🚀 **Step-by-Step Guide to Push to GitHub**

### **Prerequisites Setup**

#### **1. Install Git (if not already installed)**
1. **Download Git:** Go to https://git-scm.com/download/win
2. **Install Git:** Run the installer with default settings
3. **Verify installation:** Open new PowerShell and run: `git --version`

#### **2. Configure Git (First Time Only)**
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### **🔧 Repository Setup Commands**

#### **Navigate to your project directory:**
```bash
cd "d:\Codes\sep-2025\rag\Advance RAG_Hybrid Search"
```

#### **Initialize Git repository:**
```bash
git init
```

#### **Add remote repository:**
```bash
git remote add origin https://github.com/usamabajwa86/rag.git
```

#### **Create/update .gitignore file:**
```bash
# Add this content to .gitignore (if not already present)
.env
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
pip-log.txt
pip-delete-this-directory.txt
.DS_Store
Thumbs.db
.idea/
.vscode/
*.log
temp_images/
vector_databases/
*.faiss
*.pkl
```

### **📦 Push to GitHub**

#### **Add all files to staging:**
```bash
git add .
```

#### **Create initial commit:**
```bash
git commit -m "Initial commit: Advanced RAG Application with Hybrid Search

Features:
- Local and API model support (Ollama + Groq)
- Dynamic citation system with relevance filtering
- PDF viewing capabilities
- Hybrid search (semantic + keyword)
- Interactive Streamlit interface
- Multi-database support
- Real-time configuration options"
```

#### **Push to GitHub:**
```bash
git push -u origin main
```

### **🔄 For Future Updates**

#### **Add changes:**
```bash
git add .
```

#### **Commit changes:**
```bash
git commit -m "Description of your changes"
```

#### **Push updates:**
```bash
git push
```

### **📁 Repository Structure**

Your GitHub repository will contain:

```
rag/
├── advance_rag.py                    # Main application
├── README.md                         # Project documentation
├── requirements_frozen.txt           # Dependencies with exact versions
├── .gitignore                       # Git ignore rules
├── .env.template                    # Environment template
├── LICENSE                          # License file
├── assets/                          # Static assets
├── Data/                           # Sample data (if included)
├── Advanced_RAG_Documentation.md   # Technical documentation
├── ENHANCED_CITATIONS_SUMMARY.md   # Feature summary
├── PDF_VIEWING_ENHANCEMENT_GUIDE.md # PDF viewing guide
└── vector_databases/               # (ignored - local only)
```

### **🔐 Important Security Notes**

#### **Files to NEVER commit:**
- `.env` (contains API keys)
- `.venv/` (virtual environment)
- `vector_databases/` (large database files)
- Any files containing sensitive data

#### **Safe to commit:**
- `.env.template` (template without real keys)
- `requirements_frozen.txt` (dependency list)
- Documentation files
- Source code files

### **🎯 Quick Commands Summary**

```bash
# One-time setup
git init
git remote add origin https://github.com/usamabajwa86/rag.git

# For each push
git add .
git commit -m "Your commit message"
git push -u origin main  # First push
git push                 # Subsequent pushes
```

### **⚠️ Troubleshooting**

#### **If you get authentication errors:**
1. **Use GitHub Personal Access Token** instead of password
2. **Enable 2FA** on GitHub account
3. **Configure Git credentials:**
   ```bash
   git config --global credential.helper manager-core
   ```

#### **If repository already exists:**
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

#### **If you get merge conflicts:**
```bash
git pull origin main
# Resolve conflicts manually
git add .
git commit -m "Resolved merge conflicts"
git push
```

### **🌟 Benefits of Using GitHub**

1. **Version Control** - Track all changes and history
2. **Backup** - Your code is safe in the cloud
3. **Collaboration** - Share with team members
4. **Documentation** - README and guides visible on GitHub
5. **Releases** - Tag and release versions
6. **Issues** - Track bugs and feature requests

### **📝 Recommended README.md Content**

Make sure your README.md includes:
- Project description
- Features overview
- Installation instructions
- Usage examples
- API key setup
- Model configuration
- Screenshots/demos

Your Advanced RAG application will be properly organized and accessible on GitHub! 🚀