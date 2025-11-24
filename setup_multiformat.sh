#!/bin/bash
# Setup script for RAG Intelligence Hub - Multi-Format Support

echo "================================================"
echo "RAG Intelligence Hub - Multi-Format Setup"
echo "================================================"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "🐧 Detected Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
    echo "🍎 Detected macOS"
else
    OS="other"
    echo "⚠️  Detected: $OSTYPE"
    echo "This script is optimized for Linux and macOS"
fi

echo ""
echo "Step 1: Installing Python dependencies..."
echo "----------------------------------------"
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed"
else
    echo "❌ Error installing Python dependencies"
    exit 1
fi

echo ""
echo "Step 2: Checking system dependencies..."
echo "----------------------------------------"

# Check Tesseract (for OCR)
if command_exists tesseract; then
    TESSERACT_VERSION=$(tesseract --version | head -n1)
    echo "✅ Tesseract already installed: $TESSERACT_VERSION"
else
    echo "❌ Tesseract not found"
    if [ "$OS" == "linux" ]; then
        echo "📦 Installing Tesseract (Linux)..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr
    elif [ "$OS" == "mac" ]; then
        echo "📦 Installing Tesseract (macOS)..."
        brew install tesseract
    else
        echo "⚠️  Please install Tesseract manually:"
        echo "   Ubuntu/Debian: sudo apt-get install tesseract-ocr"
        echo "   macOS: brew install tesseract"
        echo "   Windows: https://github.com/UB-Mannheim/tesseract/wiki"
    fi
fi

# Check ffmpeg (for audio/video)
if command_exists ffmpeg; then
    FFMPEG_VERSION=$(ffmpeg -version | head -n1 | cut -d' ' -f3)
    echo "✅ ffmpeg already installed: $FFMPEG_VERSION"
else
    echo "❌ ffmpeg not found"
    if [ "$OS" == "linux" ]; then
        echo "📦 Installing ffmpeg (Linux)..."
        sudo apt-get install -y ffmpeg
    elif [ "$OS" == "mac" ]; then
        echo "📦 Installing ffmpeg (macOS)..."
        brew install ffmpeg
    else
        echo "⚠️  Please install ffmpeg manually:"
        echo "   Ubuntu/Debian: sudo apt-get install ffmpeg"
        echo "   macOS: brew install ffmpeg"
        echo "   Windows: https://ffmpeg.org/download.html"
    fi
fi

echo ""
echo "Step 3: Verifying installation..."
echo "----------------------------------------"

# Test imports
python3 << EOF
import sys
errors = []

try:
    import streamlit
    print("✅ Streamlit")
except ImportError:
    print("❌ Streamlit")
    errors.append("streamlit")

try:
    import langchain
    print("✅ LangChain")
except ImportError:
    print("❌ LangChain")
    errors.append("langchain")

try:
    from docx import Document
    print("✅ python-docx (Word)")
except ImportError:
    print("❌ python-docx (Word)")
    errors.append("python-docx")

try:
    from openpyxl import load_workbook
    print("✅ openpyxl (Excel)")
except ImportError:
    print("❌ openpyxl (Excel)")
    errors.append("openpyxl")

try:
    from pptx import Presentation
    print("✅ python-pptx (PowerPoint)")
except ImportError:
    print("❌ python-pptx (PowerPoint)")
    errors.append("python-pptx")

try:
    from PIL import Image
    print("✅ Pillow (Images)")
except ImportError:
    print("❌ Pillow (Images)")
    errors.append("Pillow")

try:
    import pytesseract
    print("✅ pytesseract (OCR)")
except ImportError:
    print("❌ pytesseract (OCR)")
    errors.append("pytesseract")

try:
    import whisper
    print("✅ openai-whisper (Audio)")
except ImportError:
    print("❌ openai-whisper (Audio)")
    errors.append("openai-whisper")

try:
    from moviepy.editor import VideoFileClip
    print("✅ moviepy (Video)")
except ImportError:
    print("⚠️  moviepy (Video) - optional")

if errors:
    print(f"\n❌ Missing packages: {', '.join(errors)}")
    print("Run: pip install " + " ".join(errors))
    sys.exit(1)
else:
    print("\n✅ All core packages installed successfully!")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Some Python packages are missing. Please install them."
    exit 1
fi

echo ""
echo "Step 4: Testing system tools..."
echo "----------------------------------------"

# Test tesseract
if command_exists tesseract; then
    echo "✅ Tesseract OCR ready"
else
    echo "⚠️  Tesseract OCR not available (images won't work)"
fi

# Test ffmpeg
if command_exists ffmpeg; then
    echo "✅ ffmpeg ready"
else
    echo "⚠️  ffmpeg not available (audio/video won't work)"
fi

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "Supported formats:"
echo "  📄 Documents: PDF, Word, PowerPoint"
echo "  📊 Spreadsheets: Excel"
echo "  📝 Text: TXT, Markdown"
if command_exists tesseract; then
    echo "  🖼️  Images: PNG, JPG, etc. (with OCR)"
fi
if command_exists ffmpeg; then
    echo "  🎵 Audio: MP3, WAV, etc. (with transcription)"
    echo "  🎬 Video: MP4, AVI, etc. (with transcription)"
fi
echo ""
echo "To start the application:"
echo "  streamlit run advance_rag_modern.py"
echo ""
echo "For detailed documentation, see MULTIFORMAT_GUIDE.md"
echo ""
