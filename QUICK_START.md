# 🚀 Quick Start Guide - Multi-Format RAG

## Installation (2 minutes)

```bash
# Option 1: Easy setup (recommended)
chmod +x setup_multiformat.sh
./setup_multiformat.sh

# Option 2: Manual
pip install -r requirements.txt
sudo apt-get install tesseract-ocr ffmpeg  # Linux
brew install tesseract ffmpeg              # macOS
```

## Launch Application

```bash
streamlit run advance_rag_modern.py
```

## Create Your First Database

### Step 1: Configure Database
1. Click **Configuration** tab
2. Select **📂 Database** → **Create New Database**
3. Choose source:
   - **Upload Files**: Drag & drop your documents
   - **Server Folder**: Point to existing folder

### Step 2: Upload Files
Supported formats:
- 📄 **Documents**: PDF, Word (.docx), PowerPoint (.pptx)
- 📊 **Spreadsheets**: Excel (.xlsx, .xls)
- 📝 **Text**: .txt, .md
- 🖼️ **Images**: .png, .jpg (with OCR)
- 🎵 **Audio**: .mp3, .wav (transcribed)
- 🎬 **Video**: .mp4, .avi (transcribed)

### Step 3: Create Database
1. Click **🚀 Create Vector Database**
2. Wait for processing (1-30 seconds per file)
3. Success! ✅

### Step 4: Configure AI Model
1. Go to **🤖 AI Model** tab
2. Choose provider:
   - **Local**: Ollama (free, requires install)
   - **Cloud**: Groq API (requires key)
3. Click **🚀 Initialize AI Model**

### Step 5: Start Chatting
1. Go to **💬 Chat** tab
2. Ask questions about your documents
3. Get answers with source citations

## Add Files to Existing Database

1. **Configuration** → **Database** → **Load Existing Database**
2. Click **🔍 Scan for Databases**
3. Select your database
4. Click **➕ Add Files to Database**
5. Upload new files (any format)
6. Click **🚀 Add Files to Database**

## Common Use Cases

### Meeting Transcription
```
1. Upload: meeting_recording.mp3
2. Wait: ~3 minutes for transcription
3. Ask: "What action items were discussed?"
```

### Document Analysis
```
1. Upload: report.pdf, data.xlsx, diagram.png
2. Ask: "Summarize the key findings from the report"
3. Get: Comprehensive answer with citations
```

### Research Papers
```
1. Upload: Multiple PDF papers
2. Ask: "Compare the methodologies used"
3. Get: Cross-referenced analysis
```

## File Processing Times

| File Type | Size | Time |
|-----------|------|------|
| PDF | 100 pages | 10-15s |
| Word/Excel | 50 pages | 5-10s |
| Image (OCR) | 1 image | 2-5s |
| Audio | 10 minutes | 60-90s |
| Video | 10 minutes | 90-120s |

## Tips for Best Results

### Images (OCR)
✅ High contrast, clear text
✅ 300+ DPI resolution
❌ Blurry, low-contrast images

### Audio/Video
✅ Clear speech, minimal background noise
✅ Good quality recording
❌ Multiple overlapping speakers

### Documents
✅ Native formats (not scanned)
✅ Structured content
✅ Clear formatting

## Troubleshooting

### "Tesseract not found"
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Verify
tesseract --version
```

### "ffmpeg not found"
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Verify
ffmpeg -version
```

### "ImportError: No module named..."
```bash
pip install -r requirements.txt
```

### Slow Processing
- Process fewer files at once
- Use smaller audio/video files
- Increase system RAM
- Close other applications

### OCR Not Accurate
- Use higher resolution images
- Improve image contrast
- Ensure text is clear and readable
- Try different image formats

## Keyboard Shortcuts

- `Ctrl + C`: Stop application
- `Ctrl + Shift + R`: Reload page
- `Ctrl + K`: Clear chat (if implemented)

## API Keys (Optional)

### Groq API (Cloud Models)
```bash
# Get key from: https://console.groq.com
echo "GROQ_API_KEY=gsk_..." > .env
```

### Ollama (Local Models)
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2:3b

# Start service
ollama serve
```

## File Size Limits

| Type | Recommended Max | Hard Limit |
|------|-----------------|------------|
| PDF | 50 MB | 200 MB |
| Word/Excel | 20 MB | 100 MB |
| Image | 10 MB | 50 MB |
| Audio | 100 MB | 500 MB |
| Video | 200 MB | 1 GB |

*Larger files may work but will take longer and use more memory*

## Performance Tips

1. **Batch Processing**: Upload 5-10 files at a time
2. **File Quality**: Higher quality = better results
3. **System Resources**: Close other apps during processing
4. **Database Reuse**: Add to existing DBs instead of recreating

## Getting Help

1. Check **MULTIFORMAT_GUIDE.md** for detailed docs
2. Review error messages carefully
3. Test with smaller files first
4. Verify all dependencies installed
5. Check system requirements met

## Next Steps

- Explore different file formats
- Test with your own documents
- Experiment with different questions
- Try adding files to existing databases
- Share feedback and suggestions!

---

**Need more help?** See MULTIFORMAT_GUIDE.md for complete documentation.
