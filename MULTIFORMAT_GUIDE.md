# Multi-Format Document Processing Guide

## Overview

The RAG Intelligence Hub now supports processing multiple file formats beyond PDFs, including:
- **Documents**: PDF, Word (.docx, .doc), PowerPoint (.pptx, .ppt)
- **Spreadsheets**: Excel (.xlsx, .xls)
- **Text**: Plain text (.txt), Markdown (.md)
- **Images**: PNG, JPG, JPEG, TIFF, BMP, GIF (with OCR)
- **Audio**: MP3, WAV, M4A, FLAC, OGG (with transcription)
- **Video**: MP4, AVI, MOV, MKV, FLV (with audio transcription)

## New Features

### 1. Universal Document Processing
All supported file types are automatically detected and processed using the appropriate loader:
- PDFs are extracted page by page
- Word documents preserve formatting and structure
- Excel files process each sheet separately
- PowerPoint extracts text from slides
- Images use OCR (pytesseract) to extract text
- Audio/Video files use Whisper AI for transcription

### 2. Add Files to Existing Databases
You can now add more files to existing vector databases without recreating them:
- Load an existing database
- Click "➕ Add Files to Database"
- Upload new files in any supported format
- The system merges them into the existing database
- Metadata is updated automatically

### 3. Enhanced Metadata Tracking
The metadata.json now tracks:
- `file_types`: Count of each file type (e.g., {".pdf": 5, ".docx": 3})
- `total_files`: Total number of files processed
- `updated_at`: Last update timestamp
- `files`: List of all processed file paths
- Backward compatibility with old `total_pdfs` and `pdf_files`

## Installation

### Required Dependencies

Install all dependencies:
```bash
pip install -r requirements.txt
```

### System Requirements for Advanced Features

#### OCR (Image Processing)
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

#### Audio/Video Transcription
Whisper AI requires ffmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from: https://ffmpeg.org/download.html
```

## Usage

### Creating a New Database

1. Go to **Configuration → Database** tab
2. Select **Create New Database**
3. Choose your source:
   - **Upload Files**: Drag and drop any supported files
   - **Use Server Folder**: Select a folder containing your files

4. The system will automatically:
   - Detect file types
   - Process each file with the appropriate loader
   - Create vector embeddings
   - Build searchable database

### Adding Files to Existing Database

1. Go to **Configuration → Database** tab
2. Select **Load Existing Database**
3. Click **Scan for Databases**
4. Select the database you want to update
5. Click **➕ Add Files to Database**
6. Upload new files (any supported format)
7. Click **🚀 Add Files to Database**

The system will:
- Check for duplicate files (skip if already in database)
- Process new files
- Merge into existing vector store
- Update BM25 keyword index
- Update metadata

## Processing Details

### PDF Files
- Uses PyPDFLoader
- Extracts text page by page
- Preserves page numbers in metadata

### Word Documents (.docx)
- Uses python-docx
- Extracts all paragraph text
- Maintains document structure

### Excel Files (.xlsx, .xls)
- Uses openpyxl
- Processes each sheet separately
- Converts rows to text format
- Stores sheet name in metadata

### PowerPoint (.pptx)
- Uses python-pptx
- Extracts text from all shapes
- Processes each slide separately
- Stores slide number in metadata

### Text Files (.txt, .md)
- Direct text reading
- UTF-8 encoding with error handling
- Markdown treated as plain text

### Images (PNG, JPG, etc.)
- Uses Pillow + pytesseract
- OCR extracts text from images
- Handles processing failures gracefully
- Falls back to filename if OCR fails

### Audio Files (MP3, WAV, etc.)
- Uses OpenAI Whisper model
- Transcribes speech to text
- Base model by default (fast, accurate)
- Processes audio directly

### Video Files (MP4, AVI, etc.)
- Uses moviepy to extract audio
- Transcribes audio with Whisper
- Creates temporary audio file (auto-deleted)
- Returns transcribed text

## Performance Tips

### Image Processing
- OCR can be slow for large images
- Consider resizing very large images before upload
- Best results with clear, high-contrast text

### Audio/Video Processing
- First run downloads Whisper model (~140MB)
- Transcription time: ~1/10th of audio duration
- Longer files take more time
- Consider splitting very long videos

### Large Batches
- Process in smaller batches if memory limited
- Audio/Video files use more RAM
- Monitor system resources during processing

## File Format Recommendations

### Best Formats for Accuracy
1. **PDF** - Most reliable, preserves formatting
2. **Word/Text** - Direct text extraction, very fast
3. **Excel** - Good for structured data
4. **Images** - Quality depends on image clarity
5. **Audio/Video** - Depends on audio quality

### Optimization Tips
- **Images**: Use high resolution, good contrast
- **Audio**: Clear speech, minimal background noise
- **Video**: Good audio quality more important than video
- **Documents**: Native formats better than scanned

## Troubleshooting

### OCR Not Working
```bash
# Verify tesseract installation
tesseract --version

# Ubuntu: Reinstall if needed
sudo apt-get install --reinstall tesseract-ocr
```

### Whisper Transcription Errors
```bash
# Check ffmpeg
ffmpeg -version

# Reinstall openai-whisper
pip uninstall openai-whisper
pip install openai-whisper
```

### Memory Issues
- Process fewer files at once
- Use smaller Whisper model (base instead of large)
- Close other applications
- Increase system swap space

### Import Errors
If you see import errors for specific formats:
```bash
# Word documents
pip install python-docx

# Excel files
pip install openpyxl

# PowerPoint
pip install python-pptx

# Images (OCR)
pip install Pillow pytesseract

# Audio/Video
pip install openai-whisper moviepy pydub
```

## API Reference

### Main Functions

#### `load_document_universal(file_path)`
Universal loader that detects and processes any supported file type.
```python
documents = load_document_universal(Path("/path/to/file.docx"))
```

#### `create_new_database(file_list, save_folder)`
Creates a new vector database from multiple file types.
```python
result = create_new_database(
    [Path("doc1.pdf"), Path("doc2.docx"), Path("audio.mp3")],
    "vector_databases"
)
```

#### `add_files_to_database(new_files, db_path, db_folder, vector_store, documents, embeddings)`
Adds new files to an existing database.
```python
result = add_files_to_database(
    [Path("new_file.pdf")],
    "vector_databases",
    "db_12345",
    existing_vector_store,
    existing_documents,
    embeddings
)
```

### Supported Extensions

```python
SUPPORTED_EXTENSIONS = {
    'pdf': ['.pdf'],
    'word': ['.docx', '.doc'],
    'excel': ['.xlsx', '.xls'],
    'powerpoint': ['.pptx', '.ppt'],
    'text': ['.txt', '.md', '.markdown'],
    'image': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'],
    'audio': ['.mp3', '.wav', '.m4a', '.flac', '.ogg'],
    'video': ['.mp4', '.avi', '.mov', '.mkv', '.flv']
}
```

## Examples

### Example 1: Mixed Format Database
```python
# Create database with different file types
files = [
    Path("report.pdf"),
    Path("notes.docx"),
    Path("data.xlsx"),
    Path("diagram.png"),
    Path("meeting.mp3")
]

result = create_new_database(files, "my_database")
```

### Example 2: Adding Images to PDF Database
```python
# Load existing PDF database
vector_store, bm25, docs, emb, meta = load_existing_database("dbs", "db_abc123")

# Add new image files
new_images = [Path("chart1.png"), Path("chart2.jpg")]
result = add_files_to_database(new_images, "dbs", "db_abc123", vector_store, docs, emb)
```

### Example 3: Transcribing Multiple Videos
```python
# Process video files
videos = [
    Path("lecture1.mp4"),
    Path("lecture2.mp4"),
    Path("lecture3.mp4")
]

result = create_new_database(videos, "lecture_db")
```

## Limitations

1. **OCR Accuracy**: Depends on image quality and text clarity
2. **Transcription Accuracy**: Depends on audio quality and accents
3. **Processing Time**: Audio/video files take longer
4. **File Size**: Very large files may cause memory issues
5. **Format Support**: Some obscure formats may not work

## Future Enhancements

Potential future additions:
- HTML and web page processing
- CSV file support
- Code file syntax parsing
- ZIP archive extraction
- URL content fetching
- Database export/import

## Support

For issues or questions:
1. Check this guide first
2. Verify all dependencies installed
3. Check system requirements met
4. Review error messages carefully
5. Test with smaller files first

## License

This feature is part of the RAG Intelligence Hub project.
See LICENSE file for details.
