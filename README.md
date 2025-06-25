# Smart File Organizer AI System

## Intelligent File Organization with AI

Smart File Organizer is an advanced system that uses artificial intelligence to automatically analyze, categorize, and organize your files. It uses only **free and open-source tools**, making it accessible to everyone.

## Key Features

### AI Analysis
- **Content analysis** using NLTK for natural language processing
- **Automatic categorization** based on file content
- **Smart subcategories** for documents (contracts, invoices, reports, etc.)
- **OCR support** with Tesseract for image text extraction
- **Machine learning** with scikit-learn for advanced classification

### Smart Organization
- **Separate input/output folders** for better workflow
- **Automatic directory structure** creation
- **Duplicate handling** with intelligent renaming
- **System file protection** - won't organize program files
- **Real-time monitoring** for new files

### Supported File Types
- **Documents**: PDF, Word, Text, RTF, ODT
- **Images**: JPG, PNG, GIF, BMP, TIFF, SVG, WebP
- **Videos**: MP4, AVI, MKV, MOV, WMV, FLV, WebM
- **Audio**: MP3, WAV, FLAC, AAC, OGG, WMA
- **Code**: Python, JavaScript, HTML, CSS, Java, C++, PHP, etc.
- **Data**: Excel, CSV, JSON, XML, SQL
- **Presentations**: PowerPoint, LibreOffice Impress
- **Compressed**: ZIP, RAR, 7Z, TAR, GZ
- **Executables**: EXE, MSI, DEB, RPM, DMG

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/username/AutoOrganizer.git
cd AutoOrganizer
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the system**:
```bash
python main.py
```

## Directory Structure

```
AutoOrganizer/
├── input_files/          # Drop your files here
├── organized_files/       # Organized files appear here
│   ├── documents/
│   │   ├── contracts/
│   │   ├── invoices/
│   │   ├── reports/
│   │   └── ...
│   ├── images/
│   ├── videos/
│   └── ...
├── main.py               # Main system
├── ai_analyzer.py        # AI module
├── utils.py              # Utilities
└── requirements.txt      # Dependencies
```

## How to Use

1. **Start the program**:
   ```bash
   python main.py
   ```

2. **Place files** in the `input_files` folder

3. **Choose an option**:
   - **Option 1**: Organize existing files
   - **Option 2**: Monitor folder in real-time
   - **Option 3**: View statistics
   - **Option 4**: View folder structure
   - **Option 5**: Open input folder
   - **Option 6**: Open output folder

4. **Check results** in the `organized_files` folder

## AI Features

### Content Analysis
- **Text extraction** from PDFs and Word documents
- **Language detection** (English/Spanish)
- **Keyword extraction** using NLP
- **Topic identification** 
- **Readability analysis**

### Smart Categorization
- **Contract detection**: Legal terms, signatures, parties
- **Invoice recognition**: Payment terms, amounts, taxes
- **Report identification**: Analysis, findings, conclusions
- **Manual detection**: Instructions, procedures, guides
- **Certificate recognition**: Awards, qualifications, achievements

## Monitoring

The system can monitor the input folder in real-time:
- **Automatic detection** of new files
- **Instant organization** upon file creation
- **Background processing** without interrupting work
- **Press Ctrl+C** to stop monitoring

## Statistics and Reports

- **Files processed** count
- **Success/error** rates
- **Category distribution**
- **Processing time** tracking
- **Folder structure** visualization

## Security and Privacy

- **100% local processing** - no data sent to external servers
- **No paid APIs required** - uses only open-source tools
- **Open source** - full transparency
- **System file protection** - won't touch important files

## AI Tools Used

- **NLTK**: Natural Language Processing
- **scikit-learn**: Machine Learning
- **Tesseract**: Optical Character Recognition
- **TextStat**: Text Analysis
- **FuzzyWuzzy**: String Matching
- **OpenCV**: Image Processing

## Troubleshooting

### Common Issues

1. **AI not available**: Install missing packages
   ```bash
   pip install nltk scikit-learn textstat fuzzywuzzy
   ```

2. **OCR not working**: Install Tesseract
   - Windows: Download from GitHub releases
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

3. **File not organizing**: Check file permissions and size limits

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **NLTK Team** for natural language processing
- **scikit-learn** for machine learning tools
- **Tesseract** for OCR capabilities
- **Open source community** for AI tools

---

## Quick Start Example

```bash
# Install and run
git clone https://github.com/username/AutoOrganizer.git
cd AutoOrganizer
pip install -r requirements.txt
python main.py

# Place some files in input_files/ folder
# Select option 1 to organize them
# Check organized_files/ for results
```

**Enjoy organized files with AI power!**

## Author

- [@DiegoBanuelosS](https://github.com/DiegoBanuelosS)