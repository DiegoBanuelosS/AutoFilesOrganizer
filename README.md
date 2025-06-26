# AutoFilesOrganizer 

An AI-powered automatic file organizer that intelligently categorizes and organizes files based on their content, not just file extensions. Using advanced AI analysis, it creates structured directory hierarchies with meaningful categories and subdirectories.

## Features

- **AI-Powered Analysis**: Uses sentence transformers to understand file content and context
- **Intelligent Categorization**: Organizes files by topic and function, not just file type
- **Hierarchical Structure**: Creates detailed subdirectory structures for better organization
- **Content-Aware**: Analyzes file content, comments, and metadata for accurate categorization
- **Batch Processing**: Processes multiple files efficiently with detailed logging
- **Structure Visualization**: Shows complete organization structure with file counts
- **Safe Operation**: Tracks processed files to avoid duplicates

##  Directory Structure

```
AutoOrganizer/
├── input_files/           # Files to be organized (upload here)
├── organized_files/       # AI-organized output directory
├── main.py               # Main application entry point
├── ai_analyzer.py        # AI categorization engine
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── processed_files.json  # Tracking file (auto-generated)
└── system.log           # Application logs
```

## AI Categories

The system automatically organizes files into intelligent categories:

### Development_Code
- **Configuration**: Config files, settings, environments
- **Database**: SQL schemas, database files
- **Python_Projects**: Python scripts and applications
- **Web_Development**: Frontend code, web frameworks
- **Mobile_Development**: Mobile app code
- **Desktop_Applications**: Desktop software code

### Work_Projects
- **Reports**: Business reports, analytics
- **Presentations**: Slides, presentation files
- **Spreadsheets**: Excel files, data analysis
- **Meetings**: Meeting notes, minutes

### Finance_Money
- **Budgets**: Budget files, financial planning
- **Invoices**: Bills, invoices, receipts
- **Reports**: Financial reports, statements

### Media_Content
- **Images**: Photos, graphics, artwork
- **Videos**: Video files, multimedia
- **Audio**: Music, podcasts, sound files
- **Design**: Design files, mockups

### Education_Learning
- **Courses**: Course materials, lectures
- **Research**: Academic papers, studies
- **Tutorials**: Learning materials, guides

### Tools_Utilities
- **Configuration**: System configs, settings
- **Installers**: Installation files, setups
- **Scripts**: Utility scripts, automation

### Reference_Documentation
- **Manuals**: User guides, documentation
- **Specifications**: Technical specs
- **General**: General reference materials

## Next Changes
- UI
- Better organization
- optimized AI
- and more functions...

##  Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DiegoBanuelosS/AutoOrganizer.git
   cd AutoOrganizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **First run** (downloads AI model):
   ```bash
   python main.py
   ```

## Usage

### Basic Organization
Place files in the `input_files/` directory and run:
```bash
python main.py
```

### Show Organization Structure
View the complete organized structure:
```bash
python main.py --show-structure
```

### Continuous Monitoring
Run in continuous mode to monitor for new files:
```bash
python main.py --continuous
```

### Command Line Options
- `--show-structure`: Display the complete organization structure
- `--continuous`: Run in continuous monitoring mode
- `--help`: Show help message

## Example Output

```
 Development_Code (15 files)
├──  Python_Projects (8 files)
│   ├── web_scraper.py
│   ├── ml_house_predictor.py
│   ├── task_manager.py
│   └── sales_report_q4.py
├──  Database (3 files)
│   ├── ecommerce_schema.sql
│   └── user_database.sql
└──  Configuration (4 files)
    ├── config.json
    └── settings.yml

 Tools_Utilities (7 files)
├──  Configuration (3 files)
└──  Installers (2 files)

 Grand Total: 22 organized files
```

##  How It Works

1. **File Scanning**: Scans `input_files/` directory for new files
2. **Content Analysis**: AI analyzes file content, structure, and metadata
3. **Category Assignment**: Assigns appropriate category and subdirectory
4. **Intelligent Organization**: Moves files to structured directories
5. **Progress Tracking**: Logs all operations and tracks processed files

##  Technical Details

- **AI Model**: Uses `all-MiniLM-L6-v2` sentence transformer
- **Content Analysis**: Reads file content, comments, and metadata
- **Confidence Scoring**: Each categorization includes confidence level
- **Fallback Logic**: Uses file extension as backup for unknown files
- **Error Handling**: Comprehensive error handling and logging

##  Logging

The system provides detailed logging:
- **File Operations**: Track all file movements
- **AI Analysis**: Confidence scores and reasoning
- **Error Handling**: Detailed error messages
- **Statistics**: Processing stats and summaries

##  Configuration

### Adding Custom Categories
Edit `ai_analyzer.py` to add new categories:
```python
self.categories = {
    "Your_Custom_Category": {
        "keywords": ["keyword1", "keyword2"],
        "subdirectories": ["SubDir1", "SubDir2"]
    }
}
```

### Ignore Files
Edit `utils.py` to customize ignored files:
```python
def should_ignore_file(file_path: Path) -> bool:
    # Add your custom ignore logic
    return file_path.name.startswith('custom_prefix')
```

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for the AI model
- [Hugging Face](https://huggingface.co/) for transformer models
- Python ecosystem for excellent libraries

##  Support

If you encounter any issues:

1. Check the `system.log` file for detailed error messages
2. Ensure all dependencies are installed correctly
3. Verify file permissions for `input_files/` and `organized_files/`
4. Open an issue on GitHub with log details

---
## note
All codes have comments to facilitate their interpretation.

## Author

- [@DiegoBanuelosS](https://github.com/DiegoBanuelosS)
