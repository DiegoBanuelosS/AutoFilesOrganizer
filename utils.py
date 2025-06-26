"""
Utility functions for the AutoOrganizer system.
Contains file handling, logging, and general utility functions.
"""

import os
import json
import shutil
import logging
import filetype
import chardet
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import colorlog


def setup_logging(log_file: str = "system.log") -> logging.Logger:
    """
    Set up logging configuration with both file and console handlers.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("AutoOrganizer")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_file_info(file_path: str) -> Dict:
    """
    Extract comprehensive information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"error": "File does not exist"}
    
    try:
        # Basic file information
        stat = file_path.stat()
        file_info = {
            "name": file_path.name,
            "stem": file_path.stem,
            "suffix": file_path.suffix.lower(),
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "path": str(file_path.absolute())
        }
        
        # File type detection
        file_type = filetype.guess(str(file_path))
        if file_type:
            file_info["mime_type"] = file_type.mime
            file_info["detected_extension"] = file_type.extension
        else:
            file_info["mime_type"] = "unknown"
            file_info["detected_extension"] = None
        
        # For text files, try to detect encoding and read content
        if is_text_file(file_path):
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read(8192)  # Read first 8KB
                    encoding_result = chardet.detect(raw_data)
                    file_info["encoding"] = encoding_result.get("encoding", "unknown")
                    
                # Try to read text content
                if file_info["encoding"] and file_info["encoding"] != "unknown":
                    with open(file_path, 'r', encoding=file_info["encoding"], errors='ignore') as f:
                        content = f.read(2000)  # Read first 2000 characters
                        file_info["text_content"] = content
                        file_info["is_text"] = True
            except Exception as e:
                file_info["text_content"] = ""
                file_info["is_text"] = False
                file_info["read_error"] = str(e)
        else:
            file_info["is_text"] = False
            file_info["text_content"] = ""
        
        return file_info
        
    except Exception as e:
        return {"error": f"Failed to get file info: {str(e)}"}


def is_text_file(file_path: Path) -> bool:
    """
    Determine if a file is likely a text file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is likely text, False otherwise
    """
    text_extensions = {
        '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', 
        '.yml', '.csv', '.log', '.ini', '.cfg', '.conf', '.sh', '.bat', '.ps1',
        '.sql', '.r', '.cpp', '.c', '.h', '.java', '.php', '.rb', '.go', '.rs',
        '.swift', '.kt', '.scala', '.pl', '.lua', '.vim', '.tex', '.rst', '.org'
    }
    
    if file_path.suffix.lower() in text_extensions:
        return True
    
    # Additional check for files without extensions
    if not file_path.suffix:
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                # Check if the file contains mostly printable ASCII characters
                printable_ratio = sum(c < 128 and (c >= 32 or c in [9, 10, 13]) for c in chunk) / len(chunk)
                return printable_ratio > 0.7
        except:
            return False
    
    return False


def create_directory_structure(base_path: str, categories: List[str], subdirectories: Dict[str, List[str]] = None) -> bool:
    """
    Create the directory structure for organized files with subdirectories.
    
    Args:
        base_path: Base path for the organized files
        categories: List of category names
        subdirectories: Dictionary mapping categories to their subdirectories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True)
        
        for category in categories:
            category_path = base_path / sanitize_filename(category)
            category_path.mkdir(exist_ok=True)
            
            # Create subdirectories if provided
            if subdirectories and category in subdirectories:
                for subdir in subdirectories[category]:
                    subdir_path = category_path / sanitize_filename(subdir)
                    subdir_path.mkdir(exist_ok=True)
            
        return True
    except Exception as e:
        logging.getLogger("AutoOrganizer").error(f"Failed to create directory structure: {e}")
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Characters that are invalid in Windows filenames
    invalid_chars = '<>:"/\\|?*'
    
    # Replace invalid characters with underscores
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure the filename is not empty
    if not filename:
        filename = "unnamed"
    
    return filename


def move_file(source: str, destination: str, create_dirs: bool = True) -> bool:
    """
    Move a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        create_dirs: Whether to create destination directories if they don't exist
        
    Returns:
        True if successful, False otherwise
    """
    try:
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            return False
        
        if create_dirs:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle file name conflicts
        if dest_path.exists():
            base_name = dest_path.stem
            extension = dest_path.suffix
            counter = 1
            
            while dest_path.exists():
                new_name = f"{base_name}_{counter}{extension}"
                dest_path = dest_path.parent / new_name
                counter += 1
        
        shutil.move(str(source_path), str(dest_path))
        return True
        
    except Exception as e:
        logging.getLogger("AutoOrganizer").error(f"Failed to move file {source} to {destination}: {e}")
        return False


def load_processed_files(file_path: str = "processed_files.json") -> Dict:
    """
    Load the record of processed files.
    
    Args:
        file_path: Path to the processed files JSON file
        
    Returns:
        Dictionary of processed files
    """
    try:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logging.getLogger("AutoOrganizer").error(f"Failed to load processed files: {e}")
        return {}


def save_processed_files(processed_files: Dict, file_path: str = "processed_files.json") -> bool:
    """
    Save the record of processed files.
    
    Args:
        processed_files: Dictionary of processed files
        file_path: Path to the processed files JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.getLogger("AutoOrganizer").error(f"Failed to save processed files: {e}")
        return False


def get_file_hash(file_path: str) -> Optional[str]:
    """
    Get a simple hash of a file for duplicate detection.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File hash string or None if failed
    """
    try:
        import hashlib
        hash_md5 = hashlib.md5()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    except Exception:
        return None


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def should_ignore_file(file_path: Path) -> bool:
    """
    Determine if a file should be ignored during processing.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file should be ignored, False otherwise
    """
    filename = file_path.name.lower()
    
    # Ignore hidden files (starting with .)
    if filename.startswith('.'):
        return True
    
    # Ignore git keep files
    if filename == '.gitkeep':
        return True
    
    # Ignore temporary files
    temp_extensions = {'.tmp', '.temp', '.swp', '.bak', '.~'}
    if file_path.suffix.lower() in temp_extensions:
        return True
    
    # Ignore system files
    system_files = {'thumbs.db', 'desktop.ini', '.ds_store'}
    if filename in system_files:
        return True
    
    return False