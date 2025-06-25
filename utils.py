"""
Additional utilities for AutoOrganizer
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import zipfile
import tarfile

class BackupManager:
    """Backup manager"""
    
    def __init__(self, root_directory: Path):
        self.root_directory = Path(root_directory)
        self.backup_directory = self.root_directory / "backups"
        self.backup_directory.mkdir(exist_ok=True)
    
    def create_backup(self, description: str = "") -> Path:
        """Creates a backup of the organized file structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}.zip"
        backup_path = self.backup_directory / backup_name
        
        organized_dir = self.root_directory / "organized_files"
        if not organized_dir.exists():
            raise FileNotFoundError("No organized files to backup")
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in organized_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(organized_dir)
                    zipf.write(file_path, arcname)
        
        metadata = {
            "created_date": datetime.now().isoformat(),
            "description": description,
            "total_files": sum(1 for _ in organized_dir.rglob('*') if _.is_file()),
            "backup_size": backup_path.stat().st_size
        }
        
        metadata_path = backup_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return backup_path
    
    def restore_backup(self, backup_path: Path) -> bool:
        """Restores a backup"""
        try:
            organized_dir = self.root_directory / "organized_files"
            
            if organized_dir.exists():
                shutil.rmtree(organized_dir)
            
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(organized_dir)
            
            return True
        except Exception as e:
            print(f"Error restoring backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """Lists all available backups"""
        backups = []
        for backup_file in self.backup_directory.glob("backup_*.zip"):
            metadata_file = backup_file.with_suffix('.json')
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    metadata['file_path'] = str(backup_file)
                    backups.append(metadata)
        
        return sorted(backups, key=lambda x: x['created_date'], reverse=True)

class DuplicateDetector:
    """Duplicate file detector"""
    
    def __init__(self):
        self.file_hashes = {}
    
    def find_duplicates(self, directory: Path) -> Dict[str, List[Path]]:
        """Finds duplicate files based on MD5 hash"""
        duplicates = {}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                file_hash = self._calculate_hash(file_path)
                if file_hash:
                    if file_hash in self.file_hashes:
                        if file_hash not in duplicates:
                            duplicates[file_hash] = [self.file_hashes[file_hash]]
                        duplicates[file_hash].append(file_path)
                    else:
                        self.file_hashes[file_hash] = file_path
        
        return duplicates
    
    def _calculate_hash(self, file_path: Path) -> Optional[str]:
        """Calculates MD5 hash of a file"""
        try:
            import hashlib
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return None

class ReportGenerator:
    """Report generator"""
    
    def __init__(self, root_directory: Path):
        self.root_directory = Path(root_directory)
    
    def generate_organization_report(self, stats: Dict) -> Path:
        """Generates a detailed organization report"""
        report_path = self.root_directory / f"organization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Organization Report - Smart File Organizer</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-box {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .stat-number {{
                    font-size: 2em;
                    font-weight: bold;
                    display: block;
                }}
                .directory-tree {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #28a745;
                    margin: 20px 0;
                }}
                .timestamp {{
                    text-align: center;
                    color: #6c757d;
                    font-style: italic;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Smart File Organizer - Organization Report</h1>
                
                <div class="stats">
                    <div class="stat-box">
                        <span class="stat-number">{stats.get('total_files', 0)}</span>
                        <span>Files Processed</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-number">{stats.get('organized_files', 0)}</span>
                        <span>Files Organized</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-number">{stats.get('skipped_files', 0)}</span>
                        <span>Files Skipped</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-number">{stats.get('errors', 0)}</span>
                        <span>Errors</span>
                    </div>
                </div>
                
                <div class="directory-tree">
                    <h3>Directory Structure Created</h3>
                    <p>Files have been organized in the directory: <strong>organized_files/</strong></p>
                    {self._generate_directory_tree_html()}
                </div>
                
                <div class="timestamp">
                    <p>Report generated on: {datetime.now().strftime('%m/%d/%Y at %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _generate_directory_tree_html(self) -> str:
        """Generates HTML to show directory structure"""
        organized_dir = self.root_directory / "organized_files"
        if not organized_dir.exists():
            return "<p>No directory structure available.</p>"
        
        html = "<ul>"
        for category_dir in organized_dir.iterdir():
            if category_dir.is_dir():
                file_count = sum(1 for _ in category_dir.rglob('*') if _.is_file())
                html += f"<li><strong>{category_dir.name}/</strong> ({file_count} files)</li>"
                html += "<ul>"
                for subdir in category_dir.iterdir():
                    if subdir.is_dir():
                        sub_file_count = sum(1 for _ in subdir.rglob('*') if _.is_file())
                        html += f"<li>{subdir.name}/ ({sub_file_count} files)</li>"
                html += "</ul>"
        html += "</ul>"
        
        return html

class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.default_config = {
            "auto_monitor": False,
            "backup_before_organize": True,
            "smart_categorization": True,
            "duplicate_detection": True,
            "max_file_size_mb": 100,
            "exclude_extensions": [".tmp", ".temp", ".log"],
            "custom_categories": {}
        }
    
    def load_config(self) -> Dict:
        """Loads configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return self.default_config
        return self.default_config
    
    def save_config(self, config: Dict):
        """Saves configuration to file"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def reset_config(self):
        """Resets configuration to default values"""
        self.save_config(self.default_config)

def format_file_size(size_bytes: int) -> str:
    """Formats file size in readable format"""
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.1f} {units[unit_index]}"

def validate_directory(directory_path: str) -> bool:
    """Validates if a directory exists and is accessible"""
    try:
        path = Path(directory_path)
        return path.exists() and path.is_dir() and os.access(path, os.R_OK | os.W_OK)
    except Exception:
        return False

def create_sample_files(directory: Path, count: int = 5):
    """Creates sample files for testing"""
    samples_dir = directory / "sample_files"
    samples_dir.mkdir(exist_ok=True)
    
    for i in range(count):
        (samples_dir / f"document_{i+1}.txt").write_text(
            f"This is sample document number {i+1}\n"
            f"Created to test the Smart File Organizer system\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
    sample_data = {
        "name": "Sample data",
        "items": [f"item_{i}" for i in range(10)],
        "creation_date": datetime.now().isoformat()
    }
    with open(samples_dir / "sample_data.json", 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    import csv
    with open(samples_dir / "sample_table.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Name', 'Value'])
        for i in range(10):
            writer.writerow([i+1, f'Element {i+1}', i*10])
    
    print(f"Sample files created in: {samples_dir}")
    return samples_dir
