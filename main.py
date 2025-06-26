import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import signal

from utils import (
    setup_logging, get_file_info, create_directory_structure, 
    move_file, load_processed_files, save_processed_files,
    format_file_size, should_ignore_file
)
from ai_analyzer import AIFileAnalyzer


class AutoOrganizer:
    
    def __init__(self, input_dir: str = "input_files", output_dir: str = "organized_files"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.logger = setup_logging()
        self.ai_analyzer = AIFileAnalyzer()
        self.processed_files = load_processed_files()
        self.running = False
        
        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info("AutoOrganizer initialized")
        self.logger.info(f"Input directory: {self.input_dir.absolute()}")
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
    
    def scan_input_directory(self) -> List[str]:
        files_to_process = []
        
        try:
            self.logger.info(f"Scanning directory: {self.input_dir.absolute()}")
            total_files = 0
            ignored_files = 0
            already_processed = 0
            
            for file_path in self.input_dir.rglob("*"):
                if file_path.is_file():
                    total_files += 1
                    self.logger.debug(f"Found file: {file_path}")
                    
                    if should_ignore_file(file_path):
                        ignored_files += 1
                        self.logger.debug(f"Ignoring file: {file_path}")
                        continue
                    
                    file_key = str(file_path.absolute())
                    
                    # Check if file was already processed
                    if file_key in self.processed_files:
                        already_processed += 1
                        self.logger.debug(f"File already processed: {file_path}")
                        continue
                    
                    files_to_process.append(file_key)
                    self.logger.debug(f"Added for processing: {file_path}")
            
            self.logger.info(f"Scan complete - Total files: {total_files}, Ignored: {ignored_files}, Already processed: {already_processed}, New to process: {len(files_to_process)}")
            return files_to_process
            
        except Exception as e:
            self.logger.error(f"Error scanning input directory: {e}")
            return []
    
    def organize_files(self, file_paths: List[str]) -> Dict:

        if not file_paths:
            self.logger.info("No files to organize")
            return {"processed": 0, "errors": 0, "categories": {}}
        
        self.logger.info(f"Starting organization of {len(file_paths)} files")
        
        # Analyze files with AI
        analysis_results = self.ai_analyzer.batch_analyze(file_paths)
        
        # Create category directories with subdirectories
        categories = set(result.get("category", "Reference_Documentation") for result in analysis_results if "error" not in result)
        
        # Build subdirectory structure
        subdirectories = {}
        for result in analysis_results:
            if "error" not in result:
                category = result.get("category", "Reference_Documentation")
                subdirectory = result.get("subdirectory", "General")
                
                if category not in subdirectories:
                    subdirectories[category] = set()
                subdirectories[category].add(subdirectory)
        
        # Convert sets to lists for the create function
        subdirectories = {k: list(v) for k, v in subdirectories.items()}
        
        create_directory_structure(str(self.output_dir), list(categories), subdirectories)
        
        # Organize files
        results = {"processed": 0, "errors": 0, "categories": {}}
        
        for analysis in analysis_results:
            if "error" in analysis:
                self.logger.error(f"Analysis error: {analysis['error']}")
                results["errors"] += 1
                continue
            
            success = self._move_file_to_category(analysis)
            if success:
                results["processed"] += 1
                category = analysis["category"]
                results["categories"][category] = results["categories"].get(category, 0) + 1
                
                # Record as processed
                file_key = analysis["file_path"]
                self.processed_files[file_key] = {
                    "category": category,
                    "timestamp": analysis["analysis_timestamp"],
                    "confidence": analysis["confidence"]
                }
            else:
                results["errors"] += 1
        
        # Save processed files record
        save_processed_files(self.processed_files)
        
        # Generate and log statistics
        stats = self.ai_analyzer.get_category_statistics(analysis_results)
        self._log_organization_stats(stats, results)
        
        return results
    
    def _move_file_to_category(self, analysis: Dict) -> bool:
        """
        Move a file to its categorized directory and subdirectory.
        
        Args:
            analysis: File analysis result
            
        Returns:
            True if successful, False otherwise
        """
        try:
            source_path = Path(analysis["file_path"])
            category = analysis["category"]
            subdirectory = analysis.get("subdirectory", "General")
            
            # Create destination path with subdirectory
            if subdirectory and subdirectory != "General":
                dest_dir = self.output_dir / category / subdirectory
                dest_path = dest_dir / source_path.name
                log_path = f"{category}/{subdirectory}/"
            else:
                dest_dir = self.output_dir / category
                dest_path = dest_dir / source_path.name
                log_path = f"{category}/"
            
            # Move the file
            success = move_file(str(source_path), str(dest_path))
            
            if success:
                self.logger.info(f"Moved {source_path.name} to {log_path}")
                return True
            else:
                self.logger.error(f"Failed to move {source_path.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error moving file {analysis.get('file_name', 'unknown')}: {e}")
            return False
    
    def _log_organization_stats(self, analysis_stats: Dict, organization_results: Dict):
        """
        Log detailed statistics about the organization process.
        
        Args:
            analysis_stats: Statistics from AI analysis
            organization_results: Results from file organization
        """
        self.logger.info("=== Organization Statistics ===")
        self.logger.info(f"Total files analyzed: {analysis_stats['total_files']}")
        self.logger.info(f"Files successfully organized: {organization_results['processed']}")
        self.logger.info(f"Errors encountered: {organization_results['errors']}")
        self.logger.info(f"Average confidence: {analysis_stats['average_confidence']:.2f}")
        
        self.logger.info("=== Categories and Subdirectories ===")
        for category, count in organization_results['categories'].items():
            self.logger.info(f"  {category}: {count} files")
            
            # Show subdirectory breakdown if available
            category_path = self.output_dir / category
            if category_path.exists():
                subdirs = [d for d in category_path.iterdir() if d.is_dir()]
                if subdirs:
                    for subdir in subdirs:
                        subdir_files = len([f for f in subdir.rglob("*") if f.is_file()])
                        if subdir_files > 0:
                            self.logger.info(f"    └── {subdir.name}: {subdir_files} files")
        
        if analysis_stats.get('errors', 0) > 0:
            self.logger.warning(f"Analysis errors: {analysis_stats['errors']}")
    
    def watch_mode(self, interval: int = 30):

        self.logger.info(f"Starting watch mode (checking every {interval} seconds)")
        self.logger.info("Press Ctrl+C to stop")
        
        self.running = True
        
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            self.logger.info("Shutdown signal received, stopping...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while self.running:
                files_to_process = self.scan_input_directory()
                
                if files_to_process:
                    self.organize_files(files_to_process)
                else:
                    self.logger.debug("No new files found")
                
                # Wait for next check
                for _ in range(interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in watch mode: {e}")
        finally:
            self.running = False
            self.logger.info("Watch mode stopped")
    
    def one_time_scan(self):

        self.logger.info("Performing one-time scan")
        files_to_process = self.scan_input_directory()
        
        if files_to_process:
            results = self.organize_files(files_to_process)
            self.logger.info(f"One-time scan completed: {results['processed']} files organized")
        else:
            self.logger.info("No files to process")
    
    def status_report(self):

        self.logger.info("=== AutoOrganizer Status Report ===")
        
        # Input directory status
        input_files = list(self.input_dir.rglob("*"))
        input_file_count = len([f for f in input_files if f.is_file()])
        
        self.logger.info(f"Input directory: {self.input_dir.absolute()}")
        self.logger.info(f"Files in input directory: {input_file_count}")
        
        # Output directory status
        if self.output_dir.exists():
            categories = [d for d in self.output_dir.iterdir() if d.is_dir()]
            self.logger.info(f"Output directory: {self.output_dir.absolute()}")
            self.logger.info(f"Categories created: {len(categories)}")
            
            total_organized = 0
            for category_dir in categories:
                category_files = 0
                subdirs = [d for d in category_dir.iterdir() if d.is_dir()]
                
                if subdirs:
                    # Count files in subdirectories
                    for subdir in subdirs:
                        subdir_files = len([f for f in subdir.rglob("*") if f.is_file()])
                        category_files += subdir_files
                        if subdir_files > 0:
                            self.logger.info(f"    {category_dir.name}/{subdir.name}: {subdir_files} files")
                    
                    # Count files directly in category (not in subdirs)
                    direct_files = len([f for f in category_dir.iterdir() if f.is_file()])
                    category_files += direct_files
                    if direct_files > 0:
                        self.logger.info(f"    {category_dir.name}/: {direct_files} files")
                else:
                    # No subdirectories, count all files
                    category_files = len([f for f in category_dir.rglob("*") if f.is_file()])
                    self.logger.info(f"  {category_dir.name}: {category_files} files")
                
                total_organized += category_files
            
            self.logger.info(f"Total organized files: {total_organized}")
        
        # Processed files record
        self.logger.info(f"Total processed files recorded: {len(self.processed_files)}")
        
        # AI Analyzer status
        if hasattr(self.ai_analyzer, 'model') and self.ai_analyzer.model:
            self.logger.info(f"AI Model: {self.ai_analyzer.model_name} (loaded)")
        else:
            self.logger.info("AI Model: Not available (using rule-based categorization)")
    
    def clean_processed_records(self):

        self.logger.info("Cleaning processed file records")
        
        initial_count = len(self.processed_files)
        files_to_remove = []
        
        for file_path in self.processed_files.keys():
            if not Path(file_path).exists():
                files_to_remove.append(file_path)
        
        for file_path in files_to_remove:
            del self.processed_files[file_path]
        
        if files_to_remove:
            save_processed_files(self.processed_files)
            self.logger.info(f"Removed {len(files_to_remove)} obsolete records")
        else:
            self.logger.info("No cleanup needed")
    
    def show_directory_structure(self):
        """
        Display the complete directory structure that was created.
        """
        self.logger.info("=== Complete Directory Structure ===")
        
        if not self.output_dir.exists():
            self.logger.info("No organized files directory found.")
            return
        
        total_files = 0
        categories = sorted([d for d in self.output_dir.iterdir() if d.is_dir()])
        
        for category_dir in categories:
            category_files = 0
            subdirs = sorted([d for d in category_dir.iterdir() if d.is_dir()])
            
            # Files directly in category
            direct_files = [f for f in category_dir.iterdir() if f.is_file()]
            category_files += len(direct_files)
            
            self.logger.info(f"📁 {category_dir.name}/")
            
            if direct_files:
                self.logger.info(f"  📄 {len(direct_files)} files")
            
            # Show subdirectories
            for subdir in subdirs:
                subdir_files = [f for f in subdir.rglob("*") if f.is_file()]
                category_files += len(subdir_files)
                
                if subdir_files:
                    self.logger.info(f"  📁 {subdir.name}/")
                    self.logger.info(f"    📄 {len(subdir_files)} files")
                    
                    # Show some example files (first 3)
                    for i, file in enumerate(subdir_files[:3]):
                        self.logger.info(f"      • {file.name}")
                    if len(subdir_files) > 3:
                        self.logger.info(f"      ... and {len(subdir_files) - 3} more files")
            
            total_files += category_files
            if category_files > 0:
                self.logger.info(f"  Total: {category_files} files")
            self.logger.info("")  # Empty line for readability
        
        self.logger.info(f"📊 Grand Total: {total_files} organized files")


def main():

    parser = argparse.ArgumentParser(
        description="AutoOrganizer - Automatic File Organization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # One-time scan and organize
  python main.py --watch                  # Continuous monitoring mode
  python main.py --watch --interval 60    # Monitor every 60 seconds
  python main.py --status                 # Show status report
  python main.py --show-structure         # Show complete directory structure
  python main.py --clean                  # Clean up processed records
        """
    )
    
    parser.add_argument(
        "--input-dir", 
        default="input_files",
        help="Input directory to monitor (default: input_files)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="organized_files",
        help="Output directory for organized files (default: organized_files)"
    )
    
    parser.add_argument(
        "--watch", 
        action="store_true",
        help="Run in watch mode (continuous monitoring)"
    )
    
    parser.add_argument(
        "--interval", 
        type=int, 
        default=30,
        help="Check interval in seconds for watch mode (default: 30)"
    )
    
    parser.add_argument(
        "--status", 
        action="store_true",
        help="Show status report and exit"
    )
    
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Clean up processed file records"
    )
    
    parser.add_argument(
        "--show-structure", 
        action="store_true",
        help="Show the complete directory structure"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    try:
        organizer = AutoOrganizer(args.input_dir, args.output_dir)
        
        if args.verbose:
            organizer.logger.setLevel(logging.DEBUG)
        
        if args.status:
            organizer.status_report()
        elif args.clean:
            organizer.clean_processed_records()
        elif args.show_structure:
            organizer.show_directory_structure()
        elif args.watch:
            organizer.watch_mode(args.interval)
        else:
            organizer.one_time_scan()
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()