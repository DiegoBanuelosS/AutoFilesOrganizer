"""
AI-powered file analyzer for the AutoOrganizer system.
Uses transformer models and machine learning to categorize files intelligently.
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from utils import get_file_info, sanitize_filename


class AIFileAnalyzer:
    """
    AI-powered file analyzer that categorizes files based on content and metadata.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the AI file analyzer.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.logger = logging.getLogger("AutoOrganizer.AIAnalyzer")
        self.model_name = model_name
        self.model = None
        self.categories = self._get_default_categories()
        self.category_embeddings = None
        
        if DEPENDENCIES_AVAILABLE:
            self._initialize_model()
        else:
            self.logger.warning("AI dependencies not available. Using fallback categorization.")
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self._precompute_category_embeddings()
            self.logger.info("AI model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load AI model: {e}")
            self.model = None
    
    def _get_default_categories(self) -> Dict[str, Dict]:
        """
        Get topic-based categories with their descriptions and contextual information.
        
        Returns:
            Dictionary of categories with metadata
        """
        return {
            "Work_Projects": {
                "description": "Professional work documents, business reports, project files, meeting notes, presentations for work",
                "extensions": [".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".pdf"],
                "keywords": ["project", "meeting", "report", "business", "work", "presentation", "proposal", "contract", "invoice", "budget", "quarterly", "annual", "team", "client", "deadline"],
                "subdirectories": {
                    "Reports": ["report", "quarterly", "annual", "summary", "analysis", "review"],
                    "Presentations": ["presentation", "slides", "ppt", "keynote", "demo"],
                    "Meetings": ["meeting", "agenda", "minutes", "notes", "discussion"],
                    "Projects": ["project", "plan", "timeline", "milestone", "deliverable"],
                    "Contracts": ["contract", "agreement", "legal", "terms", "conditions"],
                    "Finance": ["budget", "invoice", "expense", "cost", "financial", "accounting"],
                    "Communications": ["email", "correspondence", "memo", "announcement"],
                    "General": []
                }
            },
            "Personal_Documents": {
                "description": "Personal files, letters, resumes, personal notes, certificates, personal records",
                "extensions": [".pdf", ".doc", ".docx", ".txt"],
                "keywords": ["personal", "resume", "cv", "letter", "certificate", "passport", "license", "birth", "marriage", "insurance", "medical", "family", "diary", "journal"],
                "subdirectories": {
                    "Identity": ["passport", "license", "id", "birth", "certificate", "legal"],
                    "Career": ["resume", "cv", "job", "employment", "career", "application"],
                    "Medical": ["medical", "health", "doctor", "prescription", "insurance"],
                    "Financial": ["bank", "tax", "insurance", "loan", "investment", "personal_finance"],
                    "Family": ["family", "marriage", "birth", "relationship", "genealogy"],
                    "Personal_Notes": ["diary", "journal", "notes", "thoughts", "personal"],
                    "General": []
                }
            },
            "Education_Learning": {
                "description": "Educational materials, courses, tutorials, research papers, study notes, academic content",
                "extensions": [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".txt", ".mp4", ".mp3"],
                "keywords": ["education", "course", "tutorial", "study", "learning", "university", "college", "research", "thesis", "homework", "assignment", "lecture", "exam", "book", "textbook", "academic"],
                "subdirectories": {
                    "Courses": ["course", "class", "semester", "module", "curriculum"],
                    "Research": ["research", "thesis", "paper", "study", "academic", "journal"],
                    "Tutorials": ["tutorial", "guide", "how-to", "lesson", "training"],
                    "Books": ["book", "textbook", "ebook", "manual", "reference"],
                    "Assignments": ["homework", "assignment", "exercise", "project", "lab"],
                    "Exams": ["exam", "test", "quiz", "assessment", "evaluation"],
                    "Notes": ["notes", "lecture", "summary", "review", "study_notes"],
                    "Certificates": ["certificate", "diploma", "degree", "achievement", "completion"],
                    "General": []
                }
            },
            "Development_Code": {
                "description": "Programming projects, source code, development tools, software projects, scripts",
                "extensions": [".py", ".js", ".html", ".css", ".cpp", ".java", ".php", ".rb", ".go", ".rs", ".swift", ".json", ".xml", ".sql"],
                "keywords": ["code", "programming", "development", "software", "app", "website", "script", "function", "class", "variable", "database", "api", "framework", "library", "github", "repository"],
                "subdirectories": {
                    "Web_Development": ["html", "css", "javascript", "web", "frontend", "backend", "react", "vue", "angular"],
                    "Python_Projects": ["python", ".py", "django", "flask", "pandas", "numpy"],
                    "Mobile_Development": ["android", "ios", "mobile", "app", "swift", "kotlin"],
                    "Database": ["sql", "database", "db", "mysql", "postgresql", "mongodb"],
                    "Scripts": ["script", "automation", "tool", "utility", "batch"],
                    "APIs": ["api", "rest", "graphql", "endpoint", "service"],
                    "Documentation": ["readme", "docs", "documentation", "api_docs"],
                    "Configuration": ["config", "settings", "env", "docker", "yaml", "json"],
                    "General": []
                }
            },
            "Creative_Projects": {
                "description": "Creative work, art projects, design files, photography, videos, music, creative writing",
                "extensions": [".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mp3", ".wav", ".psd", ".ai", ".svg"],
                "keywords": ["creative", "art", "design", "photo", "photography", "video", "music", "drawing", "painting", "creative writing", "portfolio", "artwork", "graphic", "logo", "brand", "illustration"],
                "subdirectories": {
                    "Photography": ["photo", "photography", "camera", "portrait", "landscape", "wedding"],
                    "Graphic_Design": ["logo", "brand", "graphic", "design", "poster", "flyer"],
                    "Video_Projects": ["video", "movie", "film", "editing", "animation", "motion"],
                    "Music": ["music", "audio", "song", "composition", "recording", "soundtrack"],
                    "Art": ["art", "drawing", "painting", "sketch", "illustration", "digital_art"],
                    "Writing": ["story", "poem", "creative_writing", "script", "novel", "blog"],
                    "Portfolio": ["portfolio", "showcase", "gallery", "collection"],
                    "General": []
                }
            },
            "Finance_Money": {
                "description": "Financial documents, bank statements, taxes, investments, budgets, receipts",
                "extensions": [".pdf", ".xls", ".xlsx", ".csv"],
                "keywords": ["finance", "money", "bank", "tax", "taxes", "investment", "budget", "receipt", "invoice", "payment", "salary", "income", "expense", "financial", "accounting", "loan", "mortgage"],
                "subdirectories": {
                    "Banking": ["bank", "statement", "account", "deposit", "withdrawal", "balance"],
                    "Taxes": ["tax", "taxes", "irs", "deduction", "refund", "1040", "w2"],
                    "Investments": ["investment", "stock", "bond", "portfolio", "trading", "retirement"],
                    "Budget": ["budget", "expense", "income", "spending", "financial_plan"],
                    "Receipts": ["receipt", "purchase", "expense", "shopping", "bill"],
                    "Insurance": ["insurance", "policy", "claim", "coverage", "premium"],
                    "Loans": ["loan", "mortgage", "credit", "debt", "payment", "refinance"],
                    "Business_Finance": ["business", "invoice", "revenue", "profit", "accounting"],
                    "General": []
                }
            },
            "Health_Medical": {
                "description": "Medical records, health documents, fitness data, medical reports, prescriptions",
                "extensions": [".pdf", ".doc", ".docx", ".jpg", ".png"],
                "keywords": ["health", "medical", "doctor", "hospital", "prescription", "medicine", "fitness", "exercise", "diet", "wellness", "therapy", "treatment", "diagnosis", "symptoms", "appointment"],
                "subdirectories": {
                    "Medical_Records": ["medical", "record", "history", "chart", "diagnosis", "treatment"],
                    "Prescriptions": ["prescription", "medicine", "medication", "pharmacy", "drug"],
                    "Lab_Results": ["lab", "test", "blood", "result", "analysis", "screening"],
                    "Appointments": ["appointment", "schedule", "doctor", "clinic", "visit"],
                    "Insurance": ["insurance", "coverage", "claim", "policy", "benefits"],
                    "Fitness": ["fitness", "exercise", "workout", "gym", "training", "health"],
                    "Diet_Nutrition": ["diet", "nutrition", "food", "meal", "calories", "weight"],
                    "Mental_Health": ["therapy", "counseling", "mental", "psychology", "wellness"],
                    "General": []
                }
            },
            "Entertainment_Media": {
                "description": "Movies, TV shows, games, entertainment content, recreational media",
                "extensions": [".mp4", ".avi", ".mkv", ".mp3", ".wav", ".jpg", ".png"],
                "keywords": ["entertainment", "movie", "film", "tv", "show", "game", "gaming", "music", "song", "album", "fun", "leisure", "hobby", "recreation", "comedy", "drama", "action"],
                "subdirectories": {
                    "Movies": ["movie", "film", "cinema", "dvd", "bluray", "action", "comedy", "drama"],
                    "TV_Shows": ["tv", "show", "series", "episode", "season", "television"],
                    "Music": ["music", "song", "album", "artist", "playlist", "concert"],
                    "Games": ["game", "gaming", "video_game", "pc", "console", "mobile"],
                    "Books": ["book", "novel", "fiction", "non-fiction", "audiobook", "ebook"],
                    "Podcasts": ["podcast", "audio", "talk", "interview", "discussion"],
                    "Sports": ["sport", "football", "basketball", "soccer", "baseball", "hockey"],
                    "General": []
                }
            },
            "Travel_Adventure": {
                "description": "Travel documents, itineraries, maps, travel photos, vacation planning, trip records",
                "extensions": [".pdf", ".jpg", ".png", ".doc", ".docx"],
                "keywords": ["travel", "trip", "vacation", "holiday", "flight", "hotel", "booking", "itinerary", "passport", "visa", "map", "destination", "adventure", "journey", "tourism"],
                "subdirectories": {
                    "Trip_Planning": ["itinerary", "plan", "booking", "reservation", "schedule"],
                    "Documents": ["passport", "visa", "ticket", "boarding", "confirmation"],
                    "Photos": ["photo", "picture", "vacation", "trip", "destination", "memory"],
                    "Receipts": ["receipt", "expense", "hotel", "restaurant", "souvenir"],
                    "Maps": ["map", "direction", "route", "navigation", "guide"],
                    "Accommodation": ["hotel", "airbnb", "hostel", "lodge", "accommodation"],
                    "Transportation": ["flight", "train", "car", "bus", "transportation"],
                    "General": []
                }
            },
            "Archive_Backup": {
                "description": "Compressed files, backups, old files, archived content, storage",
                "extensions": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"],
                "keywords": ["archive", "backup", "compressed", "old", "storage", "zip", "backup", "restore", "copy", "duplicate", "historical"],
                "subdirectories": {
                    "System_Backups": ["system", "backup", "restore", "image", "full"],
                    "File_Archives": ["archive", "old", "historical", "compressed", "storage"],
                    "Project_Archives": ["project", "old_project", "completed", "archived"],
                    "Personal_Backup": ["personal", "backup", "family", "important"],
                    "General": []
                }
            },
            "Tools_Utilities": {
                "description": "Software tools, utilities, installers, system files, configuration files",
                "extensions": [".exe", ".msi", ".dmg", ".deb", ".ini", ".cfg", ".conf"],
                "keywords": ["tool", "utility", "installer", "software", "program", "system", "config", "configuration", "setup", "application", "executable"],
                "subdirectories": {
                    "Software": ["software", "program", "application", "tool", "utility"],
                    "Installers": ["installer", "setup", "install", "msi", "exe", "dmg"],
                    "Configuration": ["config", "configuration", "settings", "preferences"],
                    "System": ["system", "driver", "patch", "update", "os"],
                    "Development_Tools": ["dev", "development", "compiler", "editor", "ide"],
                    "General": []
                }
            },
            "Reference_Documentation": {
                "description": "Reference materials, manuals, guides, documentation, specifications",
                "extensions": [".pdf", ".doc", ".docx", ".txt", ".html"],
                "keywords": ["reference", "manual", "guide", "documentation", "specification", "instruction", "help", "tutorial", "readme", "wiki", "encyclopedia", "dictionary"],
                "subdirectories": {
                    "Manuals": ["manual", "instruction", "guide", "handbook", "user_guide"],
                    "Specifications": ["specification", "spec", "standard", "protocol", "format"],
                    "References": ["reference", "dictionary", "encyclopedia", "lookup", "catalog"],
                    "Tutorials": ["tutorial", "how-to", "guide", "lesson", "walkthrough"],
                    "Documentation": ["documentation", "docs", "readme", "wiki", "help"],
                    "General": []
                }
            }
        }
    
    def _precompute_category_embeddings(self):
        """Precompute embeddings for all categories."""
        if not self.model:
            return
        
        try:
            category_texts = []
            for category, info in self.categories.items():
                # Combine category name, description, and keywords
                text = f"{category} {info['description']} {' '.join(info['keywords'])}"
                category_texts.append(text)
            
            self.category_embeddings = self.model.encode(category_texts)
            self.logger.info("Category embeddings precomputed successfully")
        except Exception as e:
            self.logger.error(f"Failed to precompute category embeddings: {e}")
            self.category_embeddings = None
    
    def analyze_file(self, file_path: str) -> Dict:
        """
        Analyze a single file and determine its category using AI and contextual analysis.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Get basic file information
            file_info = get_file_info(file_path)
            
            if "error" in file_info:
                return {"error": file_info["error"]}
            
            # Analyze file context
            context = self.analyze_file_context(file_path)
            
            # Determine category using AI and enhanced rules
            if self.model and self.category_embeddings is not None:
                category = self._categorize_with_ai(file_info)
                method_used = "AI"
            else:
                category = self._categorize_with_enhanced_rules(file_info)
                method_used = "Enhanced_Rules"
            
            # Apply context-based refinements
            category = self._refine_with_context(category, context, file_info)
            
            # Determine specific subdirectory
            subdirectory = self.determine_subdirectory(file_info, category)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(file_info, category)
            
            # Boost confidence if context supports the decision
            if context["path_hints"] and any(hint in category.lower() for hint in context["path_hints"]):
                confidence = min(confidence + 0.1, 1.0)
            
            result = {
                "file_path": file_path,
                "file_name": file_info["name"],
                "category": category,
                "subdirectory": subdirectory,
                "full_path": f"{category}/{subdirectory}" if subdirectory != "General" else category,
                "confidence": confidence,
                "file_size": file_info["size"],
                "file_type": file_info.get("mime_type", "unknown"),
                "analysis_timestamp": datetime.now().isoformat(),
                "method_used": method_used,
                "context_hints": context,
                "reasoning": self._generate_reasoning(file_info, category, context, subdirectory)
            }
            
            self.logger.info(f"Analyzed {file_info['name']} -> {category} (confidence: {confidence:.2f}, method: {method_used})")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze file {file_path}: {e}")
            return {"error": str(e)}
    
    def _categorize_with_ai(self, file_info: Dict) -> str:
        """
        Categorize file using AI/ML models based on content and context.
        
        Args:
            file_info: File information dictionary
            
        Returns:
            Predicted category name
        """
        try:
            # Create comprehensive text representation of the file
            file_text = self._create_comprehensive_file_representation(file_info)
            
            # Get embedding for the file
            file_embedding = self.model.encode([file_text])
            
            # Calculate similarities with category embeddings
            similarities = cosine_similarity(file_embedding, self.category_embeddings)[0]
            
            # Get top 3 most similar categories for better decision making
            top_indices = np.argsort(similarities)[-3:][::-1]
            category_names = list(self.categories.keys())
            
            # Apply contextual rules to refine the AI decision
            best_category = self._apply_contextual_rules(file_info, similarities, category_names)
            
            return best_category
            
        except Exception as e:
            self.logger.error(f"AI categorization failed: {e}")
            return self._categorize_with_enhanced_rules(file_info)
    
    def _create_comprehensive_file_representation(self, file_info: Dict) -> str:
        """
        Create a comprehensive text representation of the file for AI analysis.
        
        Args:
            file_info: File information dictionary
            
        Returns:
            Text representation of the file
        """
        parts = []
        
        # File name analysis - extract meaningful parts
        filename = file_info.get('name', '')
        stem = file_info.get('stem', '')
        
        # Split filename into meaningful components
        filename_parts = []
        for part in stem.replace('_', ' ').replace('-', ' ').replace('.', ' ').split():
            if len(part) > 2:  # Filter out very short parts
                filename_parts.append(part)
        
        parts.append(f"filename: {' '.join(filename_parts)}")
        parts.append(f"extension: {file_info.get('suffix', '')}")
        
        # File type and format
        if file_info.get("mime_type"):
            parts.append(f"type: {file_info['mime_type']}")
        
        # Content analysis for text files
        if file_info.get("is_text", False) and file_info.get("text_content"):
            content = file_info["text_content"]
            
            # Extract key phrases and context
            content_keywords = self._extract_content_keywords(content)
            if content_keywords:
                parts.append(f"content_keywords: {' '.join(content_keywords)}")
            
            # Add first few lines of content for context
            lines = content.split('\n')[:5]  # First 5 lines
            meaningful_lines = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
            if meaningful_lines:
                parts.append(f"content_preview: {' '.join(meaningful_lines[:2])}")
        
        # File path context
        file_path = file_info.get('path', '')
        path_parts = []
        for part in Path(file_path).parts:
            if part not in ['/', '\\', 'C:', 'Users'] and len(part) > 2:
                path_parts.append(part)
        if path_parts:
            parts.append(f"path_context: {' '.join(path_parts[-3:])}")  # Last 3 path components
        
        return " ".join(parts)
    
    def _extract_content_keywords(self, content: str) -> List[str]:
        """
        Extract meaningful keywords from file content.
        
        Args:
            content: Text content of the file
            
        Returns:
            List of extracted keywords
        """
        import re
        
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        # Extract words and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        keywords = []
        
        # Count word frequency
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most frequent meaningful words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:15] if freq > 1]  # Top 15 words that appear more than once
        
        return keywords
    
    def _apply_contextual_rules(self, file_info: Dict, similarities: np.ndarray, category_names: List[str]) -> str:
        """
        Apply contextual rules to refine AI categorization decision.
        
        Args:
            file_info: File information dictionary
            similarities: Similarity scores for each category
            category_names: List of category names
            
        Returns:
            Refined category name
        """
        filename = file_info.get('name', '').lower()
        extension = file_info.get('suffix', '').lower()
        content = file_info.get('text_content', '').lower()
        
        # Get top 3 categories by similarity
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_categories = [category_names[i] for i in top_indices]
        
        # Strong contextual rules that can override AI decision
        
        # Financial keywords have high priority
        financial_keywords = ['invoice', 'receipt', 'tax', 'bank', 'payment', 'budget', 'expense', 'income', 'salary']
        if any(keyword in filename or keyword in content for keyword in financial_keywords):
            if 'Finance_Money' in top_categories[:2]:  # If it's in top 2, choose it
                return 'Finance_Money'
        
        # Medical/Health keywords
        medical_keywords = ['medical', 'doctor', 'hospital', 'prescription', 'health', 'medicine']
        if any(keyword in filename or keyword in content for keyword in medical_keywords):
            if 'Health_Medical' in top_categories[:2]:
                return 'Health_Medical'
        
        # Development/Code files - strong extension matching
        if extension in ['.py', '.js', '.html', '.css', '.cpp', '.java', '.php', '.rb', '.go', '.json', '.xml', '.sql']:
            return 'Development_Code'
        
        # Work-related keywords
        work_keywords = ['project', 'meeting', 'business', 'report', 'presentation', 'client', 'proposal']
        if any(keyword in filename or keyword in content for keyword in work_keywords):
            if 'Work_Projects' in top_categories[:2]:
                return 'Work_Projects'
        
        # Education/Learning keywords
        education_keywords = ['course', 'tutorial', 'study', 'university', 'college', 'research', 'homework', 'assignment']
        if any(keyword in filename or keyword in content for keyword in education_keywords):
            if 'Education_Learning' in top_categories[:2]:
                return 'Education_Learning'
        
        # Creative keywords
        creative_keywords = ['photo', 'design', 'art', 'creative', 'portfolio', 'drawing', 'music', 'video']
        if any(keyword in filename or keyword in content for keyword in creative_keywords):
            if 'Creative_Projects' in top_categories[:2]:
                return 'Creative_Projects'
        
        # Travel keywords
        travel_keywords = ['travel', 'trip', 'vacation', 'flight', 'hotel', 'booking', 'itinerary']
        if any(keyword in filename or keyword in content for keyword in travel_keywords):
            if 'Travel_Adventure' in top_categories[:2]:
                return 'Travel_Adventure'
        
        # If no strong contextual match, return the AI's top choice
        return top_categories[0]
    def _categorize_with_enhanced_rules(self, file_info: Dict) -> str:
        """
        Enhanced rule-based categorization focusing on content and context.
        
        Args:
            file_info: File information dictionary
            
        Returns:
            Predicted category name
        """
        extension = file_info.get("suffix", "").lower()
        file_name = file_info.get("name", "").lower()
        content = file_info.get("text_content", "").lower() if file_info.get("is_text", False) else ""
        
        # Priority-based categorization
        
        # 1. Development/Code files (highest priority for code extensions)
        if extension in ['.py', '.js', '.html', '.css', '.cpp', '.java', '.php', '.rb', '.go', '.rs', '.swift', '.json', '.xml', '.sql']:
            return "Development_Code"
        
        # 2. Check content-based keywords for text files
        if content:
            # Financial content
            financial_patterns = ['invoice', 'receipt', 'tax', 'bank', 'payment', 'budget', 'expense', 'income', 'salary', 'financial', 'money', 'cost', 'price', 'billing']
            if any(pattern in content or pattern in file_name for pattern in financial_patterns):
                return "Finance_Money"
            
            # Work/Business content
            work_patterns = ['project', 'meeting', 'business', 'report', 'presentation', 'client', 'proposal', 'contract', 'deadline', 'team', 'manager', 'professional']
            if any(pattern in content or pattern in file_name for pattern in work_patterns):
                return "Work_Projects"
            
            # Educational content
            education_patterns = ['course', 'tutorial', 'study', 'university', 'college', 'research', 'homework', 'assignment', 'lecture', 'exam', 'grade', 'student', 'teacher']
            if any(pattern in content or pattern in file_name for pattern in education_patterns):
                return "Education_Learning"
            
            # Medical/Health content
            medical_patterns = ['medical', 'doctor', 'hospital', 'prescription', 'health', 'medicine', 'treatment', 'diagnosis', 'symptoms', 'therapy', 'clinic']
            if any(pattern in content or pattern in file_name for pattern in medical_patterns):
                return "Health_Medical"
            
            # Travel content
            travel_patterns = ['travel', 'trip', 'vacation', 'flight', 'hotel', 'booking', 'itinerary', 'passport', 'visa', 'destination', 'journey']
            if any(pattern in content or pattern in file_name for pattern in travel_patterns):
                return "Travel_Adventure"
            
            # Creative content
            creative_patterns = ['creative', 'art', 'design', 'photo', 'photography', 'drawing', 'painting', 'music', 'video', 'portfolio', 'artwork', 'graphic']
            if any(pattern in content or pattern in file_name for pattern in creative_patterns):
                return "Creative_Projects"
        
        # 3. Extension-based fallback categorization
        if extension in ['.pdf', '.doc', '.docx'] and any(pattern in file_name for pattern in ['personal', 'resume', 'cv', 'letter']):
            return "Personal_Documents"
        
        # Creative file types
        if extension in ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3', '.wav', '.psd', '.ai', '.svg']:
            return "Creative_Projects"
        
        # Archive files
        if extension in ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2']:
            return "Archive_Backup"
        
        # Tools and executables
        if extension in ['.exe', '.msi', '.dmg', '.deb']:
            return "Tools_Utilities"
        
        # Configuration files
        if extension in ['.ini', '.cfg', '.conf', '.config', '.yaml', '.yml']:
            return "Tools_Utilities"
        
        # Documents (general)
        if extension in ['.pdf', '.doc', '.docx', '.txt', '.rtf']:
            return "Reference_Documentation"
        
        # Data files
        if extension in ['.csv', '.xlsx', '.xls']:
            return "Finance_Money"  # Spreadsheets often contain financial data
        
        # Entertainment media
        if extension in ['.mp4', '.avi', '.mkv', '.mp3', '.wav'] and any(pattern in file_name for pattern in ['movie', 'song', 'music', 'entertainment', 'game']):
            return "Entertainment_Media"
        
        # Default fallback
        return "Reference_Documentation"
    
    def _create_file_text_representation(self, file_info: Dict) -> str:
        """
        Create a text representation of the file for AI analysis (legacy method).
        
        Args:
            file_info: File information dictionary
            
        Returns:
            Text representation of the file
        """
        return self._create_comprehensive_file_representation(file_info)
    
    def _calculate_confidence(self, file_info: Dict, category: str) -> float:
        """
        Calculate confidence score for the categorization based on multiple factors.
        
        Args:
            file_info: File information dictionary
            category: Predicted category
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.3  # Base confidence
        
        extension = file_info.get("suffix", "").lower()
        file_name = file_info.get("name", "").lower()
        content = file_info.get("text_content", "").lower() if file_info.get("is_text", False) else ""
        
        # Strong extension matches
        if extension in self.categories[category]["extensions"]:
            confidence += 0.4
        
        # Keyword matches in filename (weighted)
        filename_matches = 0
        for keyword in self.categories[category]["keywords"]:
            if keyword in file_name:
                filename_matches += 1
        
        if filename_matches > 0:
            confidence += min(0.3, filename_matches * 0.1)
        
        # Content keyword matches (for text files)
        if content:
            content_matches = 0
            for keyword in self.categories[category]["keywords"]:
                if keyword in content:
                    content_matches += 1
            
            if content_matches > 0:
                confidence += min(0.2, content_matches * 0.05)
        
        # Special boosts for strong indicators
        special_indicators = {
            "Development_Code": ['.py', '.js', '.html', '.css', '.cpp', '.java'],
            "Finance_Money": ['invoice', 'receipt', 'tax', 'bank', 'payment'],
            "Health_Medical": ['medical', 'doctor', 'hospital', 'prescription'],
            "Work_Projects": ['project', 'meeting', 'business', 'report'],
            "Education_Learning": ['course', 'tutorial', 'study', 'university']
        }
        
        if category in special_indicators:
            indicators = special_indicators[category]
            if any(indicator in file_name or indicator in content or extension == indicator for indicator in indicators):
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def batch_analyze(self, file_paths: List[str]) -> List[Dict]:
        """
        Analyze multiple files in batch.
        
        Args:
            file_paths: List of file paths to analyze
            
        Returns:
            List of analysis results
        """
        results = []
        total_files = len(file_paths)
        
        self.logger.info(f"Starting batch analysis of {total_files} files")
        
        for i, file_path in enumerate(file_paths, 1):
            self.logger.info(f"Processing file {i}/{total_files}: {Path(file_path).name}")
            result = self.analyze_file(file_path)
            results.append(result)
        
        self.logger.info(f"Batch analysis completed: {total_files} files processed")
        return results
    
    def get_category_statistics(self, analysis_results: List[Dict]) -> Dict:
        """
        Generate statistics from analysis results.
        
        Args:
            analysis_results: List of analysis results
            
        Returns:
            Dictionary containing statistics
        """
        stats = {
            "total_files": len(analysis_results),
            "categories": {},
            "average_confidence": 0.0,
            "errors": 0
        }
        
        confidences = []
        
        for result in analysis_results:
            if "error" in result:
                stats["errors"] += 1
                continue
            
            category = result.get("category", "Unknown")
            confidence = result.get("confidence", 0.0)
            
            if category not in stats["categories"]:
                stats["categories"][category] = {
                    "count": 0,
                    "files": [],
                    "average_confidence": 0.0
                }
            
            stats["categories"][category]["count"] += 1
            stats["categories"][category]["files"].append(result["file_name"])
            confidences.append(confidence)
        
        # Calculate average confidences
        if confidences:
            stats["average_confidence"] = sum(confidences) / len(confidences)
            
            for category in stats["categories"]:
                category_confidences = [
                    r.get("confidence", 0.0) for r in analysis_results 
                    if r.get("category") == category and "error" not in r
                ]
                if category_confidences:
                    stats["categories"][category]["average_confidence"] = sum(category_confidences) / len(category_confidences)
        
        return stats
    
    def suggest_custom_categories(self, analysis_results: List[Dict], min_files: int = 3) -> List[str]:
        """
        Suggest custom categories based on file analysis.
        
        Args:
            analysis_results: List of analysis results
            min_files: Minimum number of files to suggest a new category
            
        Returns:
            List of suggested category names
        """
        # This is a simplified implementation
        # In a more advanced version, this could use clustering or pattern recognition
        
        file_extensions = {}
        for result in analysis_results:
            if "error" in result:
                continue
            
            file_path = result.get("file_path", "")
            if file_path:
                ext = Path(file_path).suffix.lower()
                if ext and ext not in [".tmp", ".temp"]:
                    file_extensions[ext] = file_extensions.get(ext, 0) + 1
        
        suggestions = []
        for ext, count in file_extensions.items():
            if count >= min_files:
                # Check if this extension is already covered by existing categories
                covered = False
                for category_info in self.categories.values():
                    if ext in category_info["extensions"]:
                        covered = True
                        break
                
                if not covered:
                    suggestions.append(f"Files_{ext[1:].upper()}")
        
        return suggestions
    
    def analyze_file_context(self, file_path: str) -> Dict:
        """
        Analyze the contextual information of a file to improve categorization.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing contextual analysis
        """
        file_path_obj = Path(file_path)
        context = {
            "path_hints": [],
            "temporal_hints": [],
            "size_category": "",
            "likely_purpose": []
        }
        
        # Analyze path components for hints
        path_parts = [part.lower() for part in file_path_obj.parts]
        path_keywords = {
            "work": ["work", "office", "business", "projects", "client"],
            "personal": ["personal", "private", "documents", "my"],
            "education": ["school", "university", "college", "courses", "education"],
            "creative": ["photos", "pictures", "videos", "music", "creative", "art"],
            "downloads": ["downloads", "download", "temp", "temporary"],
            "archive": ["archive", "backup", "old", "storage"]
        }
        
        for category, keywords in path_keywords.items():
            if any(keyword in ' '.join(path_parts) for keyword in keywords):
                context["path_hints"].append(category)
        
        # Analyze file size for context
        try:
            size = file_path_obj.stat().st_size
            if size < 1024:  # < 1KB
                context["size_category"] = "tiny"
            elif size < 1024 * 1024:  # < 1MB
                context["size_category"] = "small"
            elif size < 10 * 1024 * 1024:  # < 10MB
                context["size_category"] = "medium"
            elif size < 100 * 1024 * 1024:  # < 100MB
                context["size_category"] = "large"
            else:
                context["size_category"] = "very_large"
        except:
            context["size_category"] = "unknown"
        
        # Analyze filename patterns for purpose hints
        filename = file_path_obj.name.lower()
        purpose_patterns = {
            "backup": ["backup", "bak", "copy", "old", "archive"],
            "temp": ["temp", "tmp", "cache", "temporary"],
            "config": ["config", "settings", "preferences", "options"],
            "log": ["log", "logs", "debug", "error", "trace"],
            "draft": ["draft", "wip", "work_in_progress", "unfinished"],
            "final": ["final", "completed", "finished", "done", "v1", "release"]
        }
        
        for purpose, patterns in purpose_patterns.items():
            if any(pattern in filename for pattern in patterns):
                context["likely_purpose"].append(purpose)
        
        return context
    
    def _refine_with_context(self, category: str, context: Dict, file_info: Dict) -> str:
        """
        Refine the category decision based on contextual information.
        
        Args:
            category: Initially predicted category
            context: Contextual analysis results
            file_info: File information
            
        Returns:
            Refined category
        """
        # If path hints strongly suggest a different category, consider switching
        path_hints = context.get("path_hints", [])
        
        # Strong path indicators that might override AI decision
        if "work" in path_hints and category in ["Reference_Documentation", "Personal_Documents"]:
            return "Work_Projects"
        
        if "education" in path_hints and category in ["Reference_Documentation", "Personal_Documents"]:
            return "Education_Learning"
        
        if "creative" in path_hints and category in ["Reference_Documentation", "Entertainment_Media"]:
            return "Creative_Projects"
        
        # Archive/backup indicators
        if "archive" in path_hints or "backup" in context.get("likely_purpose", []):
            if category not in ["Development_Code", "Finance_Money", "Health_Medical"]:  # Don't override critical categories
                return "Archive_Backup"
        
        return category
    
    def _generate_reasoning(self, file_info: Dict, category: str, context: Dict, subdirectory: str = None) -> str:
        """
        Generate human-readable reasoning for the categorization decision.
        
        Args:
            file_info: File information
            category: Chosen category
            context: Contextual analysis
            subdirectory: Chosen subdirectory
            
        Returns:
            Reasoning string
        """
        reasons = []
        
        extension = file_info.get("suffix", "").lower()
        filename = file_info.get("name", "").lower()
        
        # Extension-based reasoning
        if extension in self.categories[category]["extensions"]:
            reasons.append(f"file extension '{extension}' matches {category}")
        
        # Keyword-based reasoning
        matching_keywords = []
        for keyword in self.categories[category]["keywords"]:
            if keyword in filename:
                matching_keywords.append(keyword)
        
        if matching_keywords:
            reasons.append(f"filename contains keywords: {', '.join(matching_keywords)}")
        
        # Content-based reasoning
        if file_info.get("is_text", False) and file_info.get("text_content"):
            content_keywords = []
            content = file_info["text_content"].lower()
            for keyword in self.categories[category]["keywords"][:5]:  # Check first 5 keywords
                if keyword in content:
                    content_keywords.append(keyword)
            
            if content_keywords:
                reasons.append(f"content contains: {', '.join(content_keywords)}")
        
        # Subdirectory reasoning
        if subdirectory and subdirectory != "General":
            subdir_keywords = self.categories[category]["subdirectories"].get(subdirectory, [])
            matching_subdir_keywords = []
            for keyword in subdir_keywords:
                if keyword in filename or (file_info.get("text_content", "") and keyword in file_info["text_content"].lower()):
                    matching_subdir_keywords.append(keyword)
            
            if matching_subdir_keywords:
                reasons.append(f"placed in {subdirectory} based on: {', '.join(matching_subdir_keywords)}")
        
        # Context-based reasoning
        if context.get("path_hints"):
            reasons.append(f"path suggests: {', '.join(context['path_hints'])}")
        
        if not reasons:
            reasons.append("categorized based on general file characteristics")
        
        return "; ".join(reasons)
    
    def determine_subdirectory(self, file_info: Dict, category: str) -> str:
        """
        Determine the specific subdirectory within a category for a file.
        
        Args:
            file_info: File information dictionary
            category: Main category
            
        Returns:
            Subdirectory name
        """
        if category not in self.categories:
            return "General"
        
        subdirs = self.categories[category].get("subdirectories", {})
        if not subdirs:
            return "General"
        
        filename = file_info.get("name", "").lower()
        content = file_info.get("text_content", "").lower() if file_info.get("is_text", False) else ""
        extension = file_info.get("suffix", "").lower()
        
        # Score each subdirectory based on keyword matches
        subdir_scores = {}
        
        for subdir_name, keywords in subdirs.items():
            if subdir_name == "General":
                continue
                
            score = 0
            
            # Check filename matches
            for keyword in keywords:
                if keyword in filename:
                    score += 2  # Higher weight for filename matches
                
                # Check content matches
                if content and keyword in content:
                    score += 1
            
            # Special scoring for extensions
            if extension:
                for keyword in keywords:
                    if keyword == extension or keyword in extension:
                        score += 3  # Very high weight for extension matches
            
            if score > 0:
                subdir_scores[subdir_name] = score
        
        # Return the subdirectory with the highest score
        if subdir_scores:
            best_subdir = max(subdir_scores.items(), key=lambda x: x[1])[0]
            return best_subdir
        
        return "General"