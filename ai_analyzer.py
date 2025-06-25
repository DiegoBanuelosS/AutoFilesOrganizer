#!/usr/bin/env python3

import os
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer
    from nltk.chunk import ne_chunk
    from nltk import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    import pytesseract
    import cv2
    import numpy as np
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TextAnalysis:
    language: str
    readability_score: float
    word_count: int
    sentence_count: int
    keywords: List[str]
    entities: List[str]
    topics: List[str]
    category_suggestions: List[str]

class AIAnalyzer:
    def __init__(self):
        self.stemmer = None
        self.stop_words = set()
        self.vectorizer = None
        self._initialize_nltk()
        self._initialize_categories()
    
    def _initialize_nltk(self):
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('maxent_ne_chunker', quiet=True)
                nltk.download('words', quiet=True)
                
                self.stemmer = PorterStemmer()
                self.stop_words = set(stopwords.words('english'))
                self.stop_words.update(set(stopwords.words('spanish')))
                
                logger.info("NLTK initialized correctly")
            except Exception as e:
                logger.warning(f"Error initializing NLTK: {e}")
    
    def _initialize_categories(self):
        self.category_keywords = {
            'contracts': [
                'contract', 'agreement', 'terms', 'conditions', 'legal', 'signature',
                'parties', 'obligations', 'clause', 'covenant', 'breach'
            ],
            'invoices': [
                'invoice', 'bill', 'payment', 'amount', 'total', 'tax', 'cost',
                'price', 'client', 'customer', 'due', 'balance'
            ],
            'reports': [
                'report', 'analysis', 'findings', 'results', 'conclusion',
                'summary', 'data', 'statistics', 'metrics', 'evaluation'
            ],
            'manuals': [
                'manual', 'guide', 'instructions', 'tutorial', 'steps',
                'procedure', 'help', 'documentation', 'handbook'
            ],
            'certificates': [
                'certificate', 'diploma', 'award', 'qualification', 'competence',
                'skill', 'achievement', 'certification', 'accreditation'
            ],
            'academic': [
                'thesis', 'paper', 'research', 'study', 'university',
                'academic', 'scholar', 'dissertation', 'journal'
            ],
            'financial': [
                'balance', 'budget', 'financial', 'accounting', 'revenue',
                'expenses', 'profit', 'loss', 'statement', 'audit'
            ],
            'personal': [
                'personal', 'private', 'diary', 'notes', 'memories',
                'family', 'letter', 'correspondence'
            ]
        }
    
    def analyze_text(self, text: str, file_path: Optional[Path] = None) -> TextAnalysis:
        if not text or len(text.strip()) < 10:
            return self._create_default_analysis()
        
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]+', text))
        language = self._detect_language(text)
        readability_score = self._calculate_readability(text)
        keywords = self._extract_keywords(text)
        entities = self._extract_entities(text)
        topics = self._identify_topics(text)
        category_suggestions = self._suggest_categories(text, file_path)
        
        return TextAnalysis(
            language=language,
            readability_score=readability_score,
            word_count=word_count,
            sentence_count=sentence_count,
            keywords=keywords,
            entities=entities,
            topics=topics,
            category_suggestions=category_suggestions
        )
    
    def _create_default_analysis(self) -> TextAnalysis:
        return TextAnalysis(
            language="unknown",
            readability_score=0.0,
            word_count=0,
            sentence_count=0,
            keywords=[],
            entities=[],
            topics=[],
            category_suggestions=["others"]
        )
    
    def _detect_language(self, text: str) -> str:
        spanish_words = {'de', 'la', 'que', 'el', 'en', 'y', 'a', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al'}
        english_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
        
        words = set(text.lower().split())
        
        spanish_count = len(words.intersection(spanish_words))
        english_count = len(words.intersection(english_words))
        
        if spanish_count > english_count:
            return "spanish"
        elif english_count > spanish_count:
            return "english"
        else:
            return "unknown"
    
    def _calculate_readability(self, text: str) -> float:
        if TEXTSTAT_AVAILABLE:
            try:
                return textstat.flesch_reading_ease(text)
            except:
                pass
        
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        syllables = sum([self._count_syllables(word) for word in text.split()])
        
        if sentences == 0 or words == 0:
            return 0.0
        
        score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        vowels = 'aeiouáéíóúü'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        return max(1, syllable_count)
    
    def _extract_keywords(self, text: str) -> List[str]:
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        words = text_clean.split()
        
        if self.stop_words:
            words = [word for word in words if word not in self.stop_words]
        
        word_freq = {}
        for word in words:
            if len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]
    
    def _extract_entities(self, text: str) -> List[str]:
        entities = []
        
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'currency': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            'percentage': r'\d+(?:\.\d+)?%'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append(f"{entity_type}: {match}")
        
        return entities[:20]
    
    def _identify_topics(self, text: str) -> List[str]:
        topics = []
        
        tech_keywords = ['software', 'hardware', 'computer', 'internet', 'database', 'programming', 'technology']
        if any(keyword in text.lower() for keyword in tech_keywords):
            topics.append('technology')
        
        finance_keywords = ['money', 'bank', 'investment', 'financial', 'budget', 'cost', 'price']
        if any(keyword in text.lower() for keyword in finance_keywords):
            topics.append('finance')
        
        legal_keywords = ['law', 'legal', 'contract', 'agreement', 'rights', 'obligations']
        if any(keyword in text.lower() for keyword in legal_keywords):
            topics.append('legal')
        
        medical_keywords = ['health', 'medical', 'doctor', 'patient', 'treatment', 'medicine']
        if any(keyword in text.lower() for keyword in medical_keywords):
            topics.append('medicine')
        
        education_keywords = ['education', 'school', 'university', 'student', 'teacher']
        if any(keyword in text.lower() for keyword in education_keywords):
            topics.append('education')
        
        return topics
    
    def _suggest_categories(self, text: str, file_path: Optional[Path] = None) -> List[str]:
        suggestions = []
        text_lower = text.lower()
        
        if file_path:
            filename_lower = file_path.name.lower()
            text_lower += " " + filename_lower
        
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += text_lower.count(keyword)
            
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
            suggestions = [category for category, score in sorted_categories[:3]]
        
        if not suggestions:
            suggestions = ['general_documents']
        
        return suggestions

def check_dependencies():
    status = {
        "NLTK": NLTK_AVAILABLE,
        "TextStat": TEXTSTAT_AVAILABLE,
        "FuzzyWuzzy": FUZZYWUZZY_AVAILABLE,
        "OCR (Tesseract)": OCR_AVAILABLE,
        "Scikit-learn": SKLEARN_AVAILABLE
    }
    
    print("🔍  AI dependencies status:")
    for tool, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {tool}")
    
    return status

if __name__ == "__main__":
    analyzer = AIAnalyzer()
    check_dependencies()
    
    test_text = """
    This is a professional services contract between the mentioned parties.
    The agreement establishes the obligations and rights of each party.
    The terms and conditions are clearly defined in the following clauses.
    """
    
    result = analyzer.analyze_text(test_text)
    print(f"\n📄 Text analysis test:")
    print(f"  Language: {result.language}")
    print(f"  Words: {result.word_count}")
    print(f"  Readability: {result.readability_score:.1f}")
    print(f"  Keywords: {result.keywords}")
    print(f"  Suggested categories: {result.category_suggestions}")
