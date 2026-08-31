"""
Citation & Reference Awareness System
Comprehensive system for managing citations, references, and academic integrity
Integrated with sustainability, skill wallet, plagiarism, and AI detection systems
"""

import re
import json
import datetime
import hashlib
import uuid
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import statistics
from difflib import SequenceMatcher
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (uncomment if needed)
# nltk.download('punkt')
# nltk.download('stopwords')


class CitationStyle(Enum):
    """Common citation styles"""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    IEEE = "ieee"
    AMA = "ama"
    VANCOUVER = "vancouver"
    CSE = "cse"
    ACS = "acs"
    OTHER = "other"


class CitationType(Enum):
    """Types of citations"""
    BOOK = "book"
    JOURNAL_ARTICLE = "journal_article"
    WEBSITE = "website"
    CONFERENCE_PAPER = "conference_paper"
    THESIS = "thesis"
    REPORT = "report"
    NEWSPAPER = "newspaper"
    PATENT = "patent"
    VIDEO = "video"
    PODCAST = "podcast"
    SOCIAL_MEDIA = "social_media"
    OTHER = "other"


class ReferenceQuality(Enum):
    """Quality levels of references"""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 70-90%
    AVERAGE = "average"  # 50-70%
    POOR = "poor"  # 30-50%
    INADEQUATE = "inadequate"  # 0-30%


@dataclass
class Citation:
    """Represents a single citation"""
    citation_id: str
    text: str
    citation_style: CitationStyle
    citation_type: CitationType
    authors: List[str] = field(default_factory=list)
    title: str = ""
    year: Optional[int] = None
    journal: str = ""
    volume: str = ""
    pages: str = ""
    doi: str = ""
    url: str = ""
    publisher: str = ""
    location: str = ""
    in_text_citation: str = ""
    reference_count: int = 0
    quality_score: float = 0.0  # 0-100
    verified: bool = False
    verified_date: Optional[datetime.datetime] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Reference:
    """Represents a complete reference"""
    reference_id: str
    citations: List[Citation] = field(default_factory=list)
    total_citations: int = 0
    unique_sources: int = 0
    quality_score: float = 0.0
    quality_level: ReferenceQuality = ReferenceQuality.AVERAGE
    missing_information: List[str] = field(default_factory=list)
    format_errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class CitationAnalysis:
    """Analysis of citations in a document"""
    document_id: str
    total_citations: int
    unique_references: int
    citation_style: CitationStyle
    citation_density: float  # Citations per 1000 words
    reference_quality: ReferenceQuality
    quality_score: float
    style_consistency: float  # 0-1
    proper_citations: int
    improper_citations: int
    missing_citations: int
    common_errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


class CitationExtractor:
    """Extract and parse citations from text"""
    
    def __init__(self):
        self.citation_patterns = self._initialize_citation_patterns()
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.doi_pattern = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)
        
    def _initialize_citation_patterns(self) -> Dict[CitationStyle, List[str]]:
        """Initialize regex patterns for different citation styles"""
        return {
            CitationStyle.APA: [
                r'\(([A-Z][a-z]+,\s\d{4})\)',  # (Smith, 2020)
                r'\(([A-Z][a-z]+\s&\s[A-Z][a-z]+,\s\d{4})\)',  # (Smith & Jones, 2020)
                r'([A-Z][a-z]+\set\sal\.\,\s\d{4})',  # Smith et al., 2020
                r'([A-Z][a-z]+,\s\d{4})\s[a-z]'  # Smith, 2020 p.
            ],
            CitationStyle.MLA: [
                r'\(([A-Z][a-z]+)\s\d+\)',  # (Smith 123)
                r'\(([A-Z][a-z]+)\sand\s([A-Z][a-z]+)\s\d+\)',  # (Smith and Jones 123)
                r'([A-Z][a-z]+)\s\d+\s[a-z]'  # Smith 123 p.
            ],
            CitationStyle.CHICAGO: [
                r'\d+\.\s([A-Z][a-z]+),\s\"[^\"]+\"',  # 1. Smith, "Title"
                r'([A-Z][a-z]+)\s\d{4},\s\d+'  # Smith 2020, 123
            ],
            CitationStyle.IEEE: [
                r'\[\d+\]',  # [1]
                r'\[\d+-\d+\]',  # [1-5]
                r'(\w+\set\sal\.\s\[\d+\])'  # Smith et al. [1]
            ]
        }
    
    def extract_citations(self, text: str) -> List[Dict]:
        """Extract citations from text"""
        citations = []
        
        # Extract in-text citations
        for style, patterns in self.citation_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    citations.append({
                        'text': match if isinstance(match, str) else str(match),
                        'style': style,
                        'type': self._determine_citation_type(match),
                        'position': text.find(match)
                    })
        
        # Extract URLs and DOIs
        urls = self.url_pattern.findall(text)
        dois = self.doi_pattern.findall(text)
        
        for url in urls:
            citations.append({
                'text': url,
                'style': CitationStyle.OTHER,
                'type': CitationType.WEBSITE,
                'position': text.find(url)
            })
        
        for doi in dois:
            citations.append({
                'text': doi,
                'style': CitationStyle.OTHER,
                'type': CitationType.JOURNAL_ARTICLE,
                'position': text.find(doi)
            })
        
        # Sort by position and remove duplicates
        citations.sort(key=lambda x: x['position'])
        unique_citations = []
        seen = set()
        for cit in citations:
            if cit['text'] not in seen:
                seen.add(cit['text'])
                unique_citations.append(cit)
        
        return unique_citations
    
    def _determine_citation_type(self, citation_text: str) -> CitationType:
        """Determine the type of citation based on patterns"""
        citation_text = citation_text.lower()
        
        if 'doi' in citation_text or '10.' in citation_text:
            return CitationType.JOURNAL_ARTICLE
        
        if 'www.' in citation_text or 'http' in citation_text:
            return CitationType.WEBSITE
        
        if 'book' in citation_text or 'press' in citation_text or 'publisher' in citation_text:
            return CitationType.BOOK
        
        if 'conf' in citation_text or 'proceedings' in citation_text:
            return CitationType.CONFERENCE_PAPER
        
        if 'thesis' in citation_text or 'dissertation' in citation_text:
            return CitationType.THESIS
        
        if 'report' in citation_text:
            return CitationType.REPORT
        
        if 'patent' in citation_text:
            return CitationType.PATENT
        
        return CitationType.OTHER


class CitationValidator:
    """Validate citations for completeness and correctness"""
    
    def __init__(self):
        self.required_fields = {
            CitationType.BOOK: ['authors', 'title', 'year', 'publisher'],
            CitationType.JOURNAL_ARTICLE: ['authors', 'title', 'year', 'journal', 'volume', 'pages'],
            CitationType.WEBSITE: ['title', 'url', 'year'],
            CitationType.CONFERENCE_PAPER: ['authors', 'title', 'year', 'conference'],
            CitationType.THESIS: ['authors', 'title', 'year', 'institution'],
            CitationType.REPORT: ['authors', 'title', 'year', 'publisher'],
            CitationType.PATENT: ['authors', 'title', 'year', 'patent_number']
        }
        
        self.common_errors = {
            'missing_author': 'No author information provided',
            'missing_year': 'Publication year missing',
            'missing_title': 'Title missing',
            'missing_publisher': 'Publisher information missing',
            'missing_url': 'URL missing for web citation',
            'missing_doi': 'DOI missing for journal article',
            'invalid_year': 'Invalid or missing year',
            'incomplete_pages': 'Page numbers incomplete',
            'inconsistent_format': 'Citation format inconsistent with style'
        }
    
    def validate_citation(self, citation: Dict) -> Dict:
        """Validate a single citation"""
        errors = []
        warnings = []
        suggestions = []
        completeness_score = 100.0
        
        # Check for required fields
        citation_type = citation.get('type', CitationType.OTHER)
        required = self.required_fields.get(citation_type, [])
        
        for field in required:
            if not citation.get(field):
                errors.append(self.common_errors.get(f'missing_{field}', f'Missing {field}'))
                completeness_score -= 20
        
        # Validate specific fields
        if citation.get('year'):
            try:
                year = int(citation['year'])
                if year < 1900 or year > datetime.datetime.now().year + 1:
                    errors.append(self.common_errors['invalid_year'])
                    completeness_score -= 15
            except ValueError:
                errors.append(self.common_errors['invalid_year'])
                completeness_score -= 15
        
        if citation.get('url') and not citation['url'].startswith(('http://', 'https://')):
            warnings.append('URL should start with http:// or https://')
        
        if citation.get('doi') and not re.match(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', citation['doi'], re.IGNORECASE):
            warnings.append('DOI format appears incorrect')
        
        # Check for authors
        if citation.get('authors'):
            if isinstance(citation['authors'], list):
                if len(citation['authors']) == 0:
                    errors.append(self.common_errors['missing_author'])
                    completeness_score -= 15
            elif isinstance(citation['authors'], str):
                if len(citation['authors'].strip()) == 0:
                    errors.append(self.common_errors['missing_author'])
                    completeness_score -= 15
        
        # Generate suggestions
        if citation_type == CitationType.JOURNAL_ARTICLE and not citation.get('doi'):
            suggestions.append('Consider adding DOI for journal article')
        
        if citation_type == CitationType.WEBSITE and not citation.get('year'):
            suggestions.append('Add access date for web citation')
        
        if citation_type == CitationType.BOOK and not citation.get('publisher'):
            suggestions.append('Add publisher information for book citation')
        
        return {
            'completeness_score': max(0, completeness_score),
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions,
            'is_valid': len(errors) == 0
        }


class ReferenceQualityAnalyzer:
    """Analyze reference quality and provide improvement suggestions"""
    
    def __init__(self):
        self.quality_criteria = self._initialize_quality_criteria()
        self.citation_extractor = CitationExtractor()
        self.citation_validator = CitationValidator()
    
    def _initialize_quality_criteria(self) -> Dict:
        """Initialize quality criteria for references"""
        return {
            'completeness': {'weight': 0.30, 'description': 'Completeness of citation information'},
            'accuracy': {'weight': 0.25, 'description': 'Accuracy of citation details'},
            'consistency': {'weight': 0.20, 'description': 'Consistency with citation style'},
            'relevance': {'weight': 0.15, 'description': 'Relevance to the text'},
            'diversity': {'weight': 0.10, 'description': 'Diversity of sources'}
        }
    
    def analyze_references(self, text: str, citations: List[Dict]) -> Reference:
        """Analyze the quality of references"""
        reference = Reference(
            reference_id=str(uuid.uuid4())
        )
        
        # Process citations
        citation_objects = []
        for cit_data in citations:
            # Validate citation
            validation = self.citation_validator.validate_citation(cit_data)
            
            # Create Citation object
            citation = Citation(
                citation_id=str(uuid.uuid4()),
                text=cit_data.get('text', ''),
                citation_style=cit_data.get('style', CitationStyle.OTHER),
                citation_type=cit_data.get('type', CitationType.OTHER),
                authors=cit_data.get('authors', []),
                title=cit_data.get('title', ''),
                year=cit_data.get('year'),
                journal=cit_data.get('journal', ''),
                volume=cit_data.get('volume', ''),
                pages=cit_data.get('pages', ''),
                doi=cit_data.get('doi', ''),
                url=cit_data.get('url', ''),
                publisher=cit_data.get('publisher', ''),
                quality_score=validation['completeness_score'],
                verified=False
            )
            
            citation_objects.append(citation)
            
            # Collect errors and suggestions
            reference.format_errors.extend(validation['errors'])
            reference.suggestions.extend(validation['suggestions'])
            reference.missing_information.extend(validation['warnings'])
        
        reference.citations = citation_objects
        reference.total_citations = len(citation_objects)
        
        # Count unique sources
        unique_sources = set()
        for cit in citation_objects:
            if cit.title:
                unique_sources.add(cit.title)
            elif cit.text:
                unique_sources.add(cit.text)
        reference.unique_sources = len(unique_sources)
        
        # Calculate overall quality score
        if reference.total_citations > 0:
            avg_quality = sum(c.quality_score for c in citation_objects) / reference.total_citations
            reference.quality_score = avg_quality
            
            # Adjust for diversity
            diversity_factor = min(1.0, reference.unique_sources / reference.total_citations)
            reference.quality_score = avg_quality * (0.8 + 0.2 * diversity_factor)
        else:
            reference.quality_score = 0
        
        # Determine quality level
        reference.quality_level = self._get_quality_level(reference.quality_score)
        
        return reference
    
    def _get_quality_level(self, score: float) -> ReferenceQuality:
        """Determine quality level from score"""
        if score >= 90:
            return ReferenceQuality.EXCELLENT
        elif score >= 70:
            return ReferenceQuality.GOOD
        elif score >= 50:
            return ReferenceQuality.AVERAGE
        elif score >= 30:
            return ReferenceQuality.POOR
        else:
            return ReferenceQuality.INADEQUATE
    
    def calculate_citation_density(self, text: str, citation_count: int) -> float:
        """Calculate citation density (citations per 1000 words)"""
        words = len(text.split())
        if words == 0:
            return 0
        return (citation_count / words) * 1000
    
    def check_style_consistency(self, citations: List[Citation]) -> float:
        """Check consistency of citation styles"""
        if not citations:
            return 0.0
        
        style_counts = Counter(c.citation_style for c in citations)
        most_common = style_counts.most_common(1)[0][1] if style_counts else 0
        
        return most_common / len(citations)


class ReferenceGenerator:
    """Generate properly formatted citations and references"""
    
    def __init__(self):
        self.style_formats = self._initialize_style_formats()
    
    def _initialize_style_formats(self) -> Dict:
        """Initialize formatting templates for different styles"""
        return {
            CitationStyle.APA: {
                'book': '{authors} ({year}). {title}. {publisher}.',
                'journal': '{authors} ({year}). {title}. {journal}, {volume}({issue}), {pages}. https://doi.org/{doi}',
                'website': '{authors} ({year}). {title}. Retrieved from {url}',
                'conference': '{authors} ({year}). {title}. In {conference} (pp. {pages}).',
                'thesis': '{authors} ({year}). {title} [Thesis]. {institution}.'
            },
            CitationStyle.MLA: {
                'book': '{authors}. {title}. {publisher}, {year}.',
                'journal': '{authors}. "{title}." {journal} {volume}.{issue} ({year}): {pages}.',
                'website': '{authors}. "{title}." {website}, {year}, {url}. Accessed {access_date}.'
            },
            CitationStyle.CHICAGO: {
                'book': '{authors}. {title}. {location}: {publisher}, {year}.',
                'journal': '{authors}. "{title}." {journal} {volume}, no. {issue} ({year}): {pages}.',
                'website': '{authors}. "{title}." {website}. Accessed {access_date}. {url}.'
            },
            CitationStyle.IEEE: {
                'book': '[{id}] {authors}, {title}. {location}: {publisher}, {year}.',
                'journal': '[{id}] {authors}, "{title}," {journal}, vol. {volume}, no. {issue}, pp. {pages}, {year}.',
                'website': '[{id}] {authors}, "{title}," {website}. Available: {url}. [Accessed {access_date}].'
            }
        }
    
    def generate_citation(self, citation_data: Dict, style: CitationStyle = CitationStyle.APA) -> str:
        """Generate a formatted citation"""
        citation_type = citation_data.get('type', CitationType.OTHER)
        formats = self.style_formats.get(style, {})
        
        # Get appropriate format template
        if citation_type == CitationType.BOOK:
            template = formats.get('book', '{authors} ({year}). {title}. {publisher}.')
        elif citation_type == CitationType.JOURNAL_ARTICLE:
            template = formats.get('journal', '{authors} ({year}). {title}. {journal}, {volume}, {pages}.')
        elif citation_type == CitationType.WEBSITE:
            template = formats.get('website', '{authors} ({year}). {title}. Retrieved from {url}')
        else:
            template = '{authors} ({year}). {title}.'
        
        # Fill in template
        try:
            # Handle authors formatting
            authors = citation_data.get('authors', [])
            if isinstance(authors, list):
                if len(authors) == 1:
                    authors_str = authors[0]
                elif len(authors) == 2:
                    authors_str = f"{authors[0]} and {authors[1]}"
                else:
                    authors_str = f"{authors[0]} et al."
            else:
                authors_str = str(authors) if authors else 'Unknown'
            
            # Create format dictionary
            format_dict = {
                'authors': authors_str,
                'year': citation_data.get('year', 'n.d.'),
                'title': citation_data.get('title', 'Untitled'),
                'publisher': citation_data.get('publisher', ''),
                'journal': citation_data.get('journal', ''),
                'volume': citation_data.get('volume', ''),
                'issue': citation_data.get('issue', ''),
                'pages': citation_data.get('pages', ''),
                'doi': citation_data.get('doi', ''),
                'url': citation_data.get('url', ''),
                'location': citation_data.get('location', ''),
                'conference': citation_data.get('conference', ''),
                'institution': citation_data.get('institution', ''),
                'website': citation_data.get('website', ''),
                'access_date': datetime.datetime.now().strftime('%Y-%m-%d'),
                'id': citation_data.get('id', '1')
            }
            
            # Generate citation
            citation = template.format(**format_dict)
            
            # Clean up extra spaces and punctuation
            citation = re.sub(r'\s+', ' ', citation)
            citation = re.sub(r'\.\s*\.', '.', citation)
            
            return citation
            
        except KeyError as e:
            return f"[Error generating citation: Missing {e}]"
    
    def generate_reference_list(self, citations: List[Dict], style: CitationStyle = CitationStyle.APA) -> str:
        """Generate a complete reference list"""
        reference_list = []
        
        for i, citation_data in enumerate(citations, 1):
            citation_data['id'] = str(i)
            formatted = self.generate_citation(citation_data, style)
            reference_list.append(formatted)
        
        return '\n\n'.join(reference_list)


class CitationAwarenessSystem:
    """Main system for citation and reference management with integrations"""
    
    def __init__(self, sustainability_system=None, skill_wallet_manager=None, 
                 plagiarism_system=None, ai_detection_system=None):
        self.sustainability_system = sustainability_system
        self.skill_wallet_manager = skill_wallet_manager
        self.plagiarism_system = plagiarism_system
        self.ai_detection_system = ai_detection_system
        
        self.citation_extractor = CitationExtractor()
        self.citation_validator = CitationValidator()
        self.quality_analyzer = ReferenceQualityAnalyzer()
        self.reference_generator = ReferenceGenerator()
        
        # Storage
        self.citation_analyses: Dict[str, CitationAnalysis] = {}
        self.user_citations: Dict[str, List[str]] = defaultdict(list)
        self.reference_library: Dict[str, List[Citation]] = {}
        
        # Statistics
        self.total_analyzed = 0
        self.total_citations = 0
        self.total_errors = 0
        
        # Initialize citation skills
        self._initialize_citation_skills()
    
    def _initialize_citation_skills(self):
        """Initialize skills related to citation and referencing"""
        if self.skill_wallet_manager:
            citation_skills = [
                {
                    'id': 'skill_cit_001',
                    'name': 'Academic Citation',
                    'description': 'Proper academic citation and referencing',
                    'category': SkillCategory.TECHNICAL,
                    'level': SkillLevel.BEGINNER
                },
                {
                    'id': 'skill_cit_002',
                    'name': 'Reference Management',
                    'description': 'Managing references and bibliographies',
                    'category': SkillCategory.TECHNICAL,
                    'level': SkillLevel.BEGINNER
                },
                {
                    'id': 'skill_cit_003',
                    'name': 'Citation Style Knowledge',
                    'description': 'Knowledge of different citation styles',
                    'category': SkillCategory.COMMUNICATION,
                    'level': SkillLevel.BEGINNER
                },
                {
                    'id': 'skill_cit_004',
                    'name': 'Academic Integrity',
                    'description': 'Understanding of academic integrity principles',
                    'category': SkillCategory.SOFT,
                    'level': SkillLevel.BEGINNER
                }
            ]
            
            for skill_data in citation_skills:
                skill = Skill(
                    skill_id=skill_data['id'],
                    name=skill_data['name'],
                    description=skill_data['description'],
                    category=skill_data['category'],
                    level=skill_data['level']
                )
                self.skill_wallet_manager.skill_definitions[skill.skill_id] = skill
    
    def analyze_document(self, text: str, user_id: str = None, 
                        document_id: str = None) -> CitationAnalysis:
        """Analyze citations in a document"""
        if not document_id:
            document_id = hashlib.md5(text.encode()).hexdigest()[:8]
        
        # Extract citations
        extracted_citations = self.citation_extractor.extract_citations(text)
        
        # Parse citations into structured data
        parsed_citations = []
        for cit in extracted_citations:
            citation_data = self._parse_citation(cit)
            parsed_citations.append(citation_data)
        
        # Analyze references
        reference = self.quality_analyzer.analyze_references(text, parsed_citations)
        
        # Calculate metrics
        total_citations = len(parsed_citations)
        unique_references = reference.unique_sources
        citation_density = self.quality_analyzer.calculate_citation_density(text, total_citations)
        
        # Determine citation style (most common)
        style_counts = Counter()
        for cit in parsed_citations:
            style_counts[cit.get('style', CitationStyle.OTHER)] += 1
        
        most_common_style = style_counts.most_common(1)
        primary_style = most_common_style[0][0] if most_common_style else CitationStyle.OTHER
        
        # Check style consistency
        citation_objects = []
        for cit_data in parsed_citations:
            try:
                citation = Citation(
                    citation_id=str(uuid.uuid4()),
                    text=cit_data.get('text', ''),
                    citation_style=cit_data.get('style', CitationStyle.OTHER),
                    citation_type=cit_data.get('type', CitationType.OTHER),
                    authors=cit_data.get('authors', []),
                    title=cit_data.get('title', ''),
                    year=cit_data.get('year'),
                    journal=cit_data.get('journal', ''),
                    volume=cit_data.get('volume', ''),
                    pages=cit_data.get('pages', ''),
                    doi=cit_data.get('doi', ''),
                    url=cit_data.get('url', ''),
                    publisher=cit_data.get('publisher', '')
                )
                citation_objects.append(citation)
            except:
                continue
        
        style_consistency = self.quality_analyzer.check_style_consistency(citation_objects)
        
        # Identify common errors
        common_errors = list(set(reference.format_errors))[:5]
        if not common_errors:
            # Check for common issues
            if total_citations == 0:
                common_errors.append("No citations found in document")
            elif total_citations < 5 and len(text.split()) > 500:
                common_errors.append("Low citation density - consider adding more references")
        
        # Generate suggestions
        suggestions = list(set(reference.suggestions))[:5]
        if not suggestions and total_citations > 0:
            suggestions.append("Consider adding DOIs to journal article citations")
            suggestions.append("Check for consistent formatting across all citations")
        
        # Create analysis
        analysis = CitationAnalysis(
            document_id=document_id,
            total_citations=total_citations,
            unique_references=unique_references,
            citation_style=primary_style,
            citation_density=citation_density,
            reference_quality=reference.quality_level,
            quality_score=reference.quality_score,
            style_consistency=style_consistency,
            proper_citations=sum(1 for c in citation_objects if c.quality_score >= 70),
            improper_citations=sum(1 for c in citation_objects if c.quality_score < 70),
            missing_citations=self._detect_missing_citations(text, parsed_citations),
            common_errors=common_errors,
            suggestions=suggestions,
            citations=citation_objects
        )
        
        # Store analysis
        self.citation_analyses[document_id] = analysis
        
        if user_id:
            self.user_citations[user_id].append(document_id)
            
            # Update sustainability system
            if self.sustainability_system:
                self._update_sustainability_for_citations(user_id, analysis)
            
            # Update skill wallet
            if self.skill_wallet_manager:
                self._update_skill_wallet_for_citations(user_id, analysis)
        
        # Update statistics
        self.total_analyzed += 1
        self.total_citations += total_citations
        self.total_errors += len(common_errors)
        
        return analysis
    
    def _parse_citation(self, citation_data: Dict) -> Dict:
        """Parse raw citation data into structured format"""
        text = citation_data.get('text', '')
        style = citation_data.get('style', CitationStyle.OTHER)
        cit_type = citation_data.get('type', CitationType.OTHER)
        
        parsed = {
            'text': text,
            'style': style,
            'type': cit_type
        }
        
        # Try to extract author names
        author_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)
        if author_match:
            parsed['authors'] = [author_match.group(1)]
        
        # Try to extract year
        year_match = re.search(r'(19|20)\d{2}', text)
        if year_match:
            parsed['year'] = int(year_match.group())
        
        # Try to extract title
        title_match = re.search(r'"([^"]+)"|' + r'([A-Z][a-z\s]+(?=\.|,|$))', text)
        if title_match:
            title = title_match.group(1) or title_match.group(2)
            if title and len(title) > 10:
                parsed['title'] = title.strip()
        
        # Extract journal name
        journal_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\d+', text)
        if journal_match:
            parsed['journal'] = journal_match.group(1)
        
        # Extract DOI
        doi_match = re.search(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', text, re.IGNORECASE)
        if doi_match:
            parsed['doi'] = doi_match.group()
        
        # Extract URL
        url_match = re.search(r'https?://[^\s]+', text, re.IGNORECASE)
        if url_match:
            parsed['url'] = url_match.group()
        
        return parsed
    
    def _detect_missing_citations(self, text: str, citations: List[Dict]) -> int:
        """Detect potential missing citations in text"""
        # This is a simplified detection - checks for statements that might need citations
        missing = 0
        
        # Patterns that might indicate claims needing citations
        claim_patterns = [
            r'according\s+to\s+research',
            r'studies\s+show',
            r'it\s+has\s+been\s+found',
            r'research\s+indicates',
            r'evidence\s+suggests',
            r'data\s+shows',
            r'statistics\s+show',
            r'experts\s+say',
            r'it\s+is\s+widely\s+accepted',
            r'it\s+is\s+known\s+that'
        ]
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Check if there's a citation nearby (within 50 characters)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Check for citation indicators
                has_citation = any(c.get('text', '') in context for c in citations)
                if not has_citation:
                    missing += 1
        
        return missing
    
    def _update_sustainability_for_citations(self, user_id: str, analysis: CitationAnalysis):
        """Update sustainability system based on citation quality"""
        user = self.sustainability_system.get_user(user_id)
        if not user:
            return
        
        points = 0
        
        # Award points for good citation practices
        if analysis.quality_score >= 80:
            points += 30
            print(f"  📚 Excellent citation practices! +30 points")
        elif analysis.quality_score >= 60:
            points += 15
            print(f"  📚 Good citation practices! +15 points")
        elif analysis.quality_score >= 40:
            points += 5
            print(f"  📚 Adequate citations +5 points")
        
        # Bonus for proper citations
        if analysis.proper_citations > 0 and analysis.improper_citations == 0:
            points += 10
            print(f"  ✅ All citations are properly formatted! +10 bonus points")
        
        # Penalty for missing citations
        if analysis.missing_citations > 3:
            penalty = min(20, analysis.missing_citations * 5)
            points -= penalty
            print(f"  ⚠️ {analysis.missing_citations} potential missing citations detected. -{penalty} points")
        
        # Update user points
        user.total_points = max(0, user.total_points + points)
        user.add_xp(max(0, points // 2))
    
    def _update_skill_wallet_for_citations(self, user_id: str, analysis: CitationAnalysis):
        """Update skill wallet based on citation performance"""
        wallet = self.skill_wallet_manager.get_skill_wallet(user_id)
        if not wallet:
            return
        
        # Award Academic Citation skill
        if analysis.total_citations > 0:
            skill_id = 'skill_cit_001'
            if skill_id in wallet.skills:
                skill = wallet.skills[skill_id]
                skill.add_experience(10 * analysis.proper_citations)
                if skill.experience_points > 100:
                    skill.level = SkillLevel.INTERMEDIATE
                    print(f"  📈 Upgraded Academic Citation skill to INTERMEDIATE!")
            else:
                self.skill_wallet_manager.award_skill(user_id, skill_id, SkillLevel.BEGINNER)
                print(f"  🎯 Awarded Academic Citation skill!")
        
        # Award Reference Management skill for high quality
        if analysis.quality_score >= 70:
            skill_id = 'skill_cit_002'
            if skill_id not in wallet.skills:
                self.skill_wallet_manager.award_skill(user_id, skill_id, SkillLevel.BEGINNER)
                print(f"  📊 Awarded Reference Management skill!")
        
        # Award Citation Style Knowledge for consistency
        if analysis.style_consistency > 0.8:
            skill_id = 'skill_cit_003'
            if skill_id not in wallet.skills:
                self.skill_wallet_manager.award_skill(user_id, skill_id, SkillLevel.BEGINNER)
                print(f"  📝 Awarded Citation Style Knowledge skill!")
        
        # Award Academic Integrity for proper citations
        if analysis.proper_citations > 0 and analysis.missing_citations == 0:
            skill_id = 'skill_cit_004'
            if skill_id not in wallet.skills:
                self.skill_wallet_manager.award_skill(user_id, skill_id, SkillLevel.BEGINNER)
                print(f"  🎓 Awarded Academic Integrity skill!")
    
    def generate_citation_advice(self, user_id: str) -> Dict:
        """Generate personalized citation advice for a user"""
        user_docs = self.user_citations.get(user_id, [])
        
       
