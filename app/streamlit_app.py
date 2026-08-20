"""
Semantic Plagiarism Detector - Main Streamlit Application Entry Point.

Lightweight coordinator responsible for page setup, routing, state management initialization,
and delegating view rendering to modular components.
"""

import asyncio
import hashlib
import io as _io
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from src.core.hybrid_scorer import HybridScorer, HybridConfig
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import streamlit as st
from src.errors import UI_SESSION_EXPIRED, EmptyDocumentError

# 1. Fix Streamlit import paths FIRST so 'app' can be found
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent  # Points to semantic-plagiarism-detector/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Silence harmless Windows asyncio Proactor connection lost bugs
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv

load_dotenv()

# After existing imports, add:
from app.components.advanced_analytics import (
    AdvancedTextPreprocessor,
    ComparisonHistoryManager,
    ContextPreservingChunker,
    OptimizedBatchProcessor,
    PerformanceMonitor,
    ProcessingStatus,
    initialize_advanced_features,
    render_advanced_features_sidebar,
    render_document_analysis_widget,
    render_processing_status_widget,
    run_pipeline_with_tracking,
    track_comparison,
)


from src.core.lexical_similarity import (
    jaccard_similarity,
    dice_coefficient,
    overlap_coefficient,
    lexical_similarity_matrix,
    calculate_lexical_similarity,
    n_gram_overlap,
    scale_lexical_score,
    compute_char_ngram_similarity,
    render_report_generator_ui
)
def integrate_pattern_recognition():
    """Initialize and integrate pattern recognition"""
    if 'pattern_engine' not in st.session_state:
        st.session_state['pattern_engine'] = PatternRecognitionEngine()
    
    # Add pattern recognition tab to main app
    render_pattern_recognition_ui(st.session_state['pattern_engine'])
def integrate_text_analysis():
    """Initialize and integrate text analysis engine"""
    if 'text_analyzer' not in st.session_state:
        st.session_state['text_analyzer'] = TextAnalysisEngine()
    
    # Add text analysis tab to main app
    render_text_analysis_ui(st.session_state['text_analyzer'])
def integrate_preprocessing():
    """Initialize and integrate preprocessing engine"""
    if 'preprocessor' not in st.session_state:
        st.session_state['preprocessor'] = DocumentPreprocessor()
    
    # Add preprocessing tab to main app
    render_preprocessing_ui(st.session_state['preprocessor'])

from app.components.cross_lingual_ui import (
    render_cross_lingual_settings,
    render_cross_lingual_ui_in_drilldown,
    get_cross_lingual_metadata,
    is_cross_lingual_enabled,
    render_cross_lingual_stats,
)



# ── Document Version Control Imports ─────────────────────────────────────
from app.components.document_version_control import (
    render_version_control_ui,
    initialize_version_control,
    VersionManager,
    ChangeTracker,
    DocumentVersion,
    VersionDiff,
)
# ── Smart Notifications Imports ──────────────────────────────────────────
from app.components.smart_notifications import (
    render_notification_center,
    render_notification_stats,
    render_notification_badge,
    initialize_notifications,
    NotificationManager,
    Notification,
    AlertRule,
    NotificationPriority,
    NotificationChannel,
    NotificationStatus,
    SmartFilter,
    UserNotificationPreferences,
)
# ── Export Integration Imports ───────────────────────────────────────────
from app.components.export_integration import (
    render_export_center,
    render_export_bulk,
    initialize_export_system,
    ExportManager,
    ExportFormat,
    ExportJob,
    ExportTemplate,
    CloudConfig,
    CloudProvider,
    ExportStatus,
)
# ── Advanced Analytics Engine Imports ────────────────────────────────────
from app.components.advanced_analytics_engine import (
    render_analytics_engine,
    initialize_analytics_engine,
    PredictiveEngine,
    AnomalyDetector,
    InsightGenerator,
    RiskScorer,
    PatternRecognizer,
    PredictiveInsight,
    TrendForecast,
    RiskAssessment,
    AnomalyDetectionResult,
    RiskLevel,
    InsightType,
    ForecastPeriod,
)
# ── Audit Compliance Engine Imports ──────────────────────────────────────
from app.components.audit_compliance_engine import (
    render_audit_compliance_engine,
    initialize_audit_compliance_engine,
    AuditTrailManager,
    PolicyEnforcer,
    ComplianceReportGenerator,
    CertificateManager,
    AuditEvent,
    PolicyViolation,
    ComplianceReport,
    ComplianceCertificate,
    ComplianceStatus,
    AuditSeverity,
    PolicyType,
    Regulation,
)
# ── Workflow Automation Imports ──────────────────────────────────────────
from app.components.workflow_automation import (
    render_workflow_automation,
    initialize_workflow_automation,
    WorkflowEngine,
    Workflow,
    WorkflowExecution,
    Task,
    ApprovalRequest,
    WorkflowTemplate,
    WorkflowStatus,
    TaskStatus,
    TriggerType,
    WorkflowCategory,
)
# ── API Gateway Imports ──────────────────────────────────────────────────
from app.components.api_gateway import (
    render_api_gateway,
    initialize_api_gateway,
    ApiGateway,
    ApiEndpoint,
    ApiKey,
    Webhook,
    ServiceConnection,
    ApiLog,
    ApiMethod,
    ApiStatus,
    WebhookStatus,
    ServiceType,
)
# Heavy clustering imports removed (Issue #2811) - offloaded to FastAPI background task

# ── Document Version Control Imports ─────────────────────────────────────
from app.components.document_version_control import (
    render_version_control_ui,
    initialize_version_control,
    VersionManager,
    ChangeTracker,
    DocumentVersion,
    VersionDiffGenerator,
    VersionStorageManager,
    PlagiarismEvolutionAnalyzer,
    SmartChangePatternDetector,
    render_plagiarism_evolution_ui,
    render_smart_detection_ui,
    render_global_version_dashboard,
    render_version_control_dashboard,
    integrate_version_control_with_analysis,
    migrate_existing_documents_to_version_control,
)
# ── Collaboration System Imports ──────────────────────────────────────────
from app.components.collaboration_system import (
    DocumentAnnotation,
    ReviewWorkflow,
    UserSession,
    AnnotationManager,
    WorkflowManager,
    ActivityManager,
    ReviewSystem,
    render_annotation_ui,
    render_workflow_ui,
    render_review_dashboard,
    render_activity_ui,
    render_decision_history,
    render_collaboration_dashboard,
    initialize_review_system,
)
# ───────────────────────────────────────────────────────────────────────────────
# ── SECTION: INTELLIGENT DOCUMENT TAGGING & CATEGORIZATION (Issue #1988) ────
# ───────────────────────────────────────────────────────────────────────────────

import re
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class DocumentTag:
    """Represents a tag associated with a document"""
    id: str
    name: str
    category: str  # 'topic', 'type', 'status', 'custom'
    confidence: float
    created_at: datetime
    created_by: str
    is_auto_generated: bool = False
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat()
        }

@dataclass
class DocumentCategory:
    """Represents a document category"""
    id: str
    name: str
    description: str
    parent_id: Optional[str] = None
    color: str = '#808080'
    tags: List[str] = None
    created_at: datetime = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat()
        }

@dataclass
class TagAssignment:
    """Represents a tag assignment to a document"""
    id: str
    document_name: str
    tag_id: str
    assigned_at: datetime
    assigned_by: str
    is_auto: bool = False
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'assigned_at': self.assigned_at.isoformat()
        }

# ── Tag Generator ───────────────────────────────────────────────────────────

class IntelligentTagGenerator:
    """Generates intelligent tags from document content"""
    
    def __init__(self):
        self.common_words = {
            'plagiarism': ['academic', 'integrity', 'ethics', 'copying', 'similarity'],
            'research': ['study', 'analysis', 'methodology', 'literature', 'review'],
            'data': ['analysis', 'statistics', 'results', 'findings', 'visualization'],
            'algorithm': ['code', 'implementation', 'performance', 'optimization'],
            'machine learning': ['ai', 'neural', 'deep', 'training', 'model'],
            'software': ['development', 'programming', 'system', 'application'],
            'education': ['learning', 'teaching', 'curriculum', 'student', 'assessment'],
            'ethics': ['privacy', 'security', 'compliance', 'policy', 'regulation']
        }
        
        self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                         'for', 'of', 'with', 'without', 'by', 'from', 'up', 'down',
                         'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        
        self.tag_cache = {}
    
    def generate_tags(self, content: str, max_tags: int = 10) -> List[Tuple[str, float]]:
        """Generate tags from document content"""
        if not content:
            return []
        
        # Check cache
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self.tag_cache:
            return self.tag_cache[content_hash]
        
        # Extract keywords
        keywords = self._extract_keywords(content)
        
        # Score tags
        tags = self._score_tags(keywords, content)
        
        # Sort by confidence and limit
        tags = sorted(tags, key=lambda x: x[1], reverse=True)[:max_tags]
        
        # Cache results
        self.tag_cache[content_hash] = tags
        
        return tags
    
    def _extract_keywords(self, content: str) -> Dict[str, float]:
        """Extract keywords from content with TF-IDF scoring"""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Remove stopwords
        words = [w for w in words if w not in self.stopwords]
        
        # Count frequencies
        freq = Counter(words)
        total = len(words) or 1
        
        # Calculate scores (simple TF)
        scores = {word: count/total for word, count in freq.items()}
        
        # Boost common topics
        for topic, keywords in self.common_words.items():
            for keyword in keywords:
                if keyword in scores:
                    scores[keyword] *= 1.5
        
        return scores
    
    def _score_tags(self, keywords: Dict[str, float], content: str) -> List[Tuple[str, float]]:
        """Score potential tags"""
        tags = []
        
        # Direct keywords
        for word, score in keywords.items():
            if score > 0.01 and len(word) > 2:
                tags.append((word, min(score * 2, 1.0)))
        
        # Topic detection
        topic_scores = self._detect_topics(content)
        for topic, score in topic_scores:
            tags.append((topic, min(score, 1.0)))
        
        # Remove duplicates
        unique_tags = {}
        for tag, score in tags:
            if tag not in unique_tags or score > unique_tags[tag]:
                unique_tags[tag] = score
        
        return list(unique_tags.items())
    
    def _detect_topics(self, content: str) -> List[Tuple[str, float]]:
        """Detect topics in content using keyword matching"""
        content_lower = content.lower()
        topic_scores = []
        
        for topic, keywords in self.common_words.items():
            matches = 0
            total_keywords = len(keywords)
            
            for keyword in keywords:
                if keyword in content_lower:
                    matches += 1
            
            if matches > 0:
                score = matches / total_keywords
                topic_scores.append((topic, score))
        
        return topic_scores
    
    def generate_categories(self, content: str) -> Tuple[str, float]:
        """Generate category prediction for document"""
        if not content:
            return 'Uncategorized', 0.0
        
        topic_scores = self._detect_topics(content)
        
        if topic_scores:
            best_topic, best_score = max(topic_scores, key=lambda x: x[1])
            if best_score > 0.3:
                return best_topic.title(), best_score
        
        # Default categories
        categories = ['Academic', 'Research', 'Technical', 'Review', 'Report']
        for category in categories:
            if category.lower() in content.lower():
                return category, 0.6
        
        return 'Uncategorized', 0.3

# ── Tag Manager ─────────────────────────────────────────────────────────────

class TagManager:
    """Manages document tags and categories"""
    
    def __init__(self):
        self.tags: Dict[str, DocumentTag] = {}
        self.categories: Dict[str, DocumentCategory] = {}
        self.assignments: List[TagAssignment] = []
        self.tag_counter = Counter()
        self.tag_usage = defaultdict(int)
        
        # Initialize default categories
        self._init_default_categories()
    
    def _init_default_categories(self):
        """Initialize default categories"""
        defaults = [
            ('academic', 'Academic', 'Academic integrity and plagiarism related', '#4CAF50'),
            ('research', 'Research', 'Research methodologies and findings', '#2196F3'),
            ('technical', 'Technical', 'Technical documents and implementations', '#FF9800'),
            ('review', 'Review', 'Document reviews and analysis', '#9C27B0'),
            ('report', 'Report', 'Reports and summaries', '#F44336'),
        ]
        
        for id, name, desc, color in defaults:
            if id not in self.categories:
                self.categories[id] = DocumentCategory(
                    id=id,
                    name=name,
                    description=desc,
                    color=color
                )
    
    def add_tag(self, name: str, category: str = 'custom', 
                confidence: float = 1.0, auto_generated: bool = False,
                user_id: str = 'system') -> DocumentTag:
        """Add a new tag"""
        # Check if tag exists
        existing = self.get_tag_by_name(name)
        if existing:
            return existing
        
        tag_id = str(uuid.uuid4())
        tag = DocumentTag(
            id=tag_id,
            name=name,
            category=category,
            confidence=confidence,
            created_at=datetime.now(),
            created_by=user_id,
            is_auto_generated=auto_generated
        )
        self.tags[tag_id] = tag
        self.tag_counter[name] = 0
        return tag
    
    def get_tag(self, tag_id: str) -> Optional[DocumentTag]:
        """Get a tag by ID"""
        return self.tags.get(tag_id)
    
    def get_tag_by_name(self, name: str) -> Optional[DocumentTag]:
        """Get a tag by name"""
        for tag in self.tags.values():
            if tag.name == name:
                return tag
        return None
    
    def get_all_tags(self) -> List[DocumentTag]:
        """Get all tags"""
        return list(self.tags.values())
    
    def assign_tag(self, document_name: str, tag_name: str, 
                   user_id: str = 'system', auto: bool = False) -> Optional[str]:
        """Assign a tag to a document"""
        tag = self.get_tag_by_name(tag_name)
        if not tag:
            tag = self.add_tag(tag_name, auto_generated=auto, user_id=user_id)
        
        assignment = TagAssignment(
            id=str(uuid.uuid4()),
            document_name=document_name,
            tag_id=tag.id,
            assigned_at=datetime.now(),
            assigned_by=user_id,
            is_auto=auto
        )
        self.assignments.append(assignment)
        self.tag_counter[tag_name] += 1
        self.tag_usage[tag_name] += 1
        return assignment.id
    
    def unassign_tag(self, document_name: str, tag_name: str) -> bool:
        """Remove a tag assignment"""
        tag = self.get_tag_by_name(tag_name)
        if not tag:
            return False
        
        self.assignments = [
            a for a in self.assignments 
            if not (a.document_name == document_name and a.tag_id == tag.id)
        ]
        
        if self.tag_counter[tag_name] > 0:
            self.tag_counter[tag_name] -= 1
        return True
    
    def get_document_tags(self, document_name: str) -> List[DocumentTag]:
        """Get all tags for a document"""
        doc_assignments = [a for a in self.assignments if a.document_name == document_name]
        return [self.tags[a.tag_id] for a in doc_assignments if a.tag_id in self.tags]
    
    def get_documents_by_tag(self, tag_name: str) -> List[str]:
        """Get all documents with a specific tag"""
        tag = self.get_tag_by_name(tag_name)
        if not tag:
            return []
        return [a.document_name for a in self.assignments if a.tag_id == tag.id]
    
    def add_category(self, name: str, description: str, parent_id: Optional[str] = None,
                     color: str = '#808080') -> DocumentCategory:
        """Add a new category"""
        category_id = str(uuid.uuid4())
        category = DocumentCategory(
            id=category_id,
            name=name,
            description=description,
            parent_id=parent_id,
            color=color
        )
        self.categories[category_id] = category
        return category
    
    def get_category(self, category_id: str) -> Optional[DocumentCategory]:
        """Get a category by ID"""
        return self.categories.get(category_id)
    
    def get_all_categories(self) -> List[DocumentCategory]:
        """Get all categories"""
        return list(self.categories.values())
    
    def get_tag_stats(self) -> Dict:
        """Get tag statistics"""
        return {
            'total_tags': len(self.tags),
            'total_assignments': len(self.assignments),
            'most_used': self.tag_counter.most_common(10),
            'categories': {
                cat: len([t for t in self.tags.values() if t.category == cat])
                for cat in set(t.category for t in self.tags.values())
            }
        }
    
    def get_tag_analytics(self) -> pd.DataFrame:
        """Get tag analytics as DataFrame"""
        data = []
        for tag_name, count in self.tag_counter.items():
            tag = self.get_tag_by_name(tag_name)
            data.append({
                'Tag': tag_name,
                'Count': count,
                'Category': tag.category if tag else 'unknown',
                'Confidence': tag.confidence if tag else 0,
                'Auto': tag.is_auto_generated if tag else False
            })
        return pd.DataFrame(data)

# ── Auto-Categorizer ──────────────────────────────────────────────────────

class AutoCategorizer:
    """Automatically categorizes documents"""
    
    def __init__(self, tag_manager: TagManager, tag_generator: IntelligentTagGenerator):
        self.tag_manager = tag_manager
        self.tag_generator = tag_generator
        self.categorization_history = []
    
    def categorize_document(self, document_name: str, content: str, 
                           user_id: str = 'system') -> Dict:
        """Automatically categorize a document"""
        if not content:
            return {'status': 'failed', 'reason': 'No content'}
        
        # Generate tags
        tags = self.tag_generator.generate_tags(content)
        
        # Detect category
        category, confidence = self.tag_generator.generate_categories(content)
        
        # Assign tags
        assigned = []
        for tag_name, conf in tags[:5]:  # Limit to top 5 tags
            assignment_id = self.tag_manager.assign_tag(
                document_name, tag_name, user_id, auto=True
            )
            if assignment_id:
                assigned.append(tag_name)
        
        # Set category
        category_id = self._get_category_id(category)
        
        result = {
            'document_name': document_name,
            'category': category,
            'category_confidence': confidence,
            'assigned_tags': assigned,
            'total_tags_generated': len(tags),
            'timestamp': datetime.now()
        }
        
        self.categorization_history.append(result)
        return result
    
    def _get_category_id(self, category_name: str) -> Optional[str]:
        """Get category ID by name"""
        for cat in self.tag_manager.get_all_categories():
            if cat.name.lower() == category_name.lower():
                return cat.id
        return None
    
    def batch_categorize(self, documents: Dict[str, str], 
                         user_id: str = 'system') -> List[Dict]:
        """Categorize multiple documents"""
        results = []
        for doc_name, content in documents.items():
            result = self.categorize_document(doc_name, content, user_id)
            results.append(result)
        return results
    
    def get_categorization_stats(self) -> Dict:
        """Get categorization statistics"""
        if not self.categorization_history:
            return {'total': 0}
        
        categories = [r['category'] for r in self.categorization_history]
        category_counts = Counter(categories)
        
        return {
            'total': len(self.categorization_history),
            'categories': dict(category_counts),
            'avg_tags': sum(len(r['assigned_tags']) for r in self.categorization_history) / len(self.categorization_history)
        }

# ── Tag Suggestion Engine ──────────────────────────────────────────────────

class TagSuggestionEngine:
    """Provides tag suggestions based on content and context"""
    
    def __init__(self, tag_manager: TagManager, tag_generator: IntelligentTagGenerator):
        self.tag_manager = tag_manager
        self.tag_generator = tag_generator
        self.suggestion_history = []
    
    def suggest_tags(self, content: str, existing_tags: List[str] = None, 
                     max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Suggest tags for a document"""
        suggestions = []
        
        # Generate tags from content
        generated = self.tag_generator.generate_tags(content, max_suggestions * 2)
        
        # Filter out existing tags
        if existing_tags:
            generated = [(t, s) for t, s in generated if t not in existing_tags]
        
        # Get popular tags from the system
        popular = self.tag_manager.tag_counter.most_common(10)
        popular_tags = [t for t, _ in popular if t not in [g[0] for g in generated]]
        
        # Combine suggestions
        for tag_name, score in generated[:max_suggestions]:
            suggestions.append((tag_name, score))
        
        # Add popular tags if needed
        remaining = max_suggestions - len(suggestions)
        for tag_name in popular_tags[:remaining]:
            suggestions.append((tag_name, 0.5))
        
        self.suggestion_history.append({
            'timestamp': datetime.now(),
            'content_length': len(content),
            'suggestions': suggestions
        })
        
        return suggestions[:max_suggestions]
    
    def get_suggestion_stats(self) -> Dict:
        """Get suggestion statistics"""
        return {
            'total_suggestions': len(self.suggestion_history),
            'avg_suggestions': sum(len(s['suggestions']) for s in self.suggestion_history) / len(self.suggestion_history) if self.suggestion_history else 0
        }

# ── UI Components ──────────────────────────────────────────────────────────

def render_tag_management_ui(tag_manager: TagManager, document_name: str = None):
    """Render tag management UI"""
    st.subheader("🏷️ Tag Management")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Current Tags",
        "➕ Add Tags",
        "📊 Analytics",
        "📁 Categories"
    ])
    
    with tab1:
        if document_name:
            tags = tag_manager.get_document_tags(document_name)
            
            if tags:
                st.markdown(f"**Tags for {document_name}:**")
                cols = st.columns(4)
                for idx, tag in enumerate(tags):
                    with cols[idx % 4]:
                        st.markdown(
                            f"<span style='background:{tag.category};color:white;"
                            f"padding:4px 12px;border-radius:12px;'>"
                            f"{tag.name} {'' if tag.confidence == 1.0 else f'({tag.confidence*100:.0f}%)'}"
                            f"</span>",
                            unsafe_allow_html=True
                        )
                        
                        if st.button(f"❌", key=f"remove_{tag.id}_{document_name}"):
                            tag_manager.unassign_tag(document_name, tag.name)
                            st.rerun()
            else:
                st.info("No tags assigned to this document.")
        else:
            st.info("Select a document to view tags.")
    
    with tab2:
        if document_name:
            existing_tags = [t.name for t in tag_manager.get_document_tags(document_name)]
            
            col1, col2 = st.columns(2)
            with col1:
                new_tag = st.text_input("Tag name:", key="new_tag_input")
                category = st.selectbox(
                    "Category:",
                    ['custom', 'topic', 'type', 'status'],
                    key="tag_category_select"
                )
            
            with col2:
                confidence = st.slider("Confidence:", 0.0, 1.0, 1.0, 0.05)
            
            if st.button("➕ Add Tag", key="add_tag_btn"):
                if new_tag:
                    if new_tag not in existing_tags:
                        tag = tag_manager.add_tag(new_tag, category, confidence)
                        tag_manager.assign_tag(document_name, new_tag)
                        st.success(f"✅ Tag '{new_tag}' added!")
                        st.rerun()
                    else:
                        st.warning(f"Tag '{new_tag}' already assigned.")
        else:
            st.info("Select a document to add tags.")
    
    with tab3:
        render_tag_analytics(tag_manager)
    
    with tab4:
        render_category_manager(tag_manager)

def render_tag_analytics(tag_manager: TagManager):
    """Render tag analytics"""
    st.subheader("📊 Tag Analytics")
    
    stats = tag_manager.get_tag_stats()
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tags", stats['total_tags'])
    col2.metric("Total Assignments", stats['total_assignments'])
    col3.metric("Categories", len(stats['categories']))
    
    # Most used tags
    if stats['most_used']:
        st.subheader("🔥 Most Used Tags")
        tag_data = pd.DataFrame(stats['most_used'], columns=['Tag', 'Count'])
        st.bar_chart(tag_data.set_index('Tag'))
    
    # Category distribution
    if stats['categories']:
        st.subheader("📁 Category Distribution")
        cat_data = pd.DataFrame({
            'Category': list(stats['categories'].keys()),
            'Count': list(stats['categories'].values())
        })
        st.bar_chart(cat_data.set_index('Category'))
    
    # Tag usage table
    st.subheader("📋 Tag Usage Details")
    df = tag_manager.get_tag_analytics()
    st.dataframe(df, use_container_width=True)

def render_category_manager(tag_manager: TagManager):
    """Render category management UI"""
    st.subheader("📁 Category Manager")
    
    # Existing categories
    categories = tag_manager.get_all_categories()
    
    if categories:
        for cat in categories:
            with st.expander(f"📂 {cat.name}"):
                st.markdown(f"**Description:** {cat.description}")
                st.markdown(f"**Color:** {cat.color}")
                st.markdown(f"**Created:** {cat.created_at.strftime('%Y-%m-%d %H:%M')}")
                
                # Tags in this category
                cat_tags = [t for t in tag_manager.get_all_tags() if t.category == cat.id]
                if cat_tags:
                    st.markdown("**Tags in this category:**")
                    st.write(", ".join([t.name for t in cat_tags]))
    
    # Add new category
    with st.expander("➕ Add New Category", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            cat_name = st.text_input("Category Name:", key="new_cat_name")
            cat_desc = st.text_area("Description:", key="new_cat_desc")
        with col2:
            cat_color = st.color_picker("Color:", "#808080", key="new_cat_color")
            parent_cat = st.selectbox(
                "Parent Category (optional):",
                ['None'] + [c.name for c in categories],
                key="parent_cat_select"
            )
        
        if st.button("Create Category", key="create_cat_btn"):
            if cat_name:
                parent_id = None
                if parent_cat != 'None':
                    parent = next((c for c in categories if c.name == parent_cat), None)
                    if parent:
                        parent_id = parent.id
                
                tag_manager.add_category(cat_name, cat_desc, parent_id, cat_color)
                st.success(f"✅ Category '{cat_name}' created!")
                st.rerun()

def render_auto_categorization_ui(tag_manager: TagManager, tag_generator: IntelligentTagGenerator):
    """Render auto-categorization UI"""
    st.subheader("🤖 Auto-Categorization")
    
    # Initialize auto-categorizer
    categorizer = AutoCategorizer(tag_manager, tag_generator)
    
    # Batch categorization
    st.subheader("📂 Batch Categorization")
    
    documents = st.session_state.get('document_names', [])
    if documents:
        selected_docs = st.multiselect(
            "Select documents to categorize:",
            options=documents,
            key="batch_categorize_select"
        )
        
        if selected_docs and st.button("🚀 Run Auto-Categorization", key="auto_cat_btn"):
            with st.spinner("Categorizing documents..."):
                # Get document contents
                doc_contents = {}
                for doc_name in selected_docs:
                    if doc_name in st.session_state.get('raw_texts', {}):
                        doc_contents[doc_name] = st.session_state['raw_texts'][doc_name]
                
                if doc_contents:
                    results = categorizer.batch_categorize(doc_contents)
                    st.success(f"✅ Categorized {len(results)} documents!")
                    
                    # Show results
                    for result in results:
                        st.markdown(f"**{result['document_name']}**")
                        st.markdown(f"- Category: {result['category']} ({result['category_confidence']*100:.0f}%)")
                        st.markdown(f"- Tags: {', '.join(result['assigned_tags'])}")
                        st.divider()
    else:
        st.info("No documents available. Upload documents first.")

def render_tag_suggestions_ui(tag_manager: TagManager, tag_generator: IntelligentTagGenerator):
    """Render tag suggestions UI"""
    st.subheader("💡 Tag Suggestions")
    
    suggestion_engine = TagSuggestionEngine(tag_manager, tag_generator)
    
    document_name = st.selectbox(
        "Select document:",
        options=st.session_state.get('document_names', []),
        key="suggestion_doc_select"
    )
    
    if document_name:
        content = st.session_state.get('raw_texts', {}).get(document_name, '')
        existing_tags = [t.name for t in tag_manager.get_document_tags(document_name)]
        
        if content:
            if st.button("💡 Get Suggestions", key="suggest_tags_btn"):
                suggestions = suggestion_engine.suggest_tags(content, existing_tags)
                
                if suggestions:
                    st.subheader("Suggested Tags:")
                    for tag_name, confidence in suggestions:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"**{tag_name}** ({confidence*100:.0f}% confidence)")
                        with col2:
                            if st.button("✅ Add", key=f"add_suggest_{tag_name}"):
                                tag_manager.add_tag(tag_name, 'custom', confidence)
                                tag_manager.assign_tag(document_name, tag_name)
                                st.success(f"✅ Added tag '{tag_name}'!")
                                st.rerun()
                        with col3:
                            if st.button("❌ Dismiss", key=f"dismiss_{tag_name}"):
                                pass
                else:
                    st.info("No new tag suggestions available.")
        else:
            st.warning("No content found for this document.")

def render_tagging_dashboard():
    """Render the complete tagging dashboard"""
    # Initialize tagging system
    if 'tag_manager' not in st.session_state:
        st.session_state['tag_manager'] = TagManager()
    if 'tag_generator' not in st.session_state:
        st.session_state['tag_generator'] = IntelligentTagGenerator()
    
    tag_manager = st.session_state['tag_manager']
    tag_generator = st.session_state['tag_generator']
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏷️ Tag Management",
        "🤖 Auto-Categorization",
        "💡 Tag Suggestions",
        "📊 Analytics"
    ])
    
    with tab1:
        document_name = st.selectbox(
            "Select Document:",
            options=st.session_state.get('document_names', []),
            key="tag_doc_select"
        )
        render_tag_management_ui(tag_manager, document_name)
    
    with tab2:
        render_auto_categorization_ui(tag_manager, tag_generator)
    
    with tab3:
        render_tag_suggestions_ui(tag_manager, tag_generator)
    
    with tab4:
        render_tag_analytics(tag_manager)

# ── Integration with Main App ─────────────────────────────────────────────

def integrate_tagging_system():
    """Initialize and integrate tagging system with main app"""
    if 'tag_manager' not in st.session_state:
        st.session_state['tag_manager'] = TagManager()
    if 'tag_generator' not in st.session_state:
        st.session_state['tag_generator'] = IntelligentTagGenerator()
    
    # Auto-categorize documents during upload
    if st.session_state.get('new_documents_uploaded', False):
        tag_manager = st.session_state['tag_manager']
        tag_generator = st.session_state['tag_generator']
        categorizer = AutoCategorizer(tag_manager, tag_generator)
        
        # Get new documents
        raw_texts = st.session_state.get('raw_texts', {})
        if raw_texts:
            categorizer.batch_categorize(raw_texts)
        
        st.session_state['new_documents_uploaded'] = False
    
    # Add tagging tab to main app
    st.subheader("🏷️ Document Tagging System")
    render_tagging_dashboard()

# ── End of Tagging System ──────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
# ── Collaboration Hub Imports ────────────────────────────────────────────
from app.components.collaboration_hub import (
    render_collaboration_hub,
    initialize_collaboration_hub,
    CollaborationHub,
    PlagiarismCase,
    CaseStatus,
    CasePriority,
    TeamWorkspace,
    DiscussionThread,
    ReviewQueue,
)
from app.components.auto_ml_optimizer import (
    AutoMLOptimizer,
    AutoOptimizationIntegration,
    OptimizationConfig,
    OptimizationMetrics,
    PatternProfile,
    UserFeedback,
    initialize_auto_ml,
    render_auto_ml_dashboard,
)

# Add with other imports (around line 200-250)
from app.components.real_time_monitor import (
    HealthChecker,
    MonitoringEngine,
    initialize_monitoring,
    render_real_time_monitor,
)

# Add with other imports (around line 200-250)
from app.components.report_generator import (
    PlagiarismReportGenerator,
    initialize_report_generator,
    render_report_generator_ui,
    render_scheduled_reports_ui,
)
from src.security.metadata_stripper import strip_exif_metadata
from src.utils.filename import (
    InvalidFileExtensionError,
    format_extension_badge,
    sanitize_filename,
    unique_filename,
    validate_document_extension,
)
from app.components.smart_search import (
    render_smart_search_ui,
    render_search_analytics,
    initialize_smart_search,
    SmartSearchEngine,
)
# ── Enhanced Batch Processor Imports ─────────────────────────────────────
from app.components.batch_processor_enhanced import (
    render_batch_processor_ui,
    render_batch_analytics,
    render_batch_scheduler_ui,
    initialize_batch_processor,
    EnhancedBatchProcessor,
    BatchScheduler,
    BatchJob,
    JobPriority,
    JobStatus,
)


# AI Detection Settings
enable_ai_detection = st.session_state.get("enable_ai_detection", True)
ai_threshold = st.session_state.get("ai_threshold", 0.65)


# ── Audit Logs View Import ──────────────────────────────────────────────
from app.views.audit_logs import render_audit_view

try:
    from streamlit_plotly_events import plotly_events  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plotly_events = None

from src.core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Validate required environment variables during application startup
REQUIRED_ENV_VARS = [
    "REDIS_URL",
    "PLAGIARISM_WEBHOOK_URL",
    "API_BEARER_TOKEN",
]
missing_env_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_env_vars:
    logger.warning(
        "Missing environment variables: %s. Some features may not work correctly. "
        "Please configure them in your .env file.",
        ", ".join(missing_env_vars),
    )

from src.db.auth import get_all_users, get_upload_count, init_db

# Import DB and Core initializations
from src.db.corpus_db import get_all_documents, get_total_document_count, init_corpus_db
from src.db.incidents import (
    get_all_incidents,
    get_total_incidents_count,
    init_incident_db,
    sync_flagged_incidents,
)
from src.utils.temp_manager import purge_expired_temp_files

init_corpus_db()
init_db()
purge_expired_temp_files()

# Add after existing imports
from app.components.enhanced_dashboard import (
    DocumentTrendAnalyzer,
    PlagiarismPatternAnalyzer,
    initialize_enhanced_dashboard,
    render_enhanced_analytics_tab,
    render_enhanced_document_analysis,
)

# Centralized imports & backward compatibility re-exports
from app.session_keys import SessionKeys
from app.state_manager import (
    TIMEOUT_LIMIT,
    check_session_timeout,
    get_active_sessions_count,
    init_session_state,
    save_preferences_callback,
    ui_exception_handler,
    update_global_activity,
)
from app.theme import (
    back_to_top_html,
    get_chart_colors,
    get_theme_name,
    inject_css,
    render_session_status_banner,
    render_timezone_footer,
    set_theme,
    version_check_widget_html,
)

# Extracted tab renderers (de-monolith refactor). These are called below but
# were never imported, so every tab that uses one raised NameError at runtime.
from app.views.audit_view import render_audit_view
from app.views.auth_view import handle_oauth_callbacks, render_login_view
from app.views.corpus_view import (
    render_corpus_header,
    render_document_management_sidebar,
    render_sidebar,
)
from app.views.drilldown_view import render_drilldown_view
from app.views.history_view import render_history_view
from app.views.matrix_view import render_matrix_view
from app.views.settings_view import render_settings_view
from app.views.upload_view import render_upload_section
from app.views.users_view import render_users_view
from src.core.ai_detector import detect_documents_ai_probability
from src.core.config import (
    DEFAULT_THRESHOLDS,
    PLAGIARISM_THRESHOLD,
    get_branding_config,
)
from src.core.document_parser import (
    DEFAULT_OCR_DPI,
    DEFAULT_OCR_LANGUAGE,
    SUPPORTED_OCR_LANGUAGES,
    extract_text,
    prepare_text_for_embedding,
)
from src.core.embedding_model import embed_chunks, embed_documents
from src.core.lexical_similarity import jaccard_similarity
from src.core.text_chunking import chunk_documents
from src.core.faiss_index import (
    build_index,
    build_index_from_matrix,
    load_index,
    load_or_rebuild_index,
    save_index,
    search_similar_chunks,
)
from src.core.pipeline import ChunkRecord, run_extraction_pipeline, run_pipeline
from src.core.similarity import (
    cosine_similarity,
    document_similarity_matrix,
    flag_plagiarism,
)
from src.db import (
    clear_all_data,
    delete_document,
    get_all_embeddings,
    get_chunk_registry,
)
from src.db.auth import (
    get_tour_completed,
    get_user_last_login,
    is_user_active,
    set_tour_completed,
    update_user_preferences,
)
from src.db.corpus_db import (
    get_document_by_hash,
    get_unique_class_sections,
)
from src.i18n.translator import _SUPPORTED_LANGUAGES, get_text
from src.utils.bulk_export import create_documents_bulk_zip_archive
from src.utils.diff_highlighter import highlight_overlap
from src.utils.file_parser import truncate_filename
from src.utils.processing_time import (
    estimate_processing_seconds,
    format_processing_duration,
)
from src.utils.redis_cache import (
    cache_session_state,
    clear_session,
    get_analysis_results,
    get_faiss_index,
    get_session_state,
)
from src.utils.storage_metrics import calculate_storage_usage
from src.core.parse_durations import get_all_parse_durations, format_duration
from src.visualization.heatmap import (
    plot_similarity_heatmap,
)
from src.visualization.network_graph import plot_similarity_network

try:
    from src.utils.warning_list import (
        render_copy_button,
        render_warning_controls,
        reset_warning_page,
    )
    from src.visualization.analytics import (
        plot_high_severity_trends,
        plot_most_plagiarized_documents,
        plot_processing_time_breakdown,
        plot_similarity_distribution,
    )
except ImportError:
    render_warning_controls = None
    render_copy_button = None
    reset_warning_page = None
    plot_high_severity_trends = None
    plot_most_plagiarized_documents = None
    plot_processing_time_breakdown = None
    plot_similarity_distribution = None

try:
    from src.utils.pdf_highlighter import highlight_pdf_matches
except Exception:
    highlight_pdf_matches = None

try:
    from streamlit_tour import Tour
except ImportError:
    Tour = None

# ── Auto-refresh component for the Live Incident Stream (Issue #1384) ───────
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except ImportError:
    st_autorefresh = None
    logger.warning(
        "streamlit-autorefresh is not installed; the auto-refresh toggle "
        "on the Live Incident Stream tab will be disabled."
    )


try:
    from src.utils.google_drive import bulk_download_drive_folder
except Exception:
    bulk_download_drive_folder = None


# Initialize databases
init_corpus_db()
init_db()

# Purge stale temp files older than 2 hours on startup
purge_expired_temp_files()
# Start lightweight REST API server for /healthz endpoint in background
import threading

import uvicorn

import src.core.app_config as app_config
from src.api.app import app as fastapi_app


def update_global_activity():
    """Update the global last_activity timestamp."""
    try:
        from src.utils.redis_cache import get_cache

        cache = get_cache()
        cache.set("spd:v1:global:last_activity", time.time())
    except Exception as e:
        logger.error(f"Failed to update global activity: {e}")


# Register Streamlit user interaction (updates on every script rerun)
update_global_activity()


def _start_api_server():
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",
    )


if not getattr(app_config, "_api_server_started", False):
    app_config._api_server_started = True

    from starlette.middleware.base import BaseHTTPMiddleware

    class ActivityMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # Ignore health probes
            if request.url.path not in ("/health", "/healthz"):
                update_global_activity()
            return await call_next(request)

    fastapi_app.add_middleware(ActivityMiddleware)
    threading.Thread(target=_start_api_server, daemon=True).start()


def get_active_sessions_count() -> int:
    """Return the number of active Streamlit sessions."""
    try:
        from src.utils.redis_cache import get_cache, get_session_state

        cache = get_cache()
        now = time.time()
        active_count = 0
        keys = []

        if cache.is_available():
            try:
                raw_keys = cache._client.keys("spd:v1:session:*:last_interaction")
                keys = [
                    k.decode("utf-8") if isinstance(k, bytes) else k for k in raw_keys
                ]
            except Exception as e:
                logger.error(f"Failed to scan Redis session keys: {e}")

        try:
            fallback_keys = [
                k
                for k in cache.fallback_cache.keys()
                if k.startswith("spd:v1:session:") and k.endswith(":last_interaction")
            ]
            for k in fallback_keys:
                if k not in keys:
                    keys.append(k)
        except Exception as e:
            logger.error(f"Failed to scan fallback cache session keys: {e}")

        for key in keys:
            try:
                parts = key.split(":")
                if len(parts) >= 4:
                    session_id = parts[3]
                    last_interaction = get_session_state(
                        session_id,
                        "last_interaction",
                    )
                    if (
                        last_interaction is not None
                        and now - last_interaction <= 15 * 60
                    ):
                        active_count += 1
            except Exception as e:
                logger.error(f"Error checking session activity for {key}: {e}")

        return active_count

    except Exception as e:
        logger.error(f"Error in get_active_sessions_count: {e}")
        return 0


def _run_backup_daemon():
    """Background loop to create backups after inactivity."""
    last_backup_time = 0.0

    try:
        from src.utils.redis_cache import get_cache

        cache = get_cache()
        cached = cache.get("spd:v1:global:last_backup_time")
        if cached is not None:
            last_backup_time = float(cached)
    except Exception:
        pass

    logger.info("Database backup daemon started.")

    while True:
        time.sleep(30)

        try:
            from src.core.app_config import get_backup_idle_timeout
            from src.utils.redis_cache import get_cache

            cache = get_cache()

            timeout = get_backup_idle_timeout()

            last_activity = cache.get("spd:v1:global:last_activity")
            if last_activity is None:
                last_activity = time.time()
                cache.set("spd:v1:global:last_activity", last_activity)

            now = time.time()
            idle = now - last_activity

            if (
                get_active_sessions_count() == 0
                and idle >= timeout
                and last_activity > last_backup_time
            ):
                from src.db.corpus_db import get_corpus_db_path
                from src.db.database_backup import (
                    cleanup_old_backups,
                    create_corpus_database_snapshot,
                )

                from src.core.app_config import get_backup_dir

                snapshot = create_corpus_database_snapshot()

                backup_dir = get_backup_dir()
                backup_dir.mkdir(parents=True, exist_ok=True)

                filename = (
                    backup_dir
                    / f"corpus_backup_{datetime.now():%Y%m%d_%H%M%S}.db"
                )

                filename.write_bytes(snapshot)

                logger.info(f"Backup created: {filename}")
                cleanup_old_backups(backup_dir, max_backups=10, max_age_days=30)

                last_backup_time = now
                cache.set(
                    "spd:v1:global:last_backup_time",
                    last_backup_time,
                )

        except Exception as e:
            logger.exception(f"Backup daemon error: {e}")


if not getattr(app_config, "_backup_daemon_started", False):
    app_config._backup_daemon_started = True
    threading.Thread(
        target=_run_backup_daemon,
        daemon=True,
    ).start()

# Generate unique session ID for this Streamlit session
from app.session_manager import initialize_and_verify_session
st.session_state[SessionKeys.SESSION_ID] = initialize_and_verify_session()

SESSION_ID = st.session_state[SessionKeys.SESSION_ID]

# FAISS index location is centralized in src.core.app_config so this module,
# src/api/app.py, src/cli.py and src/utils/mock_data.py all agree on it.
# Cast to str because faiss.write_index / faiss.read_index require str paths.
from src.core.app_config import FAISS_INDEX_PATH

_INDEX_PATH = str(FAISS_INDEX_PATH)
branding_config = get_branding_config()

# -----------------------------------------------------------------------------
# Page Configuration & Session State
# -----------------------------------------------------------------------------


def configure_page_meta(title: str, icon: str) -> None:
    """Configure Streamlit page metadata including title, favicon, and layout."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Page title must be a non-empty string.")
    if not isinstance(icon, str) or not icon.strip():
        raise ValueError("Page icon must be a non-empty string.")

    st.set_page_config(
        page_title=title.strip(),
        page_icon=icon.strip(),
        layout="wide",
        initial_sidebar_state="auto",
    )


# Initialize page metadata with dynamic branding
configure_page_meta(title="Semantic Plagiarism Processor - Dashboard", icon="🔍")


def update_page_title(tab_name: str):
    """Update browser title based on active tab."""
    st.markdown(
        f"""
        <script>
            window.parent.document.title = '{tab_name} | Semantic Plagiarism Detector';
        </script>
        """,
        unsafe_allow_html=True,
    )


# Configure Page Setup
configure_page_meta(title="Semantic Plagiarism Detector - Dashboard", icon="🔍")
SESSION_ID = init_session_state()

st.markdown(back_to_top_html(), unsafe_allow_html=True)
inject_css()

# Session Timeout Check & Authentication Flow
last_interaction = check_session_timeout(SESSION_ID)
handle_oauth_callbacks(SESSION_ID)

def save_preferences_callback():
    """Persist settings to user DB profile when modified."""
    if st.session_state.get(SessionKeys.AUTHENTICATED) and st.session_state.get(
        SessionKeys.USERNAME
    ):
        prefs = {
            "threshold": st.session_state.get(
                SessionKeys.THRESHOLD_SLIDER, PLAGIARISM_THRESHOLD
            ),
            "theme": st.session_state.get("theme_selector", "Light"),
        }
        update_user_preferences(st.session_state[SessionKeys.USERNAME], prefs)


def build_visualization_lazily(is_enabled, build_fn):
    """Utility to lazily load heavy chart visualizations when requested."""
    if is_enabled:
        return build_fn()
    return None

@st.dialog("⚠️ Confirm Bulk Clear")
def clear_all_dialog():
    st.markdown(
        "**WARNING:** This action is destructive and cannot be undone. "
        "This will permanently delete all student documents, paragraph chunks, "
        "and plagiarism incidents from the database, and reset the FAISS index."
    )
    st.write("Are you absolutely sure you want to proceed?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True, key="cancel_clear_all"):
            st.rerun()
    with col2:
        if st.button(
            "Clear All",
            type="primary",
            use_container_width=True,
            key="confirm_clear_all",
        ):
            # ========== ADD THIS LINE ==========
            from src.utils.redis_cache import clear_all_large_data
            clear_all_large_data(SESSION_ID)
            # ===================================
            
            clear_all_data()
            if os.path.exists(_INDEX_PATH):
                try:
                    os.remove(_INDEX_PATH)
                except OSError as e:
                    print(f"Error removing FAISS index: {e}")
                except Exception as e:
                    logger.error(f"Error removing FAISS index: {e}")

            try:
                from src.utils.redis_cache import get_cache

                cache = get_cache()
                if cache.is_available():
                    cache.delete("faiss:index:corpus_index")
                    cache.clear_pattern("analysis:*")
            except (ImportError, RuntimeError, ConnectionError) as e:
                print(f"Error invalidating cache: {e}")
            except Exception as e:
                logger.error(f"Error invalidating cache: {e}")

            if "analysis_results" in st.session_state:
                st.session_state.analysis_results = None
            if "analysis_file_signature" in st.session_state:
                st.session_state.analysis_file_signature = None
            if "processed_pipeline_signature" in st.session_state:
                st.session_state.processed_pipeline_signature = None

            st.success("✅ All documents, chunks, and incidents have been cleared.")
            st.rerun()



# ── Issue #1383: Cosine vs Lexical Similarity Comparison Table ─────────────────
SEMANTIC_HIGH_THRESHOLD = 0.80  # vector (cosine) score considered "high"
LEXICAL_LOW_THRESHOLD = 0.30    # lexical (jaccard) score considered "low"


def render_cosine_vs_lexical_comparison_table(
    sim_df,
    raw_texts,
    *,
    semantic_threshold: float = SEMANTIC_HIGH_THRESHOLD,
    lexical_threshold: float = LEXICAL_LOW_THRESHOLD,
):
    """Render a two-column score comparison table in the results / drill-down view."""
    import itertools

    if sim_df is None or raw_texts is None or len(raw_texts) < 2:
        st.info(
            "Upload at least two documents to view the Cosine vs Lexical "
            "Similarity comparison table."
        )
        return None

    doc_names = list(sim_df.columns) if sim_df is not None else list(raw_texts.keys())
    rows = []
    for da, db in itertools.combinations(doc_names, 2):
        try:
            cosine_score = float(sim_df.loc[da, db])
        except Exception:
            cosine_score = 0.0

        text_a = raw_texts.get(da, "") or ""
        text_b = raw_texts.get(db, "") or ""
        try:
            jaccard_score = float(jaccard_similarity(text_a, text_b))
        except Exception:
            jaccard_score = 0.0

        is_semantic_only = (
            cosine_score >= semantic_threshold and jaccard_score <= lexical_threshold
        )
        rows.append(
            {
                "Document A": da,
                "Document B": db,
                "Cosine (Semantic)": cosine_score,
                "Jaccard (Lexical)": jaccard_score,
                "Semantic Only": is_semantic_only,
            }
        )

    comp_df = pd.DataFrame(rows)
    if not comp_df.empty:
        st.dataframe(comp_df, use_container_width=True)

    return comp_df


def get_date_range_preset(preset: str) -> tuple:
    """Calculate start and end dates based on a given preset string."""
    today = date.today()
    if preset == "Today":
        return today, today
    elif preset == "Last 7 Days":
        return today - timedelta(days=6), today
    elif preset == "Last 14 Days":
        return today - timedelta(days=14), today
    elif preset == "Last 30 Days":
        return today - timedelta(days=29), today
    else:  # "All Time"
        return date(2020, 1, 1), today


# ── SESSION TIMEOUT & ROUTE PROTECTION ────────────────────────────────────────
TIMEOUT_LIMIT = 15 * 60  # 15 minutes in seconds

cached_last_interaction = get_session_state(SESSION_ID, SessionKeys.LAST_INTERACTION)
if cached_last_interaction is not None:
    last_interaction = cached_last_interaction
elif SessionKeys.LAST_INTERACTION in st.session_state:
    last_interaction = st.session_state[SessionKeys.LAST_INTERACTION]
else:
    last_interaction = None

if last_interaction and st.session_state.get(SessionKeys.AUTHENTICATED, False):
    elapsed_time = time.time() - last_interaction
    if elapsed_time > TIMEOUT_LIMIT:
        for key in [
            SessionKeys.AUTHENTICATED,
            SessionKeys.USERNAME,
            SessionKeys.ROLE,
            SessionKeys.LAST_INTERACTION,
        ]:
            if key in st.session_state:
                del st.session_state[key]
        clear_session(SESSION_ID)
        from src.errors import UI_SESSION_EXPIRED

        st.warning(UI_SESSION_EXPIRED)
        st.stop()
    else:
        st.session_state[SessionKeys.LAST_INTERACTION] = time.time()
        cache_session_state(SESSION_ID, SessionKeys.LAST_INTERACTION, time.time())

# ── Handle OAuth Callback (GitHub / Google SSO) ──────────────────────────────
if not st.session_state.get(SessionKeys.AUTHENTICATED, False):
    if "code" in st.query_params and "state" in st.query_params:
        _code = st.query_params["code"]
        _state = st.query_params["state"]
        from src.db.auth import get_or_create_sso_user
        from src.utils.sso import exchange_github_code, exchange_google_code

        _user_info, _error_msg = None, None
        if _state.startswith("google_"):
            _user_info, _error_msg = exchange_google_code(_code)
        elif _state.startswith("github_"):
            _user_info, _error_msg = exchange_github_code(_code)

        if _user_info and _user_info.get("email"):
            _email = _user_info["email"]
            if not is_user_active(_email):
                st.error("🚨 Account suspended. Please contact your Administrator.")
                st.query_params.clear()
            else:
                _role = get_or_create_sso_user(_email)
                st.session_state[SessionKeys.AUTHENTICATED] = True
                st.session_state[SessionKeys.USERNAME] = _email
                st.session_state[SessionKeys.ROLE] = _role
                st.session_state[SessionKeys.LAST_INTERACTION] = time.time()
                cache_session_state(SESSION_ID, SessionKeys.AUTHENTICATED, True)
                cache_session_state(SESSION_ID, SessionKeys.USERNAME, _email)
                cache_session_state(SESSION_ID, SessionKeys.ROLE, _role)
                cache_session_state(
                    SESSION_ID, SessionKeys.LAST_INTERACTION, time.time()
                )
                st.query_params.clear()
                st.rerun()
        else:
            _err = _error_msg or "Could not retrieve your email."
            st.error(f"🚨 SSO authentication failed: {_err}")
            st.query_params.clear()

# Render Login UI if not authenticated
if not st.session_state.get(SessionKeys.AUTHENTICATED, False):
    render_login_view(SESSION_ID)

def file_uploader_callback():
    uploaded = st.session_state.get("file_uploader")
    if uploaded:
        st.session_state["staged_files_count"] = len(uploaded)
        total_size = sum(getattr(f, "size", 0) for f in uploaded)
        st.session_state["staged_files_size"] = total_size
    else:
        st.session_state["staged_files_count"] = 0
        st.session_state["staged_files_size"] = 0


if "staged_files_count" not in st.session_state:
    st.session_state["staged_files_count"] = 0
if "staged_files_size" not in st.session_state:
    st.session_state["staged_files_size"] = 0

user_role = st.session_state.get(SessionKeys.ROLE, "user")

# Top-right Theme Toggle
current_theme = get_theme_name()
_, theme_col = st.columns([0.94, 0.06])
with theme_col:
    theme_icon = "☀️" if current_theme == "Dark" else "🌙"
    if st.button(theme_icon, key="theme_toggle"):
        new_theme = "Light" if current_theme == "Dark" else "Dark"
        set_theme(new_theme)
        st.rerun()

# Corpus Overview Header & Quick Actions
render_corpus_header(_INDEX_PATH)

# ── Dialogs ───────────────────────────────────────────────────────────────────
@st.dialog("⚠️ Confirm Logout")
def logout_dialog():
    st.write("Are you sure you want to log out?")
    st.info("Your current session will be cleared.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True, key="cancel_logout"):
            st.rerun()
    with col2:
        if st.button(
            "Log Out", type="primary", use_container_width=True, key="confirm_logout"
        ):
            username = st.session_state.get(SessionKeys.USERNAME, "unknown")
            timestamp = datetime.now(timezone.utc).isoformat()
            logger.info("User '%s' logged out at %s", username, timestamp)
            
            # ========== ADD THIS LINE ==========
            from src.utils.redis_cache import clear_all_large_data
            clear_all_large_data(SESSION_ID)
            # ===================================
            
            for key in [
                SessionKeys.AUTHENTICATED,
                SessionKeys.USERNAME,
                SessionKeys.ROLE,
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            clear_session(SESSION_ID)
            st.rerun()



# ── Corpus Overview Header & Quick Actions (#1242) ───────────────────────────
header_col, action_col1, action_col2 = st.columns([0.5, 0.25, 0.25])

with header_col:
    st.subheader("📚 Corpus Overview")

with action_col1:
    if st.button(
        "🔄 Refresh Corpus Data", key="refresh_corpus_btn", use_container_width=True
    ):
        keys_to_clear = [
            SessionKeys.ANALYSIS_RESULTS,
            SessionKeys.ANALYSIS_FILE_SIGNATURE,
            SessionKeys.DRIVE_FILES_DICT,
            SessionKeys.FAILED_DOCUMENTS,
            SessionKeys.WARNING_PAGE,
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.cache_data.clear()
        st.toast("Corpus dataset refreshed.", icon="✅")
        st.rerun()

with action_col2:
    if st.button(
        "🗑️ Clear All Data",
        key="open_clear_dialog_btn",
        type="secondary",
        use_container_width=True,
    ):
        clear_all_dialog()  # type: ignore

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Account Info (Issue: logged-in user details expander) ──────────────
    if st.session_state.get(SessionKeys.AUTHENTICATED, False):
        _current_username = st.session_state.get(SessionKeys.USERNAME) or "Unknown"
        with st.sidebar.expander(f"👤 Logged in as: {_current_username}"):
            st.markdown(f"**Username:** {_current_username}")
            st.markdown(f"**Role:** {user_role.capitalize() if user_role else 'N/A'}")
            try:
                _last_login = get_user_last_login(_current_username)
            except Exception:
                _last_login = None
            st.markdown(f"**Last Login:** {_last_login if _last_login else 'N/A'}")

    try:
        from src.db.auth import get_upload_count
        total_scans_sidebar = get_upload_count()
    except Exception as e:
        logger.error(f"Failed to query total scan count for sidebar: {e}")
        total_scans_sidebar = 0

    st.markdown(f"Total Scans Processed: {total_scans_sidebar:,}")
    st.markdown("### ⚙️ Settings")

    lang_options = list(_SUPPORTED_LANGUAGES.values())
    st.selectbox(
        "🌐 Language",
        options=lang_options,
        key=SessionKeys.LANG_SELECTOR,
    )
    selected_lang_name = st.session_state.get(SessionKeys.LANG_SELECTOR, "English")
    _lang_reverse = {v: k for k, v in _SUPPORTED_LANGUAGES.items()}
    lang_code = _lang_reverse.get(selected_lang_name, "en")

    if user_role == "admin":
        initialize_enhanced_dashboard()
        initialize_report_generator()
        # ── Threshold Presets (Issue #1674) ───────────────────────────────────────
        st.markdown("### 🎯 Threshold Presets")

        # Define preset options with descriptions
        preset_options = {
            "Strict (0.80)": 0.80,
            "Balanced (0.59)": 0.59,
            "Lenient (0.45)": 0.45,
            "Custom": None,
        }

        # Determine current preset based on session state threshold
        current_threshold = st.session_state.get("threshold_slider", PLAGIARISM_THRESHOLD)
        current_preset = "Custom"
        for label, value in preset_options.items():
            if value is not None and abs(current_threshold - value) < 0.001:
                current_preset = label
                break

        selected_preset = st.radio(
            "Select Evaluation Standard:",
            options=list(preset_options.keys()),
            index=list(preset_options.keys()).index(current_preset),
            key="threshold_preset_radio",
            horizontal=True,
            help="Choose a predefined threshold standard or use the custom slider below.",
        )

        # Sync preset selection with slider value
        if selected_preset != "Custom" and preset_options[selected_preset] is not None:
            st.session_state["threshold_slider"] = preset_options[selected_preset]
            # Force rerun to update the slider widget if it changed via radio
            if current_preset != selected_preset:
                st.rerun()

        threshold = st.slider(
            "Plagiarism Threshold (Hybrid)",
            0.10,
            0.99,
            value=st.session_state.get("threshold_slider", PLAGIARISM_THRESHOLD),
            step=0.01,
            help=(
                "Combined Hybrid score threshold for flagging pair plagiarism. "
                "Calculated from Lexical (exact phrase overlap) and Semantic (meaning alignment) scores. "
                "Recommended Default: 0.59 (59%)."
            ),
            key="threshold_slider",
            on_change=save_preferences_callback,
        )

        # If user manually changes slider, reset preset to "Custom"
        if abs(threshold - preset_options.get(selected_preset, -1)) > 0.001:
            if st.session_state.get("threshold_preset_radio") != "Custom":
                st.session_state["threshold_preset_radio"] = "Custom"
                st.rerun()

        lexical_threshold = st.slider(
            "Lexical Sensitivity Threshold",
            0.10,
            1.00,
            value=0.50,
            step=0.05,
            help=(
                "Direct word-for-word and N-gram match threshold. "
                "Higher values require near-identical text phrasing to trigger alerts. "
                "Recommended Default: 0.50 (50%)."
            ),
            key=SessionKeys.LEXICAL_THRESHOLD_SLIDER,
        )

        semantic_threshold = st.slider(
            "Semantic Sensitivity Threshold",
            0.10,
            1.00,
            value=0.65,
            step=0.05,
            help=(
                "Transformer embedding vector similarity threshold measuring conceptual alignment and paraphrasing. "
                "Higher values require strong contextual similarity even if words differ. "
                "Recommended Default: 0.65 (65%)."
            ),
            key=SessionKeys.SEMANTIC_THRESHOLD_SLIDER,
        )

        # Cross-Lingual Detection Settings
        from app.components.cross_lingual_ui import render_cross_lingual_settings
        cross_lingual_mode = render_cross_lingual_settings()

        use_chunk_matrix = st.checkbox(
            "Use chunk-level similarity matrix",
            value=False,
            key=SessionKeys.CHUNK_MATRIX_CHECKBOX,
        )
        faiss_top_k = st.slider(
            "FAISS: matches per chunk",
            1,
            20,
            value=5,
            key=SessionKeys.FAISS_TOP_K_SLIDER,
        )


        use_chunk_matrix = st.checkbox(
            "Use chunk-level similarity matrix",
            value=False,
            key=SessionKeys.CHUNK_MATRIX_CHECKBOX,
        )
        faiss_top_k = st.slider(
            "FAISS: matches per chunk",
            1,
            20,
            value=5,
            key=SessionKeys.FAISS_TOP_K_SLIDER,
        )

        # ========== ADD THIS ==========
        # Cross-Lingual Detection Toggle (Issue #1956)
        cross_lingual_mode = st.toggle(
            "🌐 Cross-Lingual Detection (Beta)",
            value=False,
            key="cross_lingual_mode_toggle",
            help=(
                "Enable back-translation to detect translated plagiarism. "
                "Chunks in foreign languages will be translated to English "
                "before similarity matching. May increase processing time."
            ),
        )
        # ==============================


        from app.components.faiss_results import render_faiss_metric_badge
        render_faiss_metric_badge(st.session_state.get("faiss_index", None))

        # ── FAISS Vector Index Memory Footprint Badge (Issue #1563) ────────────
        from src.core.faiss_index import format_faiss_memory_badge
        current_faiss_index = globals().get("faiss_index")
        if current_faiss_index is None and "faiss_index" in st.session_state:
            current_faiss_index = st.session_state["faiss_index"]
        faiss_badge_text = format_faiss_memory_badge(current_faiss_index)
        st.caption(f"⚡ **{faiss_badge_text}**")

        st.markdown("### ✂️ Chunking Settings")
        chunk_size = st.slider(
            "Chunk Size (characters)",
            200,
            2000,
            value=500,
            step=50,
            help="Target character length for text chunks during embedding.",
            key=SessionKeys.CHUNK_SIZE_SLIDER,
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            0,
            500,
            value=50,
            step=10,
            help="Character overlap between consecutive chunks to preserve contextual boundary.",
            key=SessionKeys.CHUNK_OVERLAP_SLIDER,
        )

        with st.expander("🔤 OCR Settings", expanded=False):
            ocr_language_labels = {
                display_name: code
                for code, display_name in SUPPORTED_OCR_LANGUAGES.items()
            }
            language_names = list(ocr_language_labels)
            default_language_name = SUPPORTED_OCR_LANGUAGES.get(DEFAULT_OCR_LANGUAGE, "English")
            default_index = language_names.index(default_language_name) if default_language_name in language_names else 0
            selected_ocr_language_name = st.selectbox(
                "OCR Language",
                options=language_names,
                index=default_index,
                key=SessionKeys.OCR_LANGUAGE_SELECTOR,
            )
            ocr_language = ocr_language_labels[selected_ocr_language_name]

            ocr_dpi = st.slider(
                "OCR DPI Resolution",
                min_value=150,
                max_value=400,
                value=DEFAULT_OCR_DPI,
                step=25,
                key=SessionKeys.OCR_DPI_SLIDER,
            )






        # ── Multilingual Support ──────────────────────────────────────────────────
        with st.sidebar.expander("🌍 Multilingual Support", expanded=False):
            st.markdown("""
            **Multilingual support** enables plagiarism detection for non-Latin scripts
            (Arabic, Devanagari, Cyrillic, etc.)
            """)
            
            enable_multilingual = st.checkbox(
                "Enable Multilingual Support",
                value=False,
                key="enable_multilingual",
                help="Enables normalization for Arabic, Devanagari, Cyrillic scripts"
            )
            
            if enable_multilingual:
                st.info("✅ Arabic, Devanagari, Cyrillic normalization enabled")
                
                with st.expander("📖 Supported Scripts", expanded=False):
                    st.markdown("""
                    - **Arabic**: Diacritic removal, letter normalization, ligature handling
                    - **Devanagari**: Matra normalization, consonant normalization
                    - **Cyrillic**: Letter normalization (ё→е, й→и, etc.)
                    - **Latin**: Basic Unicode normalization
                    """)
                
                # Show detected script for current text
                if 'raw_texts' in locals() and raw_texts:
                    from src.core.script_normalizer import ScriptDetector
                    detector = ScriptDetector()
                    scripts = {}
                    for doc, text in raw_texts.items():
                        scripts[doc] = detector.detect(text)
                    
                    st.caption("Detected Scripts:")
                    for doc, script in scripts.items():
                        st.caption(f"- {doc}: {script}")

        # ── Cross-Lingual Detection ──────────────────────────────────────────────────
        with st.sidebar.expander("🌐 Cross-Lingual Detection", expanded=False):
            st.markdown("""
            **Cross-lingual detection** identifies plagiarism across different languages
            using translation and multilingual embeddings.
            """)
            
            enable_cross_lingual = st.checkbox(
                "Enable Cross-Lingual Detection",
                value=False,
                key="enable_cross_lingual",
                help="Detect plagiarism across different languages"
            )
            
            if enable_cross_lingual:
                cross_lingual_method = st.selectbox(
                    "Detection Method",
                    ["hybrid", "embedding", "translation"],
                    index=0,
                    help="Hybrid = translation + embeddings (best), Embedding = LaBSE only, Translation = translation only"
                )
                
        cross_lingual_threshold = st.slider(
            "Cross-Lingual Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.60,
            step=0.05,
            help="Similarity threshold for cross-lingual flagging",
        )
        st.info(f"Method: {cross_lingual_method} | Threshold: {cross_lingual_threshold:.2f}")

# ── AI Plagiarism Detection ────────────────────────────────────────────────
with st.sidebar.expander("🤖 AI Plagiarism Detection", expanded=False):
    st.markdown("""
**AI Detection** identifies text generated by LLMs (ChatGPT, Claude, etc.)
using multiple statistical techniques.
""")
    enable_ai_detection = st.checkbox(
        "Enable AI Detection",
        value=True,
        key="enable_ai_detection",
        help="Detect AI-generated text in documents"
    )
    if enable_ai_detection:
        ai_threshold = st.slider(
            "AI Detection Threshold",
            min_value=0.30,
            max_value=0.90,
            value=0.65,
            step=0.05,
            help="Higher = stricter AI detection"
        )
        st.info(f"Documents with AI probability > {ai_threshold:.2f} will be flagged")

# Show detection methods
with st.expander("🔍 Detection Methods", expanded=False):
    st.markdown("""
    - **Perplexity**: AI text is more predictable
    - **Burstiness**: Human text has more variation
    - **Pattern Analysis**: Detects repetitive AI patterns
    - **Sentence Variability**: AI text has less variety
    """)
# ── Stopword Manager ────────────────────────────────────────────────────────
with st.sidebar.expander("🛑 Stopword Manager", expanded=False):
    from app.components.stopword_manager_ui import render_stopword_manager_ui
    render_stopword_manager_ui()


# ── Hybrid Scoring Settings ────────────────────────────────────────────────
with st.sidebar.expander("🔀 Hybrid Scoring", expanded=False):
    st.markdown("""
    **Hybrid scoring** combines lexical and semantic similarity for more accurate detection.
    """)
            
    use_hybrid = st.checkbox(
        "Enable Hybrid Scoring",
        value=False,
        key="use_hybrid_scoring",
        help="Combines lexical (TF-IDF/Jaccard) and semantic (BERT) similarity"
    )
            
    if use_hybrid:
        alpha = st.slider(
            "Semantic Weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher = more semantic, Lower = more lexical"
        )
                
        lexical_method = st.selectbox(
            "Lexical Method",
            ["tfidf", "jaccard", "dice", "overlap", "ngram", "char_ngram"],
            index=0,
            help="Method for lexical similarity"
        )
                
        if lexical_method == "ngram":
            ngram_n = st.slider("N-gram Size", 2, 5, 3)
        elif lexical_method == "char_ngram":
            char_ngram_n = st.slider("Char N-gram Size", 3, 7, 5)
                
        st.info(f"Currently using: {lexical_method} + semantic (α={alpha:.2f})")

    else:
        threshold = PLAGIARISM_THRESHOLD
        use_chunk_matrix = False
        faiss_top_k = 5
        chunk_size = 500
        chunk_overlap = 50
        ocr_language = DEFAULT_OCR_LANGUAGE
        ocr_dpi = DEFAULT_OCR_DPI

    # ── API Quota Usage Gauge (Issue #1566) ──────────────────────────────────
    from app.components.api_quota_gauge import render_api_quota_gauge
    render_api_quota_gauge()


    unique_classes = get_unique_class_sections()
    selected_classes = st.multiselect(
        "Select Class/Section(s)",
        unique_classes,
        default=unique_classes,
        key=SessionKeys.CLASS_FILTER_SELECTBOX,
    )
    # The local `faiss_index` is not built until further down this script, so
    # read the cached handle from session state instead of referencing a name
    # that does not exist yet. render_sidebar() applies the same fallback.
    lang_code = render_sidebar(
        user_role,
        str(ROOT_DIR),
        st.session_state.get("faiss_index"),
    )

    # ── System Health Widget (Issue #1246) ──────────────────────────────────────
    with st.expander("🖥️ System Health & Memory", expanded=False):
        try:
            import os
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)

            host_limit_mb = 2048.0
            try:
                from src.core.app_config import get_host_memory_limit_mb
                host_limit_mb = float(get_host_memory_limit_mb())
            except Exception:
                pass

            ram_usage_percent = min(rss_mb / host_limit_mb, 1.0)
            ram_percent_val = (rss_mb / host_limit_mb) * 100

            if ram_percent_val >= 80:
                st.warning(f"⚠️ High RAM Usage: {rss_mb:.1f} MB / {host_limit_mb:.0f} MB ({ram_percent_val:.0f}%)")
            else:
                st.markdown(f"**RAM Usage:** {rss_mb:.1f} MB / {host_limit_mb:.0f} MB ({ram_percent_val:.0f}%)")

            st.progress(ram_usage_percent)

            disk_usage = psutil.disk_usage(str(ROOT_DIR))
            free_disk_gb = disk_usage.free / (1024**3)
            total_disk_gb = disk_usage.total / (1024**3)
            disk_usage_percent = disk_usage.percent

            if disk_usage_percent >= 90:
                disk_indicator = "🔴"
            elif disk_usage_percent >= 75:
                disk_indicator = "🟡"
            else:
                disk_indicator = "🟢"

            st.markdown(
                f"**💿 Disk Space:** {disk_indicator} {disk_usage_percent:.1f}% used"
            )
            st.caption(f"Free: {free_disk_gb:.1f} GB · Total: {total_disk_gb:.1f} GB")

            st.divider()
            st.markdown("**🗄️ Database Status**")

            try:
                from src.core.app_config import AUTH_DB_PATH, CORPUS_DB_PATH

                corpus_db_exists = CORPUS_DB_PATH.exists()
                if corpus_db_exists:
                    st.markdown("• **Corpus DB:** 🟢 Connected")
                    corpus_size_kb = CORPUS_DB_PATH.stat().st_size / 1024
                    st.caption(f"  Size: {corpus_size_kb:.1f} KB")
                else:
                    st.markdown("• **Corpus DB:** 🟡 Not initialized")
                    st.caption("  Will be created on first data upload.")

            except Exception as db_err:
                st.markdown("• **Corpus DB:** 🔴 Error")
                st.caption(f"  {db_err}")

            try:
                auth_db_exists = AUTH_DB_PATH.exists()
                if auth_db_exists:
                    st.markdown("• **Auth DB:** 🟢 Connected")
                    auth_size_kb = AUTH_DB_PATH.stat().st_size / 1024
                    st.caption(f"  Size: {auth_size_kb:.1f} KB")
                else:
                    st.markdown("• **Auth DB:** 🟡 Not initialized")
                    st.caption("  Will be created on first login.")

            except Exception as db_err:
                st.markdown("• **Auth DB:** 🔴 Error")
                st.caption(f"  {db_err}")

            try:
                from src.utils.redis_cache import get_cache

                cache_inst = get_cache()
                redis_online, latency = cache_inst.ping()
                if redis_online:
                    lat_str = f" ({latency} ms)" if latency is not None else ""
                    st.markdown(f"• **Cache Backend:** 🟢 Redis{lat_str}")
                else:
                    st.markdown("• **Cache Backend:** 🟡 In-Memory")
            except Exception:
                st.markdown("• **Cache Backend:** 🟡 In-Memory")

            st.divider()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count(logical=True)

            if cpu_percent >= 90:
                cpu_indicator = "🔴"
            elif cpu_percent >= 70:
                cpu_indicator = "🟡"
            else:
                cpu_indicator = "🟢"

            st.markdown(
                f"**⚡ CPU Load:** {cpu_indicator} {cpu_percent:.1f}% "
                f"({cpu_count} cores)"
            )

        except ImportError:
            st.warning("⚠️ psutil not available. System health data unavailable.")
        except Exception as health_err:
            st.error(f"Failed to load system health data: {health_err}")

        st.divider()
        render_timezone_footer()

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🔍 Semantic Plagiarism Detection System")

# Live Scan Statistics Metrics Header (#1508)
try:
    total_scans = get_upload_count()
    corpus_size = get_total_document_count()
    flagged_incidents = get_total_incidents_count()

    _incidents = get_all_incidents(limit=10000)
    if _incidents:
        avg_sim = sum(inc.get("similarity_score", 0.0) for inc in _incidents) / len(_incidents)
    else:
        avg_sim = 0.0
except Exception as e:
    logger.error(f"Failed to load dashboard metrics: {e}")
    total_scans = 0
    corpus_size = 0
    flagged_incidents = 0
    avg_sim = 0.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Scans", f"{total_scans:,}")
with col2:
    st.metric("Avg Similarity %", f"{avg_sim * 100:.1f}%")
with col3:
    st.metric("Flagged Incidents", f"{flagged_incidents:,}")
with col4:
    st.metric("Corpus Size", f"{corpus_size:,}")
if enable_ai_detection and 'ai_probabilities' in locals() and ai_probabilities:
    suspicious_count = sum(1 for p in ai_probabilities.values() if p > ai_threshold)
    st.metric("🤖 AI Generated", suspicious_count, delta=f"of {len(ai_probabilities)} docs")
else:
    st.metric("🤖 AI Generated", "0")

st.markdown("---")

with st.expander("ℹ️ How Semantic Plagiarism Detection Works"):
    st.markdown("""
        - **1. Upload files** — Upload the documents you want to compare.
        - **2. AI vector embeddings generated** — The documents are converted into vector embeddings for semantic comparison.
        - **3. View similarity heatmap & incident logs** — Review detected similarities through the heatmap and incident logs.
        """)
uploaded_files = st.file_uploader(
    "📂 Upload Assignments",
    type=["pdf", "docx", "txt", "md", "markdown", "mdown"],
    accept_multiple_files=True,
    key="file_uploader",
    on_change=file_uploader_callback,
)

if st.session_state.get("staged_files_count", 0) > 0:
    staged_count = st.session_state["staged_files_count"]
    staged_size_mb = st.session_state["staged_files_size"] / (1024 * 1024)
    st.info(f"📁 Staged {staged_count} files (Total Size: {staged_size_mb:.1f} MB)")

if user_role != "admin":
    st.subheader("🔎 Secure Student Search Portal")
    st.caption(
        "Paste a text snippet below to check its similarity against existing indexed assignments."
    )

    st.info(
        "🔒 Note: Direct assignment uploads and detailed breakdown panels are restricted to Administrator access. Your queries are anonymized for privacy."
    )

    query_text = st.text_area(
        "Paste a text snippet to check against index:",
        height=150,
        placeholder="Paste a paragraph here to check for plagiarism...",
    )

    if st.button("🔍 Run Quick Verification", key="user_query") and query_text.strip():
        from src.core.faiss_index import build_index_from_matrix
        from src.db.corpus_db import get_all_embeddings, get_chunk_registry

        with st.spinner("Loading index and searching..."):
            try:
                registry = get_chunk_registry()
                embeddings_matrix = get_all_embeddings()

                if embeddings_matrix.shape[0] == 0:
                    from src.errors import UI_NO_DOCUMENTS_INDEXED

                    st.warning(UI_NO_DOCUMENTS_INDEXED)
                else:
                    faiss_index = build_index_from_matrix(
                        embeddings_matrix, index_type="auto"
                    )

                    from src.core.embedding_model import embed_chunks

                    query_vec = embed_chunks([query_text.strip()])[0]

                    faiss_threshold = threshold
                    results = search_similar_chunks(
                        query_vec,
                        faiss_index,
                        registry,
                        top_k=faiss_top_k,
                        threshold=faiss_threshold,
                    )

                    if not results:
                        st.success(
                            "✅ No significant matches found in the assignment database."
                        )
                    else:
                        st.success(
                            f"Found **{len(results)}** potentially similar passages."
                        )

                        doc_id_map = {}
                        anon_counter = 1

                        for record, score in results:
                            if record.doc_name not in doc_id_map:
                                doc_id_map[record.doc_name] = (
                                    f"Document-{anon_counter:03d}"
                                )
                                anon_counter += 1

                        for rank, (record, score) in enumerate(results, 1):
                            anon_doc_name = doc_id_map[record.doc_name]
                            color = "#ff4b4b" if score >= 0.90 else "#ffa500"

                            with st.expander(
                                f"#{rank} · {anon_doc_name} (chunk #{record.chunk_index + 1}) "
                                f"— {score:.1%}",
                                expanded=(rank == 1),
                            ):
                                cq, cm = st.columns(2)
                                with cq:
                                    st.markdown("**Your query:**")
                                    st.info(query_text.strip())
                                with cm:
                                    st.markdown(
                                        f"**Matching passage in {anon_doc_name}:**"
                                    )
                                    st.warning(record.chunk_text)

                                st.markdown(
                                    f"<div style='text-align:right;'>"
                                    f"<span style='background:{color};color:white;padding:3px 12px;"
                                    f"border-radius:10px;font-size:0.85rem;font-weight:700;'>"
                                    f"Similarity: {score * 100:.1f}%</span></div>",
                                    unsafe_allow_html=True,
                                )

                        st.caption(
                            "🔒 Document names are anonymized to protect student privacy."
                        )

            except Exception as e:
                from src.errors import UI_INDEX_LOAD_FAILED

                st.error(UI_INDEX_LOAD_FAILED.format(error=str(e)))
                st.info(
                    "Please ensure documents have been indexed by an administrator."
                )
else:
    if os.path.exists(_INDEX_PATH):
        faiss_index = load_index(_INDEX_PATH)
        registry = get_chunk_registry()
        if faiss_index is not None and faiss_index.ntotal != len(registry):
            all_embs = get_all_embeddings()
            if len(all_embs) > 0 and len(all_embs) == len(registry):
                faiss_index = build_index_from_matrix(all_embs)
                save_index(faiss_index, _INDEX_PATH)
            elif len(all_embs) == 0:
                faiss_index = None
                registry = []
        if faiss_index is not None:
            st.info(f"📂 Loaded existing FAISS index with {faiss_index.ntotal} vectors")
    else:
        st.markdown(
            "<span style='color:#999;font-size:0.85rem;'>○ No index loaded</span>",
            unsafe_allow_html=True,
        )
    st.markdown("---")

file_bytes_dict = (
    {uploaded_file.name: uploaded_file.getvalue() for uploaded_file in uploaded_files}
    if uploaded_files
    else {}
)

with st.spinner("🧠 Processing files and building embeddings…"):
    analysis_results = run_pipeline_with_tracking(
    file_bytes_dict,
    ocr_language,
    ocr_dpi,
    chunk_size,
    chunk_overlap,
)

(
    raw_texts,
    chunked_docs,
    embeddings,
    sim_df,
    chunk_sim_df,
    faiss_index,
    registry,
    ai_probabilities,
) = analysis_results

active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
flags = flag_plagiarism(active_sim_df, threshold=threshold)

st.subheader("📊 Analysis Summary")
st.write(f"Processed **{len(raw_texts)}** documents with Chunk Size: `{chunk_size}` and Overlap: `{chunk_overlap}`.")

selected_class = st.selectbox(
    "Select Class/Section",
    unique_classes,
    index=0,
    key="class_filter_selectbox",
)

st.write(
    f"Processed **{len(raw_texts)}** documents with Chunk Size: `{chunk_size}` and Overlap: `{chunk_overlap}`."
)

st.markdown("---")
st.markdown("""
**How it works**
1. Upload **PDF, DOCX, TXT, or Markdown** assignment files or import from Google Drive
2. Text is extracted according to the file type
3. Text is split into **paragraph chunks**
4. Chunks are embedded with **SentenceTransformers**
5. A **FAISS index** is built over all chunk vectors
6. Pairs above threshold are flagged
""")
st.markdown("---")
st.caption("Semantic Plagiarism Detector · FAISS edition")

if user_role == "admin":
    st.markdown("---")
    st.markdown("### 📁 Document Management")
    existing_docs = get_all_documents()
    if existing_docs:
        st.write(f"**{len(existing_docs)}** documents in database")
        for doc in existing_docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📄 {doc['filename']}")
            with col2:
                if st.button("🗑️", key=f"del_{doc['filename']}"):
                    delete_document(doc["filename"])
                    embeddings_matrix = get_all_embeddings()
                    if embeddings_matrix.size > 0:
                        new_index = build_index_from_matrix(embeddings_matrix)
                        save_index(new_index, _INDEX_PATH)
                    else:
                        if os.path.exists(_INDEX_PATH):
                            os.remove(_INDEX_PATH)
                    st.rerun()

if user_role == "admin":
    st.markdown("---")
    st.markdown("### 📁 Document Management")
    existing_docs = get_all_documents()
    if existing_docs:
        st.write(f"**{len(existing_docs)}** documents in database")
        for doc in existing_docs:
            st.text(doc)

    safe_last_interaction = int(last_interaction or time.time())
    st.markdown(
        f"""
        <div id="session-timer" style="
            background-color: rgba(255, 165, 0, 0.1);
            border: 1px solid rgba(255, 165, 0, 0.3);
            border-radius: 8px;
            padding: 12px;
            margin-top: 16px;
            text-align: center;
            font-family: monospace;
            font-size: 14px;
            color: #ffa500;
        ">
            ⏱️ Session expires in: <span id="timer-display">15:00</span>
        </div>
        <script>
        (function() {{
            const timeoutLimit = {TIMEOUT_LIMIT};
            const lastInteraction = {safe_last_interaction};
            const display = document.getElementById('timer-display');

            function updateTimer() {{
                const now = Math.floor(Date.now() / 1000);
                const elapsed = now - lastInteraction;
                const remaining = Math.max(0, timeoutLimit - elapsed);

                if (remaining <= 0) {{
                    display.textContent = "00:00";
                    display.parentElement.style.borderColor = "#ff4b4b";
                    display.parentElement.style.color = "#ff4b4b";
                    display.parentElement.innerHTML = "⚠️ Session Expired. Reloading...";
                    setTimeout(() => window.location.reload(), 2000);
                    return;
                }}

                const minutes = Math.floor(remaining / 60);
                const seconds = remaining % 60;
                display.textContent = `${{minutes.toString().padStart(2, '0')}}:${{seconds.toString().padStart(2, '0')}}`;

                if (remaining < 60) {{
                    display.parentElement.style.borderColor = "#ff4b4b";
                    display.parentElement.style.color = "#ff4b4b";
                }}
            }}

            updateTimer();
            setInterval(updateTimer, 1000);
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

# Render Upload & Student Portal Section
file_bytes_dict = render_upload_section(user_role, lang_code, _INDEX_PATH)

# Threshold & Chunking Parameters from Session State
threshold = st.session_state.get(SessionKeys.THRESHOLD_SLIDER, PLAGIARISM_THRESHOLD)
use_chunk_matrix = st.session_state.get(SessionKeys.CHUNK_MATRIX_CHECKBOX, False)
faiss_top_k = st.session_state.get(SessionKeys.FAISS_TOP_K_SLIDER, 5)
chunk_size = st.session_state.get(SessionKeys.CHUNK_SIZE_SLIDER, 500)
chunk_overlap = st.session_state.get(SessionKeys.CHUNK_OVERLAP_SLIDER, 50)
ocr_language = st.session_state.get(SessionKeys.OCR_LANGUAGE_SELECTOR, "eng")
ocr_dpi = st.session_state.get(SessionKeys.OCR_DPI_SLIDER, 250)

has_enough_files = len(file_bytes_dict) >= 2

if has_enough_files:
    with st.spinner("🧠 Processing files and building embeddings…"):
        analysis_results = run_pipeline(
            file_bytes_dict,
            ocr_language,
            ocr_dpi,
            chunk_size,
            chunk_overlap,
        )

        st.markdown("---")
        st.markdown("### 📁 Document Management & Bulk Export")
        existing_docs = get_all_documents()
        if existing_docs:
            raw_assignment_titles = sorted(
                list(
                    {
                        (
                            doc.assignment_title
                            if hasattr(doc, "assignment_title")
                            else (doc.get("assignment_title") if isinstance(doc, dict) else None)
                        )
                        for doc in existing_docs
                    }
                    - {None, ""}
                )
            )
            assignment_titles = ["All Assignments"] + raw_assignment_titles
            selected_assignment = st.selectbox(
                "Filter by Assignment",
                options=assignment_titles,
                key="corpus_assignment_filter_selectbox",
            )
            if selected_assignment != "All Assignments":
                existing_docs = [
                    doc
                    for doc in existing_docs
                    if (
                        doc.assignment_title
                        if hasattr(doc, "assignment_title")
                        else (doc.get("assignment_title") if isinstance(doc, dict) else None)
                    )
                    == selected_assignment
                ]

            # ── File Extension Filter Dropdown (Issue #1883) ─────────────────────
            # Extract unique file extensions from the currently filtered documents
            # to dynamically populate the dropdown, while ensuring standard types
            # are always available for user convenience.
            from pathlib import Path
            
            # Define standard extensions that should always appear in the filter
            standard_extensions = ["All", ".pdf", ".docx", ".txt", ".csv", ".md"]
            
            # Extract extensions from the current document list
            available_extensions = set()
            for doc in existing_docs:
                fn = (
                    doc.filename
                    if hasattr(doc, "filename")
                    else (doc.get("filename") if isinstance(doc, dict) else str(doc))
                )
                ext = Path(fn).suffix.lower()
                if ext:
                    available_extensions.add(ext)
            
            # Combine standard and available extensions, maintaining order
            filter_options = ["All"]
            for ext in standard_extensions[1:]:
                if ext in available_extensions or ext in [".pdf", ".docx", ".txt", ".csv", ".md"]:
                    filter_options.append(ext)
            
            # Add any non-standard extensions found in the database
            for ext in sorted(list(available_extensions)):
                if ext not in filter_options:
                    filter_options.append(ext)
            
            # Render the selectbox with a unique key to prevent state collisions
            selected_file_type = st.selectbox(
                "Filter by File Type",
                options=filter_options,
                index=0,
                key="corpus_file_type_filter_selectbox",
                help="Filter the document corpus table to show only specific file extensions.",
            )
            
            # Apply the file extension filter to the existing_docs list
            if selected_file_type != "All":
                existing_docs = [
                    doc for doc in existing_docs
                    if Path(
                        doc.filename if hasattr(doc, "filename") 
                        else (doc.get("filename") if isinstance(doc, dict) else str(doc))
                    ).suffix.lower() == selected_file_type
                ]
            
            # ── End File Extension Filter (Issue #1883) ──────────────────────────

            # Update the document count display to reflect BOTH assignment and file type filters
            st.write(
                f"**{len(existing_docs)}** documents in database"
                + (f" (Filtered by: {selected_file_type})" if selected_file_type != "All" else "")
                + (f" | Assignment: {selected_assignment}" if selected_assignment != "All Assignments" else "")
            )

            import pandas as pd

            from src.db.corpus_db import (
                get_document_char_counts,
                get_document_word_counts,
            )

            word_counts = get_document_word_counts()
            char_counts = get_document_char_counts()

            doc_rows = []
            for doc in existing_docs:
                fn = (
                    doc.filename
                    if hasattr(doc, "filename")
                    else (doc.get("filename") if isinstance(doc, dict) else str(doc))
                )
                doc_rows.append(
                    {
                        "Select": False,
                        "Format": format_extension_badge(fn),
                        "Filename": fn,
                        "Word Count": word_counts.get(fn, 0),
                        "Char Count": char_counts.get(fn, 0),
                    }
                )

            corpus_df = pd.DataFrame(doc_rows)

            sel_col1, sel_col2 = st.columns(2)
            with sel_col1:
                if st.button(
                    "☑️ Select All",
                    key="sidebar_select_all_corpus_btn",
                    use_container_width=True,
                ):
                    st.session_state["corpus_select_all_toggle"] = True
                    st.rerun()
            with sel_col2:
                if st.button(
                    "⬜ Clear",
                    key="sidebar_clear_corpus_btn",
                    use_container_width=True,
                ):
                    st.session_state["corpus_select_all_toggle"] = False
                    st.rerun()

            if st.session_state.get("corpus_select_all_toggle", False):
                corpus_df["Select"] = True

            edited_df = st.data_editor(
                corpus_df,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select",
                        default=False,
                        help="Select for bulk ZIP export",
                    ),
                    "Format": st.column_config.TextColumn("Format", disabled=True, width="small"),
                    "Filename": st.column_config.TextColumn("Filename", disabled=True),
                    "Word Count": st.column_config.NumberColumn(
                        "Word Count", format="%d words", disabled=True
                    ),
                    "Char Count": st.column_config.NumberColumn(
                        "Char Count", format="%d chars", disabled=True
                    ),
                },
                disabled=["Format", "Filename", "Word Count", "Char Count"],
                hide_index=True,
                key="sidebar_corpus_data_editor",
                use_container_width=True,
            )

            selected_rows = edited_df[edited_df["Select"]]
            selected_filenames = selected_rows["Filename"].tolist()

            if selected_filenames:
                zip_data = create_documents_bulk_zip_archive(selected_filenames)
                st.download_button(
                    label=f"📦 Export Selected as ZIP ({len(selected_filenames)})",
                    data=zip_data,
                    file_name=f"corpus_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    key="sidebar_export_selected_zip_btn",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.button(
                    "📦 Export Selected as ZIP (0)",
                    disabled=True,
                    key="sidebar_export_zip_disabled_btn",
                    use_container_width=True,
                )

            st.markdown("---")
            for doc in existing_docs:
                fn = (
                    doc.filename
                    if hasattr(doc, "filename")
                    else (doc.get("filename") if isinstance(doc, dict) else str(doc))
                )
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"{format_extension_badge(fn)} {fn}")
                with col2:
                    if st.button("🗑️", key=f"del_{fn}"):
                        delete_document(fn)
                        embeddings_matrix = get_all_embeddings()
                        if embeddings_matrix.size > 0:
                            new_index = build_index_from_matrix(embeddings_matrix)
                            save_index(new_index, _INDEX_PATH)
                        else:
                            if os.path.exists(_INDEX_PATH):
                                os.remove(_INDEX_PATH)
                        st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "🗑️ Clear All Documents",
            key="clear_all_documents_button",
            use_container_width=True,
        ):
            clear_all_dialog()  # type: ignore
        st.markdown("<br>", unsafe_allow_html=True)


st.markdown("---")
if st.button("🚪 Log Out", use_container_width=True, key="logout_button"):
    for key in ["authenticated", "username", "role", "last_interaction"]:
        if key in st.session_state:
            del st.session_state[key]
    clear_session(SESSION_ID)
    st.rerun()


# ── Onboarding Tour for First-Time Admin Users ───────────────────────────────────
# Onboarding Tour
if (
    Tour is not None
    and user_role == "admin"
    and not get_tour_completed(st.session_state[SessionKeys.USERNAME])
):
    username = st.session_state[SessionKeys.USERNAME]
    if st.button("🎯 Start Guided Tour", key="start_tour_button", type="primary"):
        st.session_state[SessionKeys.SHOW_TOUR] = True

    if st.session_state.get(SessionKeys.SHOW_TOUR, False):
        tour_steps = [
            Tour.info(
                title="👋 Welcome to the Plagiarism Detection System!",
                desc="This guided tour will walk you through the key features to help you get started.",
            ),
            Tour.bind(
                SessionKeys.THRESHOLD_SLIDER,
                title="⚙️ Plagiarism Threshold",
                desc=f"Adjust the flagging threshold. Default is {DEFAULT_THRESHOLDS.plagiarism:.0%}.",
                side="right",
            ),
            Tour.bind(
                SessionKeys.CLASS_FILTER_SELECTBOX,
                title="🔍 Class Filter",
                desc="Filter analysis results by specific class sections.",
                side="right",
            ),
            Tour.info(
                title="📊 Analysis Dashboard",
                desc="View similarity metrics, flagged pairs, and comparisons in the tabs below.",
            ),
        ]
        tour = Tour(steps=tour_steps)
        tour.start()
        if st.button("✅ Finish Tour", use_container_width=True):
            set_tour_completed(username, True)
            st.session_state[SessionKeys.SHOW_TOUR] = False
            st.success("✅ Onboarding tour completed!")
            st.rerun()

st.title(get_text("title", lang=lang_code))
st.markdown(get_text("subtitle", lang=lang_code))
st.divider()

if user_role == "admin":
    initialize_advanced_features()
    cached_index_data = get_faiss_index("corpus_index")

    if cached_index_data is not None and os.path.exists(_INDEX_PATH):
        try:
            import faiss

            index_buffer = _io.BytesIO(cached_index_data)
            faiss_index = faiss.deserialize_index(faiss.read_index(index_buffer))
            registry = get_chunk_registry()
            st.info(
                f"📂 Loaded FAISS index from Redis cache with {faiss_index.ntotal} vectors"
            )
        except Exception as e:
            print(f"[Redis] Error loading cached index: {e}, falling back to disk")
            from src.core.faiss_index import load_or_rebuild_index

            faiss_index, registry, index_recovered = load_or_rebuild_index(_INDEX_PATH)

            if index_recovered:
                if faiss_index.ntotal:
                    st.warning(
                        "FAISS index was missing, corrupted, or inconsistent and was "
                        f"automatically rebuilt from {faiss_index.ntotal} stored vectors."
                    )
                else:
                    st.info(
                        "No stored embeddings were found. An empty FAISS index was "
                        "initialized safely."
                    )
            else:
                st.info(f"Loaded and validated the existing FAISS index with {faiss_index.ntotal} vectors.")

    from src.utils.redis_cache import store_large_data, get_large_data, clear_large_data
    from src.utils.similarity_cache import build_similarity_cache_key

    analysis_cache_key = build_similarity_cache_key(SESSION_ID, use_hybrid=use_hybrid)
    analysis_metadata_key = f"{analysis_cache_key}_metadata"

    if SessionKeys.ANALYSIS_RESULTS not in st.session_state:
        # Store only metadata in session state
        cached_metadata = get_large_data(analysis_metadata_key)
        if cached_metadata is not None:
            st.session_state[SessionKeys.ANALYSIS_RESULTS] = cached_metadata
        else:
            st.session_state[SessionKeys.ANALYSIS_RESULTS] = {
                "has_results": False,
                "doc_count": 0,
                "timestamp": time.time(),
                "cache_key": None
            }
    
    # Check if we have cached results
    cached_results = get_large_data(analysis_cache_key)
    if cached_results is not None:
        st.session_state[SessionKeys.ANALYSIS_RESULTS]["has_results"] = True
        st.session_state[SessionKeys.ANALYSIS_RESULTS]["doc_count"] = cached_results.get("doc_count", 0)
        st.session_state[SessionKeys.ANALYSIS_RESULTS]["cache_key"] = analysis_cache_key

    if SessionKeys.ANALYSIS_FILE_SIGNATURE not in st.session_state:
        st.session_state[SessionKeys.ANALYSIS_FILE_SIGNATURE] = None

        cached_signature = get_session_state(SESSION_ID, "analysis_file_signature")
        if cached_signature is not None:
            st.session_state[SessionKeys.ANALYSIS_FILE_SIGNATURE] = cached_signature
            faiss_index = (
                load_index(_INDEX_PATH) if os.path.exists(_INDEX_PATH) else None
            )
            registry = get_chunk_registry()
    else:
        faiss_index = load_index(_INDEX_PATH) if os.path.exists(_INDEX_PATH) else None

    uploaded_files = st.file_uploader(
        get_text("upload_title", lang=lang_code),
        type=["pdf", "docx", "txt", "md", "markdown", "mdown", "zip", "csv"],
        accept_multiple_files=True,
        key="file_uploader",
        on_change=file_uploader_callback,
    )

    if st.session_state.get("staged_files_count", 0) > 0:
        staged_count = st.session_state["staged_files_count"]
        staged_size_mb = st.session_state["staged_files_size"] / (1024 * 1024)
        st.info(f"📁 Staged {staged_count} files (Total Size: {staged_size_mb:.1f} MB)")

    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB limit
    file_bytes_dict = {}

    if bulk_download_drive_folder is not None:
        with st.expander("📁 Import from Google Drive", expanded=False):
            drive_folder_input = st.text_input(
                "Google Drive folder URL or ID",
                key="drive_folder_input",
                placeholder="https://drive.google.com/drive/folders/…",
            )
            drive_api_key = st.text_input(
                "Google Drive API key",
                key="drive_api_key",
                type="password",
                help="Optional if GOOGLE_DRIVE_API_KEY is set in the environment.",
            )
            if st.button("Import from Drive", key="drive_import_btn"):
                if not drive_folder_input:
                    st.error("Please enter a Google Drive folder URL or ID.")
                else:
                    drive_progress_bar = st.progress(
                        0, text="Connecting to Google Drive…"
                    )

                    def _update_drive_progress(bytes_downloaded, total_bytes):
                        fraction = (
                            min(bytes_downloaded / total_bytes, 1.0)
                            if total_bytes
                            else 0
                        )
                        drive_progress_bar.progress(
                            fraction,
                            text=(
                                f"Downloading from Drive… "
                                f"{bytes_downloaded / 1024:.0f} KB"
                                + (
                                    f" / {total_bytes / 1024:.0f} KB"
                                    if total_bytes
                                    else ""
                                )
                            ),
                        )

                    try:
                        drive_files, drive_names = bulk_download_drive_folder(
                            drive_folder_input,
                            api_key=drive_api_key or None,
                            progress_callback=_update_drive_progress,
                        )
                        st.session_state.setdefault("drive_imported_files", {})
                        st.session_state["drive_imported_files"].update(drive_files)
                        drive_progress_bar.progress(
                            1.0, text=f"Imported {len(drive_names)} file(s)."
                        )
                        st.success(
                            f"Imported {len(drive_names)} file(s) from Google Drive: "
                            f"{', '.join(drive_names)}"
                        )
                    except Exception as exc:
                        drive_progress_bar.empty()
                        st.error(f"⚠️ Google Drive import failed: {exc}")

    if uploaded_files:
        for uploaded_file in uploaded_files:
            original_name = uploaded_file.name
            try:
                validate_document_extension(
                    original_name,
                    allowed_extensions={
                        ".csv",
                        ".docx",
                        ".md",
                        ".markdown",
                        ".mdown",
                        ".pdf",
                        ".txt",
                        ".zip",
                    },
                )
            except InvalidFileExtensionError as exc:
                st.error(
                    f"⚠️ File **'{sanitize_filename(original_name)}'** was rejected: {exc}"
                )
                continue

            safe_name = unique_filename(original_name, file_bytes_dict)

            if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                st.error(
                    f"⚠️ File **'{safe_name}'** exceeds maximum size limit of 10MB."
                )
                continue

            file_bytes = uploaded_file.read()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            existing_doc = get_document_by_hash(file_hash)

            if existing_doc:
                st.warning(f"⚠️ File **'{original_name}'** is identical to **'{existing_doc}'** already in the database.")
                action = st.radio(
                    f"Action for duplicate file '{original_name}':",
                    ["Skip", "Reprocess"],
                    key=f"dup_{file_hash}_{original_name}",
                    horizontal=True
                )
                if action == "Skip":
                    continue

            file_bytes_dict[safe_name] = strip_exif_metadata(
                file_bytes, safe_name
            )

    for drive_name, drive_bytes in st.session_state.get(
        "drive_imported_files", {}
    ).items():
        safe_drive_name = unique_filename(drive_name, file_bytes_dict)
        file_bytes_dict[safe_drive_name] = drive_bytes

    has_enough_files = len(file_bytes_dict) >= 2

    @st.cache_data(show_spinner=False)
    def run_extraction_pipeline(
        raw_texts_items: tuple,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        raw_texts_dict = dict(raw_texts_items)
        chunked_docs = chunk_documents(
            raw_texts_dict, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        translated_chunked_docs = {}

        for doc_name, chunks in chunked_docs.items():
            translated_chunked_docs[doc_name] = []
            for chunk in chunks:
                prepared = prepare_text_for_embedding(chunk)
                translated_chunked_docs[doc_name].append(prepared["embedding_text"])

        embeddings = embed_documents(translated_chunked_docs)
        sim_df = document_similarity_matrix(embeddings)

        names = list(embeddings.keys())
        n = len(names)
        chunk_mat = np.zeros((n, n))

        for i, na in enumerate(names):
            for j, nb in enumerate(names):
                if i == j:
                    chunk_mat[i, j] = 1.0
                elif j > i:
                    ea, eb = embeddings[na], embeddings[nb]
                    score = (
                        float(np.max(cosine_similarity(ea, eb)))
                        if ea.size and eb.size
                        else 0.0
                    )
                    chunk_mat[i, j] = score
                    chunk_mat[j, i] = score

        chunk_sim_df = pd.DataFrame(chunk_mat, index=names, columns=names)
        faiss_index, registry = build_index(embeddings, chunked_docs)
        ai_probabilities = detect_documents_ai_probability(chunked_docs)

        # ========== ADD AI DETECTION HERE ==========
        if enable_ai_detection:
            from src.core.ai_detector_enhanced import detect_ai_probability_enhanced
            ai_probabilities = detect_ai_probability_enhanced(
                chunked_docs if chunked_docs else {},
                threshold=ai_threshold
            )
        else:
            ai_probabilities = {}
        # ==========================================

        return (
            chunked_docs,
            embeddings,
            sim_df,
            chunk_sim_df,
            faiss_index,
            registry,
            ai_probabilities,
        )

        

    if has_enough_files:
        st.session_state[SessionKeys.SCANNING] = True
        total_bytes = sum(len(data) for data in file_bytes_dict.values())
        file_count = len(file_bytes_dict)

        progress_bar = st.progress(0, text="Preparing files…")
        raw_texts = {}
        failed_documents = []
        
        for i, (name, data) in enumerate(file_bytes_dict.items()):
            try:
                extracted = extract_text(
                    _io.BytesIO(data), name, ocr_language=ocr_language, ocr_dpi=ocr_dpi
                )
                raw_texts[name] = extracted
                
            except EmptyDocumentError as ede:
                # Issue #2724: Catch empty documents and log them as warnings
                # instead of crashing the entire analysis pipeline
                logger.warning("Skipping empty document %s: %s", name, ede)
                failed_documents.append({
                    "filename": name,
                    "error": str(ede),
                    "type": "empty_document"
                })
                st.warning(f"⚠️ **{name}**: {ede}")
                
            except Exception as e:
                # Catch other extraction errors
                logger.error("Failed to extract text from %s: %s", name, e)
                failed_documents.append({
                    "filename": name,
                    "error": str(e),
                    "type": "extraction_error"
                })
                st.error(f"❌ **{name}**: Failed to extract text. {e}")
                
            fraction = (i + 1) / file_count
            remaining_bytes = total_bytes * (file_count - i - 1) // max(1, file_count)
            remaining_est = estimate_processing_seconds(remaining_bytes)
            eta = (
                format_processing_duration(remaining_est)
                if remaining_est
                else "a moment"
            )
            progress_bar.progress(
                fraction,
                text=f"Processing file {i + 1} of {file_count} (ETA: {eta})",
            )

        # Filter out failed documents from further processing
        if failed_documents:
            st.session_state["failed_documents"] = failed_documents
            st.info(f"Skipped {len(failed_documents)} file(s) due to extraction errors.")
            
        # Only proceed if we have enough valid texts
        if len(raw_texts) < 2:
            st.error("Not enough valid documents remaining for comparison after filtering errors.")
            st.session_state[SessionKeys.SCANNING] = False
            progress_bar.empty()
            st.stop()

        raw_texts_tuple = tuple(sorted(raw_texts.items()))
        (
            chunked_docs,
            embeddings,
            sim_df,
            chunk_sim_df,
            faiss_index,
            registry,
            ai_probabilities,
        ) = run_extraction_pipeline(
            raw_texts_tuple,
            chunk_size,
            chunk_overlap,
        )
        
        # ========== ADD THIS BLOCK ==========
        from src.utils.redis_cache import store_large_data
        
        # Store large results in Redis with compression
        large_results = {
            "chunked_docs": chunked_docs,
            "embeddings": embeddings,
            "sim_df": sim_df,
            "chunk_sim_df": chunk_sim_df,
            "faiss_index": faiss_index,
            "registry": registry,
            "ai_probabilities": ai_probabilities,
            "doc_count": len(chunked_docs),
            "timestamp": time.time()
        }
        
        store_large_data(analysis_cache_key, large_results, ttl=1800)
        
        # Update session state with metadata only
        st.session_state[SessionKeys.ANALYSIS_RESULTS] = {
            "has_results": True,
            "doc_count": len(chunked_docs),
            "timestamp": time.time(),
            "cache_key": analysis_cache_key
        }
        
        store_large_data(analysis_metadata_key, {
            "has_results": True,
            "doc_count": len(chunked_docs),
            "timestamp": time.time(),
            "cache_key": analysis_cache_key
        }, ttl=1800)
        # ===================================
        
        st.session_state[SessionKeys.SCANNING] = False
        active_sim_df = chunk_sim_df if use_chunk_matrix else sim_df
        flags = flag_plagiarism(active_sim_df, threshold=threshold)

        init_incident_db()
        incidents = sync_flagged_incidents(flags)
    else:
        flags = []
        active_sim_df = None
        raw_texts = {}
        ai_probabilities = {}

st.subheader(get_text("analysis_summary", lang=lang_code))
doc_names = list(raw_texts.keys()) if raw_texts else []
n_docs = len(doc_names)
total_pairs = n_docs * (n_docs - 1) // 2 if n_docs > 1 else 0
n_flagged = len(flags)
total_doc_count = max(n_docs, get_total_document_count())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Documents", total_doc_count)
col2.metric("Pairs Evaluated", total_pairs)
col3.metric("Flagged Pairs", n_flagged)
col4.metric("FAISS Vectors", faiss_index.ntotal if faiss_index is not None else 0)
col5.metric("🎯 Threshold", f"{threshold:.0%}")

# ── Issue #1728: File Parsing Duration Summary ──────────────
_parse_durations = get_all_parse_durations()
if _parse_durations and raw_texts:
    with st.expander("⏱️ File Parsing Times", expanded=False):
        for _doc_name in doc_names:
            _dur = _parse_durations.get(_doc_name)
            _dur_str = format_duration(_dur)
            if _dur_str:
                st.markdown(
                    f"**{_doc_name}** — Parsed in `{_dur_str}`"
                )

st.divider()

# Main Application Tabs
(
    tab_warnings,
    tab_faiss,
    tab_matrix,
    tab_heatmap,
    tab_drill,
    tab_compare,
    tab_analytics,
    tab_patterns,
    tab_users,
    tab_settings,
    tab_history,
    tab_audit,
) = st.tabs(
    [
        get_text("tab_warnings", lang=lang_code),
        get_text("tab_faiss", lang=lang_code),
        get_text("tab_matrix", lang=lang_code),
        get_text("tab_heatmap", lang=lang_code),
        get_text("tab_drill", lang=lang_code),
        "🔬 Comparison",
        get_text("tab_analytics", lang=lang_code),
        "🧠 Patterns",
        get_text("tab_users", lang=lang_code),
        get_text("tab_settings", lang=lang_code),
        "📊 History",
        get_text("tab_audit_logs", lang=lang_code),
    ],
    key="main_tabs",
)

# Record scan summary for historical tracking (Issue #1672)
if flags and len(file_bytes_dict) >= 2:
    from src.db.corpus_db import record_scan_summary

    all_sims = [f["similarity"] for f in flags]
    avg_sim = sum(all_sims) / len(all_sims) if all_sims else 0.0
    max_sim = max(all_sims) if all_sims else 0.0

    record_scan_summary(
        document_count=len(file_bytes_dict),
        avg_similarity=avg_sim,
        max_similarity=max_sim,
        flagged_count=len(flags),
        threshold_used=threshold,
    )

# Render View Components into Tabs
with tab_warnings:
    update_page_title("Warnings")
    st.subheader(get_text("tab_warnings", lang=lang_code))

    auto_refresh_enabled = st.toggle(
        "Auto-refresh live feed (30s)",
        value=False,
        key=SessionKeys.INCIDENT_STREAM_AUTO_REFRESH,
        help=(
            "When enabled, the incident feed re-runs every 30 seconds "
            "to surface newly flagged submissions automatically."
        ),
    )

    if auto_refresh_enabled and st_autorefresh is not None:
        st_autorefresh(
            interval=30 * 1000,
            key="incident_stream_autorefresh",
        )

    st.session_state[SessionKeys.INCIDENT_STREAM_AUTO_REFRESH] = auto_refresh_enabled

    if auto_refresh_enabled:
        if st_autorefresh is None:
            st.warning(
                "Auto-refresh is enabled, but the `streamlit-autorefresh` "
                "package is not installed. Install it via "
                "`pip install streamlit-autorefresh`."
            )
        else:
            st.caption("🔴 Live — refreshing every 30 seconds.")
    else:
        st.caption("⚪ Live feed paused — toggle on to auto-refresh.")

    # ── AI Detection Results ──────────────────────────────────────────────────
    if enable_ai_detection and ai_probabilities:
        st.markdown("### 🤖 AI-Generated Text Detection")
        
        # Summary metrics
        ai_col1, ai_col2, ai_col3 = st.columns(3)
        
        suspicious_count = sum(1 for p in ai_probabilities.values() if p > ai_threshold)
        avg_ai_prob = sum(ai_probabilities.values()) / len(ai_probabilities) if ai_probabilities else 0
        
        with ai_col1:
            st.metric("Documents Analyzed", len(ai_probabilities))
        with ai_col2:
            st.metric("Suspicious (AI)", suspicious_count)
        with ai_col3:
            st.metric("Avg AI Probability", f"{avg_ai_prob * 100:.1f}%")
        
        # Display AI probabilities for each document
        if ai_probabilities:
            import pandas as pd
            ai_df = pd.DataFrame([
                {"Document": doc, "AI Probability": f"{prob * 100:.1f}%", "Status": "⚠️ AI" if prob > ai_threshold else "✅ Human"}
                for doc, prob in ai_probabilities.items()
            ])
            st.dataframe(ai_df, use_container_width=True, hide_index=True)
            
            # Highlight suspicious documents
            for doc, prob in ai_probabilities.items():
                if prob > ai_threshold:
                    st.warning(f"⚠️ **{doc}**: {prob * 100:.1f}% AI probability")

    st.divider()

    if SessionKeys.WARNINGS_EXPAND_ALL not in st.session_state:
        st.session_state[SessionKeys.WARNINGS_EXPAND_ALL] = False

    st.markdown("### 📅 Incident Date Filter")
    date_preset = st.radio(
        "Select Date Range",
        options=["Today", "Last 7 Days", "Last 14 Days", "Last 30 Days", "All Time"],
        horizontal=True,
        key="incident_date_preset",
        help="Quickly filter the incident table by common date ranges.",
    )

    start_date, end_date = get_date_range_preset(date_preset)
    st.caption(
        f"Filtering incidents from **{start_date.strftime('%Y-%m-%d')}** to "
        f"**{end_date.strftime('%Y-%m-%d')}**"
    )

    if not flags:
        st.info("No plagiarism incidents detected above configured threshold.")
    elif render_warning_controls is not None:
        if "warning_page" not in st.session_state:
            st.session_state.warning_page = reset_warning_page()

        def _set_warning_page(page: int) -> None:
            st.session_state.warning_page = page

        render_warning_controls(
            flags,
            threshold=threshold,
            ai_probabilities=ai_probabilities,
            set_warning_page=_set_warning_page,
        )

        button_label = (
            "📂 Expand All"
            if not st.session_state[SessionKeys.WARNINGS_EXPAND_ALL]
            else "📁 Collapse All"
        )

        if st.button(button_label, key="toggle_warning_accordions"):
            st.session_state[SessionKeys.WARNINGS_EXPAND_ALL] = not st.session_state[
                SessionKeys.WARNINGS_EXPAND_ALL
            ]
            st.rerun()

        faiss_query = st.text_input(
            "Query FAISS Index:",
            placeholder="Type a text snippet to search vector index...",
            key="faiss_query_input",
        )
        if st.button("🔍 Run FAISS Search", key="run_faiss_search_btn"):
            if faiss_query.strip() and faiss_index is not None:
                from src.core.embedding_model import embed_chunks

                q_vec = embed_chunks([faiss_query.strip()])[0]
                q_results = search_similar_chunks(
                    q_vec,
                    faiss_index,
                    registry,
                    top_k=faiss_top_k,
                    threshold=threshold,
                )
                if q_results:
                    rows = []
                    for rec, score in q_results:
                        truncated_name = truncate_filename(rec.doc_name)
                        rows.append({
                            "Document": truncated_name,
                            "Chunk Index": rec.chunk_index,
                            "Similarity": f"{score:.1%}",
                            "Content Preview": rec.chunk_text[:120] + "..." if len(rec.chunk_text) > 120 else rec.chunk_text
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No matching vector chunks found above threshold.")


# ══ TAB 2: FAISS ══════════════════════════════════════════════════════════
with tab_faiss:
    update_page_title("FAISS")
    st.subheader("⚡ FAISS Vector Search")
    if faiss_index is not None:
        st.info(f"Index total: {faiss_index.ntotal} vectors.")
        faiss_query = st.text_input("Query FAISS Index:", key="faiss_query_input_tab2")
        if st.button("Run Search", key="run_search_tab2") and faiss_query.strip():
            q_vec = embed_chunks([faiss_query.strip()])[0]
            results = search_similar_chunks(
                q_vec, faiss_index, registry, top_k=faiss_top_k, threshold=threshold
            )
            from app.components.faiss_results import render_faiss_results_ui

with tab_matrix:
    update_page_title("Matrix")
    render_matrix_view(active_sim_df)

with tab_heatmap:
    update_page_title("Heatmap")
    st.subheader("🗺️ Heatmap & Network")
    heatmap_fig = None
    if active_sim_df is not None:
        heatmap_fig = ui_exception_handler("Similarity Heatmap")(
            plot_similarity_heatmap
        )(active_sim_df, threshold=threshold, theme_colors=get_chart_colors())

    if heatmap_fig is not None:
        st.pyplot(heatmap_fig, use_container_width=True)

    doc_select_options = (
        ["None"] + list(active_sim_df.columns)
        if active_sim_df is not None
        else ["None"]
    )
    selected_highlight_doc = st.selectbox(
        "Highlight Document Node",
        options=doc_select_options,
        index=0,
        key="highlight_doc_node_selector",
    )
    highlighted_doc = (
        selected_highlight_doc if selected_highlight_doc != "None" else None
    )

    network_fig = None
    if active_sim_df is not None:
        network_fig = ui_exception_handler("Plagiarism Network")(
            plot_similarity_network
        )(
            similarity_df=active_sim_df,
            threshold=threshold,
            highlighted_doc=highlighted_doc,
            title="Interactive Document Plagiarism Network",
        )

    if network_fig is not None:
        st.plotly_chart(network_fig, use_container_width=True)

    # ── Plagiarism Cluster Detection Summary (Issue #1675) ───────────────────
    if active_sim_df is not None and len(doc_names) >= 2:
        from src.core.similarity import detect_plagiarism_clusters

        cluster_data = detect_plagiarism_clusters(active_sim_df, threshold=threshold)
        suspicious_groups = cluster_data["suspicious_groups"]

        if suspicious_groups:
            with st.expander(
                f"🚨 Suspicious Collusion Rings Detected ({len(suspicious_groups)})",
                expanded=True,
            ):
                st.warning(
                    f"Found {len(suspicious_groups)} group(s) of 3+ highly similar documents. "
                    "These may indicate collusion or shared source material."
                )

                for group in suspicious_groups:
                    st.markdown(f"**Cluster #{group['cluster_id']}** ({group['size']} documents):")
                    for doc in group["documents"]:
                        st.markdown(f"- 📄 `{doc}`")
                    st.divider()


# ══ TAB 5: PAIR DRILL-DOWN ════════════════════════════════════════════════
with tab_drill:
    update_page_title("Drill Down")
    render_drilldown_view(active_sim_df, raw_texts, flags, doc_names)

    st.markdown("---")

    if active_sim_df is not None and len(doc_names) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            da = st.selectbox("Document A", doc_names, key="da")
        with c2:
            db = st.selectbox("Document B", [d for d in doc_names if d != da], key="db")
        sim_val = float(active_sim_df.loc[da, db])
        st.write(f"Overall Similarity: `{sim_val:.1%}`")

        pair_flags = [
            f
            for f in flags
            if (f["doc_a"] == da and f["doc_b"] == db)
            or (f["doc_a"] == db and f["doc_b"] == da)
        ]

        if pair_flags:
            st.markdown("### 📝 Flagged Snippets")
            for rank, flag in enumerate(pair_flags, 1):
                ca = str(flag.get("snippet_a", ""))
                cb = str(flag.get("snippet_b", ""))

                if flag["doc_a"] == db:
                    ca, cb = cb, ca

                highlighted_ca, highlighted_cb = highlight_overlap(ca, cb)

                with st.expander(
                    f"Incident #{rank} - Similarity: {flag.get('similarity', 0.0):.1%}",
                    expanded=(rank == 1),
                ):
                    # Display Translation Match badge if applicable (Issue #1956)
                    if st.session_state.get("cross_lingual_mode_toggle", False):
                        st.markdown(
                            '<span style="background-color: #3B82F6; color: white; '
                            'padding: 2px 8px; border-radius: 4px; font-size: 0.8em;">'
                            '🌐 Translation Match</span>',
                            unsafe_allow_html=True
                        )

                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.markdown(f"**{da}**")
                        st.markdown(highlighted_ca, unsafe_allow_html=True)
                        if render_copy_button:
                            render_copy_button(
                                text_to_copy=ca,
                                button_id=f"copy_ca_{rank}",
                                copy_label="📋 Copy Snippet",
                            )
                    with c_b:
                        st.markdown(f"**{db}**")
                        st.markdown(highlighted_cb, unsafe_allow_html=True)
                        if render_copy_button:
                            render_copy_button(
                                text_to_copy=cb,
                                button_id=f"copy_cb_{rank}",
                                copy_label="📋 Copy Snippet",
                            )

        # ── Semantic Diff Viewer (Issue #1957) ────────────────────────────────────
        if pair_flags and len(doc_names) >= 2:
            with st.expander("🔬 Semantic Diff Viewer (Paraphrase Detection)", expanded=False):
                st.caption(
                    "This viewer uses Dynamic Programming on sentence embeddings to align "
                    "sequences and detect structural reordering or synonym swapping that "
                    "standard lexical diffs miss."
                )

                # Resolve chunks and embeddings for the selected document pair
                try:
                    import streamlit.components.v1 as components

                    from src.core.semantic_alignment import align_semantic_sequences
                    from src.utils.diff_renderer import render_semantic_diff_html

                    # Extract chunks for the selected pair from chunked_docs
                    chunks_a = chunked_docs.get(da, [])
                    chunks_b = chunked_docs.get(db, [])

                    # Get embeddings for the selected pair
                    emb_a = embeddings.get(da, np.array([]))
                    emb_b = embeddings.get(db, np.array([]))

                    if not chunks_a or not chunks_b:
                        st.warning(
                            "Cannot generate semantic diff: chunk data is unavailable "
                            "for the selected document pair. Try re-running the analysis."
                        )
                    elif emb_a.size == 0 or emb_b.size == 0:
                        st.warning(
                            "Cannot generate semantic diff: embedding vectors are unavailable "
                            "for the selected document pair."
                        )
                    else:
                        # Get current theme for rendering
                        current_theme = get_theme_name()

                        # Compute alignment using DP on sentence embeddings
                        alignment_map = align_semantic_sequences(
                            chunks_a=chunks_a,
                            chunks_b=chunks_b,
                            embeddings_a=emb_a,
                            embeddings_b=emb_b,
                        )

                        # Render HTML diff
                        diff_html = render_semantic_diff_html(
                            alignment_map=alignment_map,
                            theme=current_theme
                        )

                        # Inject into Streamlit
                        components.html(diff_html, height=600, scrolling=True)

                        # Export button
                        st.download_button(
                            label="⬇️ Download Diff Report (HTML)",
                            data=diff_html,
                            file_name=f"semantic_diff_{da}_vs_{db}.html",
                            mime="text/html",
                            key=f"download_diff_{da}_{db}",
                        )
                except Exception as diff_err:
                    st.error(f"Failed to generate semantic diff: {diff_err}")
        # ── Bibliography Citation Graph (Issue #1958) ─────────────────────────────
        # Integrated cleanly without conflicting with separate Semantic Diff PR
        with st.expander("📚 Bibliography Analysis (Citation Lifting)", expanded=False):
            st.caption(
                "Detects structural plagiarism by comparing the reference sections. "
                "High overlap indicates students may have copied bibliographies without reading the sources."
            )

            try:
                from src.db.citation_db import (
                    compute_citation_jaccard,
                    get_shared_citations,
                )
                from src.visualization.citation_graph import plot_citation_network

                jaccard_score = compute_citation_jaccard(da, db)
                shared_cites = get_shared_citations(da, db)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Citation Jaccard Similarity", f"{jaccard_score:.1%}")
                with col2:
                    st.metric("Shared References", len(shared_cites))

                if jaccard_score > 0.80:
                    st.error(
                        f"🚨 **High Citation Overlap Detected ({jaccard_score:.1%})**: "
                        "These documents share an unusually high number of identical references."
                    )

                if shared_cites:
                    current_theme = get_theme_name()
                    fig = plot_citation_network(
                        doc_a=da,
                        doc_b=db,
                        shared_citations=shared_cites,
                        theme_colors=get_chart_colors()
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("View Shared Citations List"):
                        st.dataframe(
                            pd.DataFrame(shared_cites)[["author", "year", "title"]],
                            use_container_width=True,
                            hide_index=True
                        )
                else:
                    st.info("No shared citations found between these two documents.")

            except Exception as cite_err:
                st.warning(f"Bibliography analysis unavailable: {cite_err}")


# ══ TAB 6: COMPARISON ══════════════════════════════════════════════════════
with tab_compare:
    update_page_title("Comparison")
    from app.components.document_comparison import render_document_comparison
    render_document_comparison()

with tab_analytics:
    update_page_title("Analytics")
    st.subheader("📊 Analytics Dashboard")
    # ── Background Document Clustering UI (Issue #2811) ─────────────────────
    import requests as _requests
    import time as _time

    st.markdown("---")
    st.subheader("🧩 Background Document Clustering")
    st.caption("Clustering is offloaded to a background FastAPI task to prevent UI timeouts (Issue #2811).")

    bg_col1, bg_col2 = st.columns(2)
    with bg_col1:
        bg_n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="bg_n_clusters_2811")
    with bg_col2:
        bg_cluster_method = st.selectbox("Clustering Method", ["kmeans", "agglomerative"], key="bg_cluster_method_2811")

    bg_vectors = [[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8], [0.5, 0.5]]

    if st.button("🚀 Run Background Clustering", key="run_bg_clustering_2811"):
        payload = {"vectors": bg_vectors, "n_clusters": bg_n_clusters, "method": bg_cluster_method}
        try:
            response = _requests.post("http://localhost:8000/api/clustering/", json=payload)
            if response.status_code == 200:
                task_id = response.json()["task_id"]
                st.info(f"Task started! Task ID: `{task_id}`")
                progress_text = st.empty()
                progress_bar = st.progress(0)
                for i in range(60):
                    try:
                        status_res = _requests.get(f"http://localhost:8000/api/clustering/status/{task_id}")
                        if status_res.status_code == 200:
                            status_data = status_res.json()
                            if status_data["status"] == "completed":
                                progress_bar.progress(100)
                                progress_text.success("✅ Clustering completed successfully!")
                                st.write("**Cluster Labels:**", status_data["result"]["labels"])
                                break
                            elif status_data["status"] == "failed":
                                progress_text.error(f"❌ Clustering failed: {status_data.get('error')}")
                                break
                            else:
                                progress_text.info(f"⏳ Status: {status_data['status']}... (polling)")
                                progress_bar.progress(min(i * 5, 90))
                    except Exception:
                        pass
                    _time.sleep(2)
                else:
                    st.error("Timeout: clustering took too long.")
            else:
                st.error(f"Failed to start task (HTTP {response.status_code}).")
        except _requests.exceptions.ConnectionError:
            st.error("Could not connect to backend. Is FastAPI running on port 8000?")
    # ── End Background Clustering UI ────────────────────────────────────────

    # Get data for enhanced analytics
    history_data = []
    try:
        from datetime import datetime, timedelta

        from src.db.corpus_db import get_scan_history

        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        history_data = get_scan_history(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            limit=100
        )
    except Exception as e:
        logger.error(f"Failed to load history data: {e}")

    # Render enhanced analytics
    render_enhanced_analytics_tab(
        sim_matrix=active_sim_df if active_sim_df is not None else pd.DataFrame(),
        history_data=history_data,
        timings=st.session_state.get("last_stage_timings", {})
    )

# ══ TAB 8: PATTERNS (Issue #2840) ═════════════════════════════════════════
with tab_patterns:
    update_page_title("Pattern Recognition")
    try:
        from app.components.pattern_recognition_ui import render_pattern_recognition

        render_pattern_recognition()
    except Exception as exc:
        logger.error("Failed to render pattern recognition: %s", exc)
        st.error("Pattern recognition system unavailable.")

# ══ TAB 9: USERS ══════════════════════════════════════════════════════════
with tab_users:
    update_page_title("Users")
    render_users_view()

# ══ TAB 9: SETTINGS ═══════════════════════════════════════════════════════
with tab_settings:
    update_page_title("Settings")
    render_settings_view(user_role, lang_code, str(ROOT_DIR))

with tab_history:
    update_page_title("History")
    render_history_view()

# ══ TAB 10: SECURITY AUDIT LOGS ═════════════════════════════════════════════
with tab_audit:
    update_page_title("Security Audit Logs")
    st.subheader(get_text("tab_audit_logs", lang=lang_code))

    if user_role != "admin":
        st.error("🔒 Access Denied: Administrator privileges required.")
    else:
        st.markdown("### 📜 System Security Audit Trail")

        # ... [existing filters] ...
        
        # Issue #2732: Pagination controls
        EVENTS_PER_PAGE = 20
        
        if "audit_page_offset" not in st.session_state:
            st.session_state.audit_page_offset = 0
            
        current_offset = st.session_state.audit_page_offset
        
        # Fetch records for current page
        from src.db.security_audit import get_recent_audit_events, get_audit_events_count
        
        logs = get_recent_audit_events(
            limit=EVENTS_PER_PAGE,
            offset=current_offset,
            username=username_filter,
            event_type=event_type_filter
        )
        
        total_records = get_audit_events_count(
            username=username_filter,
            event_type=event_type_filter
        )
        
        total_pages = max(1, (total_records + EVENTS_PER_PAGE - 1) // EVENTS_PER_PAGE)
        current_page = (current_offset // EVENTS_PER_PAGE) + 1

        # Summary Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("📋 Total Log Entries", total_records)
        m2.metric("🏷️ Active Filter", selected_event_type or "All")
        m3.metric("📑 Page", f"{current_page} / {total_pages}")

        st.divider()

        # Display Data Table
        if logs:
            df = pd.DataFrame(logs)
            # ... [existing dataframe formatting] ...
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Pagination Controls (Issue #2732)
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            
            with nav_col1:
                if st.button(
                    "← Previous Page",
                    disabled=(current_offset == 0),
                    key="audit_prev_page_btn",
                    use_container_width=True
                ):
                    st.session_state.audit_page_offset = max(0, current_offset - EVENTS_PER_PAGE)
                    st.rerun()

            with nav_col2:
                st.caption(
                    f"Showing {current_offset + 1} - {min(current_offset + EVENTS_PER_PAGE, total_records)} of {total_records} logs"
                )

            with nav_col3:
                if st.button(
                    "Next Page →",
                    disabled=(current_offset + EVENTS_PER_PAGE >= total_records),
                    key="audit_next_page_btn",
                    use_container_width=True
                ):
                    st.session_state.audit_page_offset = current_offset + EVENTS_PER_PAGE
                    st.rerun()
        else:
            st.info("ℹ️ No security audit log records found matching the specified filters.")
            
        # Reset offset when filters change
        if "last_audit_filter" not in st.session_state:
            st.session_state.last_audit_filter = (username_filter, event_type_filter)
            
        current_filter = (username_filter, event_type_filter)
        if st.session_state.last_audit_filter != current_filter:
            st.session_state.audit_page_offset = 0
            st.session_state.last_audit_filter = current_filter
            st.rerun()

    # ========== USE THE NEW MODULE ==========
    render_audit_view(user_role, lang_code)
    
# Sidebar document management details
render_document_management_sidebar(user_role, _INDEX_PATH, SESSION_ID, last_interaction)

# ── Document Management & Bulk Export (Admin Only) ────────────────────────────
if user_role == "admin":
    st.markdown("---")
    st.markdown("### 📁 Document Management & Bulk Export")
    existing_docs = get_all_documents()
    if existing_docs:
        # ── File Extension Filter Dropdown (Issue #1883) ─────────────────────
        from pathlib import Path
        
        # Define standard extensions that should always appear in the filter
        standard_extensions = ["All", ".pdf", ".docx", ".txt", ".csv", ".md"]
        
        # Extract extensions from the current document list
        available_extensions = set()
        for doc in existing_docs:
            fn = (
                doc.filename
                if hasattr(doc, "filename")
                else (doc.get("filename") if isinstance(doc, dict) else str(doc))
            )
            ext = Path(fn).suffix.lower()
            if ext:
                available_extensions.add(ext)
        
        # Combine standard and available extensions, maintaining order
        filter_options = ["All"]
        for ext in standard_extensions[1:]:
            if ext in available_extensions or ext in [".pdf", ".docx", ".txt", ".csv", ".md"]:
                filter_options.append(ext)
        
        # Add any non-standard extensions found in the database
        for ext in sorted(list(available_extensions)):
            if ext not in filter_options:
                filter_options.append(ext)
        
        # Render the selectbox with a unique key to prevent state collisions
        selected_file_type = st.selectbox(
            "Filter by File Type",
            options=filter_options,
            index=0,
            key="corpus_file_type_filter_selectbox_footer",
            help="Filter the document corpus table to show only specific file extensions.",
        )
        
        # Apply the file extension filter to the existing_docs list
        if selected_file_type != "All":
            existing_docs = [
                doc for doc in existing_docs
                if Path(
                    doc.filename if hasattr(doc, "filename") 
                    else (doc.get("filename") if isinstance(doc, dict) else str(doc))
                ).suffix.lower() == selected_file_type
            ]
        
        # ── End File Extension Filter (Issue #1883) ──────────────────────────

        st.write(
            f"**{len(existing_docs)}** documents in database"
            + (f" (Filtered by: {selected_file_type})" if selected_file_type != "All" else "")
        )

        # Bulk selection and export logic
        import pandas as pd

        from src.db.corpus_db import (
            get_document_char_counts,
            get_document_word_counts,
        )

        word_counts = get_document_word_counts()
        char_counts = get_document_char_counts()

        doc_rows = []
        for doc in existing_docs:
            fn = (
                doc.filename
                if hasattr(doc, "filename")
                else (doc.get("filename") if isinstance(doc, dict) else str(doc))
            )
            doc_rows.append(
                {
                    "Select": False,
                    "Format": format_extension_badge(fn),
                    "Filename": fn,
                    "Word Count": word_counts.get(fn, 0),
                    "Char Count": char_counts.get(fn, 0),
                }
            )

        corpus_df = pd.DataFrame(doc_rows)

        if st.session_state.get("corpus_select_all_toggle", False):
            corpus_df["Select"] = True

        edited_df = st.data_editor(
            corpus_df,
            column_config={
                "Select": st.column_config.CheckboxColumn("Select", default=False),
                "Format": st.column_config.TextColumn("Format", disabled=True, width="small"),
                "Filename": st.column_config.TextColumn("Filename", disabled=True),
                "Word Count": st.column_config.NumberColumn("Word Count", format="%d words", disabled=True),
                "Char Count": st.column_config.NumberColumn("Char Count", format="%d chars", disabled=True),
            },
            disabled=["Format", "Filename", "Word Count", "Char Count"],
            hide_index=True,
            key="sidebar_corpus_data_editor",
            use_container_width=True,
        )

        selected_rows = edited_df[edited_df["Select"]]
        selected_filenames = selected_rows["Filename"].tolist()

        if selected_filenames:
            zip_data = create_documents_bulk_zip_archive(selected_filenames)
            st.download_button(
                label=f"📦 Export Selected as ZIP ({len(selected_filenames)})",
                data=zip_data,
                file_name=f"corpus_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                key="sidebar_export_selected_zip_btn",
                use_container_width=True,
                type="primary",
            )

        st.markdown("---")
        for doc in existing_docs:
            fn = (
                doc.filename
                if hasattr(doc, "filename")
                else (doc.get("filename") if isinstance(doc, dict) else str(doc))
            )
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"{format_extension_badge(fn)} {fn}")
            with col2:
                if st.button("🗑️", key=f"del_{fn}"):
                    delete_document(fn)
                    embeddings_matrix = get_all_embeddings()
                    if embeddings_matrix.size > 0:
                        new_index = build_index_from_matrix(embeddings_matrix)
                        save_index(new_index, _INDEX_PATH)
                    else:
                        if os.path.exists(_INDEX_PATH):
                            os.remove(_INDEX_PATH)
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(
        "🗑️ Clear All Documents",
        key="clear_all_documents_button",
        use_container_width=True,
    ):
        clear_all_dialog()  # type: ignore


# ── Onboarding Tour for First-Time Admin Users ───────────────────────────────────
if (
    Tour is not None
    and user_role == "admin"
    and st.session_state.get(SessionKeys.USERNAME)
    and not get_tour_completed(st.session_state[SessionKeys.USERNAME])
):
    username = st.session_state[SessionKeys.USERNAME]
    if st.button("🎯 Start Guided Tour", key="start_tour_button", type="primary"):
        st.session_state[SessionKeys.SHOW_TOUR] = True

    if st.session_state.get(SessionKeys.SHOW_TOUR, False):
        tour_steps = [
            Tour.info(
                title="👋 Welcome to the Plagiarism Detection System!",
                desc="This guided tour will walk you through the key features to help you get started.",
            ),
            Tour.bind(
                SessionKeys.THRESHOLD_SLIDER,
                title="⚙️ Plagiarism Threshold",
                desc=f"Adjust the flagging threshold. Default is {DEFAULT_THRESHOLDS.plagiarism:.0%}.",
                side="right",
            ),
            Tour.bind(
                SessionKeys.CLASS_FILTER_SELECTBOX,
                title="🔍 Class Filter",
                desc="Filter analysis results by specific class sections.",
                side="right",
            ),
            Tour.info(
                title="📊 Analysis Dashboard",
                desc="View similarity metrics, flagged pairs, and comparisons in the tabs below.",
            ),
        ]
        tour = Tour(steps=tour_steps)
        tour.start()
        if st.button("✅ Finish Tour", use_container_width=True):
            set_tour_completed(username, True)
            st.session_state[SessionKeys.SHOW_TOUR] = False
            st.success("✅ Onboarding tour completed!")
            st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
from src.utils.version_check import APP_VERSION, check_for_update_sync

if "_update_check_tag" not in st.session_state:
    st.session_state["_update_check_tag"] = check_for_update_sync(APP_VERSION)

_latest_tag: str | None = st.session_state["_update_check_tag"]

_footer_col1, _footer_col2 = st.columns([3, 1])
with _footer_col1:
    st.caption(
        f"🎓 Semantic Plagiarism Detection System · v{APP_VERSION} · Streamlit · "
        "🐛 Report Bug / Feedback"
    )
    render_session_status_banner()

    # Logout button placed in footer for easy access
    if st.button("🚪 Log Out", use_container_width=True, key="logout_button_footer"):
        logout_dialog()

with _footer_col2:
    if _latest_tag:
        st.markdown(
            version_check_widget_html(
                local_version=APP_VERSION,
                latest_tag=_latest_tag,
            ),
            unsafe_allow_html=True,
        )
    else:
        st.caption("✅ Up to date")


# ─── End of file ──────────────────────────────────────────────────────────────

# ── Hybrid Similarity (Lexical + Semantic) ─────────────────────────────────────

def compute_hybrid_similarity_matrix(
    semantic_df: pd.DataFrame,
    lexical_df: pd.DataFrame,
    alpha: float = 0.7,
) -> pd.DataFrame:
    """
    Compute hybrid similarity matrix combining semantic and lexical scores.
    
    Formula: hybrid = alpha * semantic + (1 - alpha) * lexical
    
    Args:
        semantic_df: Semantic similarity matrix (from embeddings)
        lexical_df: Lexical similarity matrix (from text)
        alpha: Weight for semantic similarity (0.7 = 70% semantic, 30% lexical)
        
    Returns:
        Hybrid similarity DataFrame
    """
    if semantic_df.shape != lexical_df.shape:
        raise ValueError("Semantic and lexical matrices must have same shape")
    
    hybrid = alpha * semantic_df + (1 - alpha) * lexical_df
    return hybrid


def compute_hybrid_similarity_score(
    text_a: str,
    text_b: str,
    semantic_score: float,
    alpha: float = 0.7,
    lexical_method: str = 'tfidf'
) -> float:
    """
    Compute hybrid similarity for a single pair.
    
    Args:
        text_a: First text
        text_b: Second text
        semantic_score: Semantic similarity score
        alpha: Weight for semantic similarity
        lexical_method: Lexical method to use ('tfidf', 'jaccard', 'dice', 'overlap', 'ngram')
        
    Returns:
        Hybrid similarity score
    """
    from src.core.lexical_similarity import (
        calculate_lexical_similarity,
        jaccard_similarity,
        dice_coefficient,
        overlap_coefficient,
        n_gram_overlap,
        compute_char_ngram_similarity
    )
    
    if lexical_method == 'tfidf':
        lexical_score = calculate_lexical_similarity(text_a, text_b)
    elif lexical_method == 'jaccard':
        lexical_score = jaccard_similarity(text_a, text_b)
    elif lexical_method == 'dice':
        lexical_score = dice_coefficient(text_a, text_b)
    elif lexical_method == 'overlap':
        lexical_score = overlap_coefficient(text_a, text_b)
    elif lexical_method == 'ngram':
        lexical_score = n_gram_overlap(text_a, text_b, n=3)
    elif lexical_method == 'char_ngram':
        lexical_score = compute_char_ngram_similarity(text_a, text_b, n=5)
    else:
        lexical_score = calculate_lexical_similarity(text_a, text_b)
    
    hybrid_score = alpha * semantic_score + (1 - alpha) * lexical_score
    return min(1.0, max(0.0, hybrid_score))


def get_hybrid_similarity_stats(
    semantic_scores: list[float],
    lexical_scores: list[float],
    alpha: float = 0.7
) -> dict[str, any]:
    """
    Get statistics about hybrid similarity distribution.
    
    Args:
        semantic_scores: List of semantic scores
        lexical_scores: List of lexical scores
        alpha: Weight for semantic similarity
        
    Returns:
        Dictionary with statistics
    """
    if not semantic_scores or not lexical_scores:
        return {
            'semantic_avg': 0.0,
            'lexical_avg': 0.0,
            'hybrid_avg': 0.0,
            'hybrid_min': 0.0,
            'hybrid_max': 0.0,
            'semantic_lexical_correlation': 0.0,
            'alpha_used': alpha
        }
    
    hybrid_scores = [
        alpha * sem + (1 - alpha) * lex
        for sem, lex in zip(semantic_scores, lexical_scores)
    ]
    
    return {
        'semantic_avg': sum(semantic_scores) / len(semantic_scores),
        'lexical_avg': sum(lexical_scores) / len(lexical_scores),
        'hybrid_avg': sum(hybrid_scores) / len(hybrid_scores),
        'hybrid_min': min(hybrid_scores),
        'hybrid_max': max(hybrid_scores),
        'semantic_lexical_correlation': np.corrcoef(semantic_scores, lexical_scores)[0, 1] 
            if len(semantic_scores) > 1 else 0.0,
        'alpha_used': alpha
    }


# ─── End of file ──────────────────────────────────────────────────────────────
