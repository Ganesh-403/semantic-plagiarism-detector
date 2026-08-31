# ───────────────────────────────────────────────────────────────────────────────
# ── SECTION: INTELLIGENT DOCUMENT TAGGING & CATEGORIZATION (Issue #1988) ────
# ───────────────────────────────────────────────────────────────────────────────

import hashlib
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

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
        return {**asdict(self), "created_at": self.created_at.isoformat()}


@dataclass
class DocumentCategory:
    """Represents a document category"""

    id: str
    name: str
    description: str
    parent_id: Optional[str] = None
    color: str = "#808080"
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
        return {**asdict(self), "created_at": self.created_at.isoformat()}


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
        return {**asdict(self), "assigned_at": self.assigned_at.isoformat()}


# ── Tag Generator ───────────────────────────────────────────────────────────


class IntelligentTagGenerator:
    """Generates intelligent tags from document content"""

    def __init__(self):
        self.common_words = {
            "plagiarism": ["academic", "integrity", "ethics", "copying", "similarity"],
            "research": ["study", "analysis", "methodology", "literature", "review"],
            "data": ["analysis", "statistics", "results", "findings", "visualization"],
            "algorithm": ["code", "implementation", "performance", "optimization"],
            "machine learning": ["ai", "neural", "deep", "training", "model"],
            "software": ["development", "programming", "system", "application"],
            "education": [
                "learning",
                "teaching",
                "curriculum",
                "student",
                "assessment",
            ],
            "ethics": ["privacy", "security", "compliance", "policy", "regulation"],
        }

        self.stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "without",
            "by",
            "from",
            "up",
            "down",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
        }

        self.tag_cache = {}

    def generate_tags(
        self, content: str, max_tags: int = 10
    ) -> List[Tuple[str, float]]:
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
        words = re.findall(r"\b[a-zA-Z]{3,}\b", content.lower())

        # Remove stopwords
        words = [w for w in words if w not in self.stopwords]

        # Count frequencies
        freq = Counter(words)
        total = len(words) or 1

        # Calculate scores (simple TF)
        scores = {word: count / total for word, count in freq.items()}

        # Boost common topics
        for topic, keywords in self.common_words.items():
            for keyword in keywords:
                if keyword in scores:
                    scores[keyword] *= 1.5

        return scores

    def _score_tags(
        self, keywords: Dict[str, float], content: str
    ) -> List[Tuple[str, float]]:
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
            return "Uncategorized", 0.0

        topic_scores = self._detect_topics(content)

        if topic_scores:
            best_topic, best_score = max(topic_scores, key=lambda x: x[1])
            if best_score > 0.3:
                return best_topic.title(), best_score

        # Default categories
        categories = ["Academic", "Research", "Technical", "Review", "Report"]
        for category in categories:
            if category.lower() in content.lower():
                return category, 0.6

        return "Uncategorized", 0.3


# ── Tag Manager ─────────────────────────────────────────────────────────────


class TagManager:
    """Manages document tags and categories"""

    def __init__(self):
        self.tags: dict[str, DocumentTag] = {}
        self.categories: dict[str, DocumentCategory] = {}
        self.assignments: list[TagAssignment] = []
        self.tag_counter = Counter()
        self.tag_usage = defaultdict(int)

        # Initialize default categories
        self._init_default_categories()

    def _init_default_categories(self):
        """Initialize default categories"""
        defaults = [
            (
                "academic",
                "Academic",
                "Academic integrity and plagiarism related",
                "#4CAF50",
            ),
            ("research", "Research", "Research methodologies and findings", "#2196F3"),
            (
                "technical",
                "Technical",
                "Technical documents and implementations",
                "#FF9800",
            ),
            ("review", "Review", "Document reviews and analysis", "#9C27B0"),
            ("report", "Report", "Reports and summaries", "#F44336"),
        ]

        for id, name, desc, color in defaults:
            if id not in self.categories:
                self.categories[id] = DocumentCategory(
                    id=id, name=name, description=desc, color=color
                )

    def add_tag(
        self,
        name: str,
        category: str = "custom",
        confidence: float = 1.0,
        auto_generated: bool = False,
        user_id: str = "system",
    ) -> DocumentTag:
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
            is_auto_generated=auto_generated,
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

    def assign_tag(
        self,
        document_name: str,
        tag_name: str,
        user_id: str = "system",
        auto: bool = False,
    ) -> Optional[str]:
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
            is_auto=auto,
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
            a
            for a in self.assignments
            if not (a.document_name == document_name and a.tag_id == tag.id)
        ]

        if self.tag_counter[tag_name] > 0:
            self.tag_counter[tag_name] -= 1
        return True

    def get_document_tags(self, document_name: str) -> List[DocumentTag]:
        """Get all tags for a document"""
        doc_assignments = [
            a for a in self.assignments if a.document_name == document_name
        ]
        return [self.tags[a.tag_id] for a in doc_assignments if a.tag_id in self.tags]

    def get_documents_by_tag(self, tag_name: str) -> List[str]:
        """Get all documents with a specific tag"""
        tag = self.get_tag_by_name(tag_name)
        if not tag:
            return []
        return [a.document_name for a in self.assignments if a.tag_id == tag.id]

    def add_category(
        self,
        name: str,
        description: str,
        parent_id: Optional[str] = None,
        color: str = "#808080",
    ) -> DocumentCategory:
        """Add a new category"""
        category_id = str(uuid.uuid4())
        category = DocumentCategory(
            id=category_id,
            name=name,
            description=description,
            parent_id=parent_id,
            color=color,
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
            "total_tags": len(self.tags),
            "total_assignments": len(self.assignments),
            "most_used": self.tag_counter.most_common(10),
            "categories": {
                cat: len([t for t in self.tags.values() if t.category == cat])
                for cat in set(t.category for t in self.tags.values())
            },
        }

    def get_tag_analytics(self) -> pd.DataFrame:
        """Get tag analytics as DataFrame"""
        data = []
        for tag_name, count in self.tag_counter.items():
            tag = self.get_tag_by_name(tag_name)
            data.append(
                {
                    "Tag": tag_name,
                    "Count": count,
                    "Category": tag.category if tag else "unknown",
                    "Confidence": tag.confidence if tag else 0,
                    "Auto": tag.is_auto_generated if tag else False,
                }
            )
        return pd.DataFrame(data)


# ── Auto-Categorizer ──────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def _categorize_content(content: str) -> tuple[list[tuple[str, float]], str, float]:
    """Cache the CPU-heavy tagging/category analysis for immutable document content."""
    generator = IntelligentTagGenerator()
    tags = generator.generate_tags(content)
    category, confidence = generator.generate_categories(content)
    return tags, category, confidence


class AutoCategorizer:
    """Automatically categorizes documents"""

    def __init__(self, tag_manager: TagManager, tag_generator: IntelligentTagGenerator):
        self.tag_manager = tag_manager
        self.tag_generator = tag_generator
        self.categorization_history = []

    def categorize_document(
        self, document_name: str, content: str, user_id: str = "system"
    ) -> Dict:
        """Automatically categorize a document"""
        if not content:
            return {"status": "failed", "reason": "No content"}

        # Cache the CPU-heavy analysis so repeated categorization does not block the UI.
        tags, category, confidence = _categorize_content(content)

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
            "document_name": document_name,
            "category": category,
            "category_confidence": confidence,
            "assigned_tags": assigned,
            "total_tags_generated": len(tags),
            "timestamp": datetime.now(),
        }

        self.categorization_history.append(result)
        return result

    def _get_category_id(self, category_name: str) -> Optional[str]:
        """Get category ID by name"""
        for cat in self.tag_manager.get_all_categories():
            if cat.name.lower() == category_name.lower():
                return cat.id
        return None

    def batch_categorize(
        self, documents: Dict[str, str], user_id: str = "system"
    ) -> List[Dict]:
        """Categorize multiple documents"""
        results = []
        for doc_name, content in documents.items():
            result = self.categorize_document(doc_name, content, user_id)
            results.append(result)
        return results

    def get_categorization_stats(self) -> Dict:
        """Get categorization statistics"""
        if not self.categorization_history:
            return {"total": 0}

        categories = [r["category"] for r in self.categorization_history]
        category_counts = Counter(categories)

        return {
            "total": len(self.categorization_history),
            "categories": dict(category_counts),
            "avg_tags": sum(
                len(r["assigned_tags"]) for r in self.categorization_history
            )
            / len(self.categorization_history),
        }


# ── Tag Suggestion Engine ──────────────────────────────────────────────────


class TagSuggestionEngine:
    """Provides tag suggestions based on content and context"""

    def __init__(self, tag_manager: TagManager, tag_generator: IntelligentTagGenerator):
        self.tag_manager = tag_manager
        self.tag_generator = tag_generator
        self.suggestion_history = []

    def suggest_tags(
        self, content: str, existing_tags: List[str] = None, max_suggestions: int = 5
    ) -> List[Tuple[str, float]]:
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

        self.suggestion_history.append(
            {
                "timestamp": datetime.now(),
                "content_length": len(content),
                "suggestions": suggestions,
            }
        )

        return suggestions[:max_suggestions]

    def get_suggestion_stats(self) -> Dict:
        """Get suggestion statistics"""
        return {
            "total_suggestions": len(self.suggestion_history),
            "avg_suggestions": sum(
                len(s["suggestions"]) for s in self.suggestion_history
            )
            / len(self.suggestion_history)
            if self.suggestion_history
            else 0,
        }


# ── UI Components ──────────────────────────────────────────────────────────


def render_tag_management_ui(tag_manager: TagManager, document_name: str = None):
    """Render tag management UI"""
    st.subheader("🏷️ Tag Management")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Current Tags", "➕ Add Tags", "📊 Analytics", "📁 Categories"]
    )

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
                            f"{tag.name} {'' if tag.confidence == 1.0 else f'({tag.confidence * 100:.0f}%)'}"
                            f"</span>",
                            unsafe_allow_html=True,
                        )

                        if st.button("❌", key=f"remove_{tag.id}_{document_name}"):
                            tag_manager.unassign_tag(document_name, tag.name)
                            st.rerun()
            else:
                st.info("No tags assigned to this document.")
        else:
            st.info("Select a document to view tags.")

    with tab2:
        if document_name:
            existing_tags = [
                t.name for t in tag_manager.get_document_tags(document_name)
            ]

            col1, col2 = st.columns(2)
            with col1:
                new_tag = st.text_input("Tag name:", key="new_tag_input")
                category = st.selectbox(
                    "Category:",
                    ["custom", "topic", "type", "status"],
                    key="tag_category_select",
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
    col1.metric("Total Tags", stats["total_tags"])
    col2.metric("Total Assignments", stats["total_assignments"])
    col3.metric("Categories", len(stats["categories"]))

    # Most used tags
    if stats["most_used"]:
        st.subheader("🔥 Most Used Tags")
        tag_data = pd.DataFrame(stats["most_used"], columns=["Tag", "Count"])
        st.bar_chart(tag_data.set_index("Tag"))

    # Category distribution
    if stats["categories"]:
        st.subheader("📁 Category Distribution")
        cat_data = pd.DataFrame(
            {
                "Category": list(stats["categories"].keys()),
                "Count": list(stats["categories"].values()),
            }
        )
        st.bar_chart(cat_data.set_index("Category"))

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
                cat_tags = [
                    t for t in tag_manager.get_all_tags() if t.category == cat.id
                ]
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
                ["None"] + [c.name for c in categories],
                key="parent_cat_select",
            )

        if st.button("Create Category", key="create_cat_btn"):
            if cat_name:
                parent_id = None
                if parent_cat != "None":
                    parent = next((c for c in categories if c.name == parent_cat), None)
                    if parent:
                        parent_id = parent.id

                tag_manager.add_category(cat_name, cat_desc, parent_id, cat_color)
                st.success(f"✅ Category '{cat_name}' created!")
                st.rerun()


def render_auto_categorization_ui(
    tag_manager: TagManager, tag_generator: IntelligentTagGenerator
):
    """Render auto-categorization UI"""
    st.subheader("🤖 Auto-Categorization")

    # Initialize auto-categorizer
    categorizer = AutoCategorizer(tag_manager, tag_generator)

    # Batch categorization
    st.subheader("📂 Batch Categorization")

    documents = st.session_state.get("document_names", [])
    if documents:
        selected_docs = st.multiselect(
            "Select documents to categorize:",
            options=documents,
            key="batch_categorize_select",
        )

        if selected_docs and st.button(
            "🚀 Run Auto-Categorization", key="auto_cat_btn"
        ):
            with st.spinner("Categorizing documents..."):
                # Get document contents
                doc_contents = {}
                for doc_name in selected_docs:
                    if doc_name in st.session_state.get("raw_texts", {}):
                        doc_contents[doc_name] = st.session_state["raw_texts"][doc_name]

                if doc_contents:
                    results = categorizer.batch_categorize(doc_contents)
                    st.success(f"✅ Categorized {len(results)} documents!")

                    # Show results
                    for result in results:
                        st.markdown(f"**{result['document_name']}**")
                        st.markdown(
                            f"- Category: {result['category']} ({result['category_confidence'] * 100:.0f}%)"
                        )
                        st.markdown(f"- Tags: {', '.join(result['assigned_tags'])}")
                        st.divider()
    else:
        st.info("No documents available. Upload documents first.")


def render_tag_suggestions_ui(
    tag_manager: TagManager, tag_generator: IntelligentTagGenerator
):
    """Render tag suggestions UI"""
    st.subheader("💡 Tag Suggestions")

    suggestion_engine = TagSuggestionEngine(tag_manager, tag_generator)

    document_name = st.selectbox(
        "Select document:",
        options=st.session_state.get("document_names", []),
        key="suggestion_doc_select",
    )

    if document_name:
        content = st.session_state.get("raw_texts", {}).get(document_name, "")
        existing_tags = [t.name for t in tag_manager.get_document_tags(document_name)]

        if content:
            if st.button("💡 Get Suggestions", key="suggest_tags_btn"):
                suggestions = suggestion_engine.suggest_tags(content, existing_tags)

                if suggestions:
                    st.subheader("Suggested Tags:")
                    for tag_name, confidence in suggestions:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(
                                f"**{tag_name}** ({confidence * 100:.0f}% confidence)"
                            )
                        with col2:
                            if st.button("✅ Add", key=f"add_suggest_{tag_name}"):
                                tag_manager.add_tag(tag_name, "custom", confidence)
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
    if "tag_manager" not in st.session_state:
        st.session_state["tag_manager"] = TagManager()
    if "tag_generator" not in st.session_state:
        st.session_state["tag_generator"] = IntelligentTagGenerator()

    tag_manager = st.session_state["tag_manager"]
    tag_generator = st.session_state["tag_generator"]

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🏷️ Tag Management",
            "🤖 Auto-Categorization",
            "💡 Tag Suggestions",
            "📊 Analytics",
        ]
    )

    with tab1:
        document_name = st.selectbox(
            "Select Document:",
            options=st.session_state.get("document_names", []),
            key="tag_doc_select",
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
    if "tag_manager" not in st.session_state:
        st.session_state["tag_manager"] = TagManager()
    if "tag_generator" not in st.session_state:
        st.session_state["tag_generator"] = IntelligentTagGenerator()

    # Auto-categorize documents during upload
    if st.session_state.get("new_documents_uploaded", False):
        tag_manager = st.session_state["tag_manager"]
        tag_generator = st.session_state["tag_generator"]
        categorizer = AutoCategorizer(tag_manager, tag_generator)

        # Get new documents
        raw_texts = st.session_state.get("raw_texts", {})
        if raw_texts:
            categorizer.batch_categorize(raw_texts)

        st.session_state["new_documents_uploaded"] = False

    # Add tagging tab to main app
    st.subheader("🏷️ Document Tagging System")
    render_tagging_dashboard()


# ── End of Tagging System ──────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
