"""
Plagiarism Graph Engine (Issue #4259).

This module implements a knowledge graph engine for detecting plagiarism rings,
collusion, and historical trends across assignments and students. It uses NetworkX
as the core implementation to ensure local operation without requiring a Neo4j server,
while organizing the graph model to be easily adapted to Neo4j if configured in the future.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

try:
    from networkx.algorithms.community import louvain_communities
except ImportError:
    louvain_communities = None

logger = logging.getLogger(__name__)


# --- Data Models ---

@dataclass
class GraphNode:
    """Base class for all graph nodes."""
    id: str
    labels: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentNode(GraphNode):
    """Represents a submitted assignment or document."""
    def __init__(self, doc_id: str, title: str, year: int = None, **kwargs):
        super().__init__(id=doc_id, labels={"Document"}, properties={"title": title, "year": year, **kwargs})


@dataclass
class StudentNode(GraphNode):
    """Represents a student or user in the system."""
    def __init__(self, student_id: str, name: str, **kwargs):
        super().__init__(id=student_id, labels={"Student"}, properties={"name": name, **kwargs})


@dataclass
class SimilarityEdge:
    """Represents a similarity or plagiarism link between two documents."""
    source_id: str
    target_id: str
    score: float
    timestamp: datetime
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphStats:
    """Statistical summary of the plagiarism graph."""
    node_count: int
    edge_count: int
    density: float
    connected_components: int
    suspicious_clusters: int
    highly_connected_students: List[Dict[str, Any]]
    historical_trends: Dict[int, int]


@dataclass
class RingDetectionResult:
    """Result of running community/ring detection algorithms."""
    community_id: int
    nodes: List[str]
    size: int
    average_similarity: float
    is_suspicious: bool


# --- Graph Engine ---

class PlagiarismGraphEngine:
    """
    Core graph engine leveraging NetworkX for analyzing plagiarism rings and ecosystems.
    Designed with a provider-agnostic interface to allow Neo4j adapters in the future.
    """

    def __init__(self):
        # We use an undirected graph for similarities since plagiarism matching is symmetric.
        # Can use MultiGraph if we want multiple matches per pair, but Graph is safer for stats.
        self.graph = nx.Graph()
        
    def add_node(self, node: GraphNode) -> None:
        """Adds a generic node to the graph."""
        if not node or not node.id:
            logger.warning("Attempted to add an invalid or empty node.")
            return
            
        self.graph.add_node(
            node.id, 
            labels=list(node.labels),
            **node.properties
        )

    def add_document(self, doc_id: str, title: str, year: Optional[int] = None, **kwargs) -> None:
        """Adds a document node."""
        node = DocumentNode(doc_id=doc_id, title=title, year=year, **kwargs)
        self.add_node(node)

    def add_student(self, student_id: str, name: str, **kwargs) -> None:
        """Adds a student node."""
        node = StudentNode(student_id=student_id, name=name, **kwargs)
        self.add_node(node)
        
    def add_authorship_edge(self, student_id: str, doc_id: str) -> None:
        """Links a student to their submitted document."""
        if not student_id or not doc_id:
            return
            
        # Ensure nodes exist
        if not self.graph.has_node(student_id):
            self.add_student(student_id, f"Student {student_id}")
        if not self.graph.has_node(doc_id):
            self.add_document(doc_id, f"Document {doc_id}")
            
        self.graph.add_edge(student_id, doc_id, type="AUTHORED", weight=1.0)

    def add_similarity_edge(self, edge: SimilarityEdge) -> None:
        """Adds a plagiarism similarity edge between two documents."""
        if not edge.source_id or not edge.target_id or edge.source_id == edge.target_id:
            return
            
        if edge.score < 0.0 or edge.score > 1.0:
            logger.warning(f"Invalid similarity score {edge.score} ignored.")
            return

        # Ensure nodes exist to prevent Orphan edges
        if not self.graph.has_node(edge.source_id):
            self.add_document(edge.source_id, f"Unknown {edge.source_id}")
        if not self.graph.has_node(edge.target_id):
            self.add_document(edge.target_id, f"Unknown {edge.target_id}")

        self.graph.add_edge(
            edge.source_id, 
            edge.target_id, 
            type="COPIED_FROM", 
            score=edge.score,
            timestamp=edge.timestamp.isoformat() if edge.timestamp else None,
            **edge.properties
        )

    def build_from_faiss_flags(self, flags: List[Dict[str, Any]], default_year: int = None) -> None:
        """
        Adapter method: Populates the graph using standard output from the existing 
        FAISS / Similarity pipelines in `processing.py`.
        """
        if not flags:
            return
            
        now = datetime.now()
        year = default_year or now.year

        for flag in flags:
            doc_a = flag.get("doc_a")
            doc_b = flag.get("doc_b")
            score = flag.get("similarity", 0.0)
            
            if not doc_a or not doc_b:
                continue
                
            self.add_document(doc_a, doc_a, year=year)
            self.add_document(doc_b, doc_b, year=year)
            
            edge = SimilarityEdge(
                source_id=doc_a,
                target_id=doc_b,
                score=float(score),
                timestamp=now,
                properties={"severity": flag.get("severity", "Unknown")}
            )
            self.add_similarity_edge(edge)

    def filter_graph(self, min_similarity: float = 0.0, year: Optional[int] = None) -> nx.Graph:
        """
        Returns a subgraph filtered by minimum similarity threshold and/or year.
        """
        if self.graph.number_of_nodes() == 0:
            return nx.Graph()

        # Filter edges by similarity threshold
        valid_edges = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("type") == "COPIED_FROM":
                if data.get("score", 0.0) >= min_similarity:
                    valid_edges.append((u, v))
            elif data.get("type") == "AUTHORED":
                valid_edges.append((u, v))
                
        filtered_graph = self.graph.edge_subgraph(valid_edges).copy()
        
        # Filter nodes by year if specified
        if year is not None:
            valid_nodes = [
                n for n, data in filtered_graph.nodes(data=True)
                if data.get("year") == year or "Student" in data.get("labels", [])
            ]
            filtered_graph = filtered_graph.subgraph(valid_nodes).copy()
            
        # Clean up isolated nodes
        filtered_graph.remove_nodes_from(list(nx.isolates(filtered_graph)))
        
        return filtered_graph

    def detect_suspicious_rings(self, min_similarity: float = 0.5) -> List[RingDetectionResult]:
        """
        Uses Community Detection (Louvain or Connected Components) to identify 
        clusters of students/documents collaborating or copying heavily.
        """
        subgraph = self.filter_graph(min_similarity=min_similarity)
        
        if subgraph.number_of_nodes() == 0:
            return []

        # Extract only the COPIED_FROM edges to find document-to-document rings
        similarity_edges = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get("type") == "COPIED_FROM"]
        doc_graph = subgraph.edge_subgraph(similarity_edges)
        
        rings = []
        
        # Prefer Louvain method if available (better for dense community detection)
        if louvain_communities is not None and doc_graph.number_of_edges() > 0:
            try:
                communities = louvain_communities(doc_graph, weight='score')
            except Exception as e:
                logger.warning(f"Louvain community detection failed, falling back to connected components: {e}")
                communities = list(nx.connected_components(doc_graph))
        else:
            communities = list(nx.connected_components(doc_graph))

        for idx, community_nodes in enumerate(communities):
            nodes_list = list(community_nodes)
            size = len(nodes_list)
            
            # A ring must involve more than just 1 isolated node
            if size < 2:
                continue
                
            # Calculate average similarity in this community
            community_subgraph = doc_graph.subgraph(nodes_list)
            scores = [d.get("score", 0.0) for u, v, d in community_subgraph.edges(data=True)]
            avg_sim = sum(scores) / len(scores) if scores else 0.0
            
            # Flag as suspicious if the community size > 2 (a triangle or larger ring)
            # or if the average similarity is extremely high
            is_suspicious = size >= 3 or avg_sim >= 0.85
            
            rings.append(RingDetectionResult(
                community_id=idx + 1,
                nodes=nodes_list,
                size=size,
                average_similarity=round(avg_sim, 4),
                is_suspicious=is_suspicious
            ))
            
        # Sort by suspiciousness and size
        rings.sort(key=lambda r: (r.is_suspicious, r.size, r.average_similarity), reverse=True)
        return rings

    def calculate_centrality(self, subgraph: nx.Graph = None) -> Dict[str, float]:
        """
        Identifies the 'central' nodes in a plagiarism ring using PageRank.
        High centrality often indicates the original source document from which others copied.
        """
        graph_to_use = subgraph if subgraph is not None else self.graph
        
        if graph_to_use.number_of_nodes() == 0:
            return {}
            
        try:
            return nx.pagerank(graph_to_use, weight='score')
        except Exception as e:
            logger.warning(f"PageRank calculation failed: {e}")
            return {}

    def get_graph_stats(self) -> GraphStats:
        """
        Generates overarching statistics for the entire plagiarism ecosystem.
        """
        if self.graph.number_of_nodes() == 0:
            return GraphStats(
                node_count=0, edge_count=0, density=0.0, 
                connected_components=0, suspicious_clusters=0,
                highly_connected_students=[], historical_trends={}
            )

        # Basic counts
        n = self.graph.number_of_nodes()
        e = self.graph.number_of_edges()
        density = nx.density(self.graph)
        
        # Connected components overall
        components = nx.number_connected_components(self.graph)
        
        # Suspicious clusters
        rings = self.detect_suspicious_rings(min_similarity=0.4)
        suspicious_count = sum(1 for r in rings if r.is_suspicious)
        
        # Highly connected nodes (centrality)
        centrality = self.calculate_centrality()
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        highly_connected = [{"id": node, "score": score} for node, score in sorted_centrality]
        
        # Historical trends
        trends = {}
        for _, data in self.graph.nodes(data=True):
            if "Document" in data.get("labels", []) and data.get("year"):
                year = data["year"]
                trends[year] = trends.get(year, 0) + 1

        return GraphStats(
            node_count=n,
            edge_count=e,
            density=round(density, 4),
            connected_components=components,
            suspicious_clusters=suspicious_count,
            highly_connected_students=highly_connected,
            historical_trends=trends
        )
