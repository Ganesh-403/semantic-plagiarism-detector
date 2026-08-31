"""
Plagiarism Network & Collaboration Detection System
Detects complex plagiarism patterns, collaboration networks, and academic misconduct rings
"""

import json
import datetime
import hashlib
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import math
import random
import re
from enum import Enum

# ==================== Core Data Structures ====================

class NodeType(Enum):
    """Types of nodes in the plagiarism network"""
    STUDENT = "student"
    DOCUMENT = "document"
    SOURCE = "source"
    INSTITUTION = "institution"
    COHORT = "cohort"
    ASSIGNMENT = "assignment"
    COLLABORATOR = "collaborator"
    SUSPICIOUS_PATTERN = "suspicious_pattern"

class EdgeType(Enum):
    """Types of edges in the plagiarism network"""
    SIMILARITY = "similarity"
    CO_AUTHORSHIP = "co_authorship"
    COLLABORATION = "collaboration"
    SHARED_SOURCE = "shared_source"
    COHORT_MEMBER = "cohort_member"
    INSTITUTIONAL = "institutional"
    TEMPORAL = "temporal"
    SUSPICIOUS = "suspicious"

@dataclass
class Node:
    """Network node representing an entity"""
    id: str
    type: NodeType
    attributes: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime.datetime

@dataclass
class Edge:
    """Network edge representing relationships"""
    source: str
    target: str
    type: EdgeType
    weight: float
    attributes: Dict[str, Any]
    created_at: datetime.datetime

@dataclass
class Community:
    """Community detection result"""
    id: str
    nodes: List[str]
    size: int
    density: float
    type: str  # "collaboration", "plagiarism", "mixed"
    risk_score: float
    key_patterns: List[str]

@dataclass
class PlagiarismRing:
    """Plagiarism ring detection result"""
    id: str
    members: List[str]
    documents: List[str]
    centrality_score: float
    collusion_level: float  # 0-1
    pattern_type: str
    evidence: List[Dict]
    risk_level: str  # Low, Medium, High, Critical

# ==================== Network Construction ====================

class PlagiarismNetwork:
    """Build and manage plagiarism detection network"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.node_counter = 0
        self.edge_counter = 0
        
    def add_node(self, node_type: NodeType, attributes: Dict = None,
                 metadata: Dict = None, node_id: str = None) -> str:
        """Add a node to the network"""
        if node_id is None:
            node_id = f"node_{self.node_counter:06d}"
            self.node_counter += 1
        
        if attributes is None:
            attributes = {}
        if metadata is None:
            metadata = {}
        
        node = Node(
            id=node_id,
            type=node_type,
            attributes=attributes,
            metadata=metadata,
            created_at=datetime.datetime.now()
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, type=node_type.value, **attributes)
        return node_id
    
    def add_edge(self, source: str, target: str, edge_type: EdgeType,
                 weight: float = 1.0, attributes: Dict = None) -> str:
        """Add an edge between nodes"""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Node not found: {source} or {target}")
        
        if attributes is None:
            attributes = {}
        
        edge_id = f"edge_{self.edge_counter:06d}"
        self.edge_counter += 1
        
        edge = Edge(
            source=source,
            target=target,
            type=edge_type,
            weight=weight,
            attributes=attributes,
            created_at=datetime.datetime.now()
        )
        
        self.edges.append(edge)
        self.graph.add_edge(source, target, key=edge_id, type=edge_type.value,
                           weight=weight, **attributes)
        return edge_id
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str, edge_type: EdgeType = None) -> List[str]:
        """Get neighbors of a node"""
        if node_id not in self.graph:
            return []
        
        if edge_type:
            neighbors = []
            for neighbor in self.graph.neighbors(node_id):
                edges = self.graph.get_edge_data(node_id, neighbor)
                if edges:
                    for edge_data in edges.values():
                        if edge_data.get('type') == edge_type.value:
                            neighbors.append(neighbor)
                            break
            return neighbors
        else:
            return list(self.graph.neighbors(node_id))
    
    def get_node_degree(self, node_id: str) -> int:
        """Get degree of a node"""
        if node_id not in self.graph:
            return 0
        return self.graph.degree(node_id)
    
    def get_subgraph(self, nodes: List[str]) -> 'PlagiarismNetwork':
        """Extract subgraph containing specified nodes"""
        subnetwork = PlagiarismNetwork()
        
        # Add nodes
        for node_id in nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                subnetwork.add_node(node.type, node.attributes, node.metadata, node_id)
        
        # Add edges between nodes in subgraph
        for edge in self.edges:
            if edge.source in nodes and edge.target in nodes:
                subnetwork.add_edge(edge.source, edge.target, edge.type,
                                   edge.weight, edge.attributes)
        
        return subnetwork

# ==================== Advanced Detection Algorithms ====================

class AdvancedPatternDetector:
    """Detect complex plagiarism and collaboration patterns"""
    
    def __init__(self, network: PlagiarismNetwork):
        self.network = network
        self.patterns = []
        self.communities = []
    
    def detect_collaboration_networks(self, min_edges: int = 3) -> List[Dict]:
        """Detect collaboration networks among students"""
        collaboration_networks = []
        student_nodes = [n for n in self.network.nodes.values() 
                        if n.type == NodeType.STUDENT]
        
        # Build collaboration graph
        collab_graph = nx.Graph()
        for node in student_nodes:
            collab_graph.add_node(node.id)
        
        # Add collaboration edges
        for edge in self.network.edges:
            if edge.type == EdgeType.CO_AUTHORSHIP:
                if edge.source in collab_graph and edge.target in collab_graph:
                    collab_graph.add_edge(edge.source, edge.target, weight=edge.weight)
        
        # Find connected components (collaboration groups)
        components = list(nx.connected_components(collab_graph))
        
        for comp in components:
            if len(comp) >= 2:
                subgraph = collab_graph.subgraph(comp)
                density = nx.density(subgraph)
                
                # Detect if it's a plagiarism ring
                is_suspicious = False
                pattern_type = "normal_collaboration"
                risk_score = 0.0
                
                # Check for suspicious patterns
                if density > 0.5 and len(comp) > 3:
                    is_suspicious = True
                    pattern_type = "dense_plagiarism_ring"
                    risk_score = 0.8
                elif any(edge.get('weight', 1) > 2 for edge in subgraph.edges):
                    risk_score = 0.5
                    pattern_type = "strong_collaboration"
                
                collaboration_networks.append({
                    'members': list(comp),
                    'size': len(comp),
                    'density': density,
                    'edges': len(subgraph.edges),
                    'is_suspicious': is_suspicious,
                    'pattern_type': pattern_type,
                    'risk_score': risk_score
                })
        
        return collaboration_networks
    
    def detect_plagiarism_rings(self, similarity_threshold: float = 0.3,
                               min_members: int = 3) -> List[PlagiarismRing]:
        """Detect organized plagiarism rings"""
        rings = []
        
        # Build similarity graph
        sim_graph = nx.Graph()
        
        # Add all student nodes
        student_nodes = [n.id for n in self.network.nodes.values() 
                        if n.type == NodeType.STUDENT]
        for node in student_nodes:
            sim_graph.add_node(node)
        
        # Add similarity edges
        for edge in self.network.edges:
            if edge.type == EdgeType.SIMILARITY and edge.weight >= similarity_threshold:
                if edge.source in sim_graph and edge.target in sim_graph:
                    sim_graph.add_edge(edge.source, edge.target, weight=edge.weight)
        
        # Find communities (plagiarism rings)
        communities = list(nx.community.greedy_modularity_communities(sim_graph))
        
        for comm in communities:
            comm_list = list(comm)
            if len(comm_list) >= min_members:
                subgraph = sim_graph.subgraph(comm_list)
                
                # Calculate centrality
                centrality_scores = nx.eigenvector_centrality(subgraph, max_iter=1000)
                avg_centrality = sum(centrality_scores.values()) / len(comm_list)
                
                # Determine pattern type
                density = nx.density(subgraph)
                avg_weight = sum(d.get('weight', 1) for u, v, d in subgraph.edges(data=True)) / len(subgraph.edges) if subgraph.edges else 0
                
                if density > 0.6 and avg_weight > 0.7:
                    pattern_type = "highly_organized_ring"
                    risk_level = "Critical"
                    collusion_level = 0.9
                elif density > 0.4:
                    pattern_type = "loose_collaboration_ring"
                    risk_level = "High"
                    collusion_level = 0.7
                else:
                    pattern_type = "possible_coordination"
                    risk_level = "Medium"
                    collusion_level = 0.5
                
                # Gather evidence
                evidence = []
                for u, v, data in subgraph.edges(data=True):
                    evidence.append({
                        'type': 'similarity',
                        'between': [u, v],
                        'weight': data.get('weight', 0),
                        'details': f"Similarity score {data.get('weight', 0):.2%}"
                    })
                
                ring = PlagiarismRing(
                    id=f"ring_{len(rings):04d}",
                    members=comm_list,
                    documents=[],  # Will be populated from document associations
                    centrality_score=avg_centrality,
                    collusion_level=collusion_level,
                    pattern_type=pattern_type,
                    evidence=evidence,
                    risk_level=risk_level
                )
                
                rings.append(ring)
        
        return rings
    
    def detect_temporal_patterns(self) -> List[Dict]:
        """Detect temporal patterns (submission timing anomalies)"""
        temporal_patterns = []
        
        # Group by time windows (e.g., hour, day)
        submissions = defaultdict(list)
        
        for node in self.network.nodes.values():
            if node.type == NodeType.DOCUMENT:
                timestamp = node.attributes.get('submission_time')
                if timestamp:
                    if isinstance(timestamp, str):
                        timestamp = datetime.datetime.fromisoformat(timestamp)
                    
                    # Group by hour
                    hour_key = timestamp.strftime('%Y-%m-%d %H:00')
                    submissions[hour_key].append(node.id)
        
        # Detect unusual submission patterns
        for hour, docs in submissions.items():
            if len(docs) > 3:  # Threshold for suspicious concentration
                # Check if documents have high similarity
                similarities = []
                for i in range(len(docs)):
                    for j in range(i+1, len(docs)):
                        # Check for similarity edges between these documents
                        edge_found = False
                        for edge in self.network.edges:
                            if ((edge.source == docs[i] and edge.target == docs[j]) or
                                (edge.source == docs[j] and edge.target == docs[i])):
                                if edge.type == EdgeType.SIMILARITY:
                                    similarities.append(edge.weight)
                                    edge_found = True
                                    break
                        if not edge_found:
                            similarities.append(0.0)
                
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                
                if avg_similarity > 0.3:
                    temporal_patterns.append({
                        'time_window': hour,
                        'document_count': len(docs),
                        'avg_similarity': avg_similarity,
                        'documents': docs,
                        'suspicious': avg_similarity > 0.5,
                        'pattern_type': 'submission_clustering'
                    })
        
        return temporal_patterns
    
    def detect_source_sharing_networks(self) -> List[Dict]:
        """Detect networks of shared sources"""
        # Build bipartite graph of students and sources
        source_graph = nx.Graph()
        
        # Add student nodes
        for node in self.network.nodes.values():
            if node.type == NodeType.STUDENT:
                source_graph.add_node(node.id, type='student')
        
        # Add source nodes and connections
        for node in self.network.nodes.values():
            if node.type == NodeType.SOURCE:
                source_graph.add_node(node.id, type='source')
        
        # Add edges from documents to sources
        for edge in self.network.edges:
            if edge.type == EdgeType.SHARED_SOURCE:
                # Find the document this source belongs to
                doc_node = None
                for node in self.network.nodes.values():
                    if node.type == NodeType.DOCUMENT:
                        if node.attributes.get('source') == edge.source:
                            doc_node = node.id
                            break
                
                # Connect student to source through document
                if doc_node:
                    student_nodes = [n for n in self.network.graph.neighbors(doc_node)
                                   if self.network.nodes.get(n, Node).type == NodeType.STUDENT]
                    for student in student_nodes:
                        source_graph.add_edge(student, edge.source)
        
        # Find communities
        communities = list(nx.community.greedy_modularity_communities(source_graph))
        
        source_sharing_networks = []
        for comm in communities:
            comm_list = list(comm)
            students = [n for n in comm_list if source_graph.nodes[n].get('type') == 'student']
            sources = [n for n in comm_list if source_graph.nodes[n].get('type') == 'source']
            
            if len(students) >= 2 and len(sources) >= 2:
                subgraph = source_graph.subgraph(comm_list)
                density = nx.density(subgraph)
                
                source_sharing_networks.append({
                    'students': students,
                    'sources': sources,
                    'size': len(comm_list),
                    'density': density,
                    'sharing_pattern': 'multiple_common_sources' if len(sources) > len(students) / 2 else 'limited_sources',
                    'risk_score': density * 0.8 + (len(sources) / (len(students) + len(sources))) * 0.2
                })
        
        return source_sharing_networks
    
    def detect_pattern_of_changes(self) -> List[Dict]:
        """Detect similar patterns of changes across documents"""
        change_patterns = []
        
        # Get all document nodes with change histories
        documents = [n for n in self.network.nodes.values() 
                    if n.type == NodeType.DOCUMENT and 'changes' in n.attributes]
        
        for i in range(len(documents)):
            for j in range(i+1, len(documents)):
                doc1 = documents[i]
                doc2 = documents[j]
                
                changes1 = doc1.attributes.get('changes', [])
                changes2 = doc2.attributes.get('changes', [])
                
                if changes1 and changes2:
                    # Compare change patterns
                    similarity = self._compare_change_patterns(changes1, changes2)
                    
                    if similarity > 0.5:
                        change_patterns.append({
                            'document1': doc1.id,
                            'document2': doc2.id,
                            'similarity': similarity,
                            'common_changes': self._get_common_changes(changes1, changes2),
                            'suspicious': similarity > 0.7
                        })
        
        return change_patterns
    
    def _compare_change_patterns(self, changes1: List, changes2: List) -> float:
        """Compare two change histories for similarity"""
        if not changes1 or not changes2:
            return 0.0
        
        # Simple sequence similarity
        len_sim = 1 - abs(len(changes1) - len(changes2)) / max(len(changes1), len(changes2))
        
        # Compare change types
        types1 = [c.get('type', '') for c in changes1]
        types2 = [c.get('type', '') for c in changes2]
        
        common_types = set(types1) & set(types2)
        type_sim = len(common_types) / max(len(set(types1)), len(set(types2))) if max(len(set(types1)), len(set(types2))) > 0 else 0
        
        return (len_sim + type_sim) / 2
    
    def _get_common_changes(self, changes1: List, changes2: List) -> List:
        """Get common changes between two histories"""
        common = []
        for c1 in changes1:
            for c2 in changes2:
                if c1.get('type') == c2.get('type') and c1.get('description') == c2.get('description'):
                    common.append(c1)
                    break
        return common
    
    def detect_similarity_networks(self, min_similarity: float = 0.3) -> List[Dict]:
        """Detect networks of similar documents"""
        similarity_networks = []
        
        # Build similarity graph
        sim_graph = nx.Graph()
        for node in self.network.nodes.values():
            if node.type == NodeType.DOCUMENT:
                sim_graph.add_node(node.id)
        
        for edge in self.network.edges:
            if edge.type == EdgeType.SIMILARITY and edge.weight >= min_similarity:
                if edge.source in sim_graph and edge.target in sim_graph:
                    sim_graph.add_edge(edge.source, edge.target, weight=edge.weight)
        
        # Find connected components
        components = list(nx.connected_components(sim_graph))
        
        for comp in components:
            if len(comp) >= 2:
                subgraph = sim_graph.subgraph(comp)
                density = nx.density(subgraph)
                
                # Find clique-like structures
                cliques = list(nx.find_cliques(subgraph))
                max_clique_size = max(len(c) for c in cliques) if cliques else 0
                
                similarity_networks.append({
                    'documents': list(comp),
                    'size': len(comp),
                    'density': density,
                    'max_clique_size': max_clique_size,
                    'is_dense': density > 0.5,
                    'clique_count': len(cliques)
                })
        
        return similarity_networks
    
    def run_full_analysis(self) -> Dict:
        """Run all detection algorithms and return comprehensive results"""
        return {
            'collaboration_networks': self.detect_collaboration_networks(),
            'plagiarism_rings': [asdict(ring) for ring in self.detect_plagiarism_rings()],
            'temporal_patterns': self.detect_temporal_patterns(),
            'source_sharing_networks': self.detect_source_sharing_networks(),
            'change_patterns': self.detect_pattern_of_changes(),
            'similarity_networks': self.detect_similarity_networks()
        }

# ==================== Community Detection ====================

class CommunityDetector:
    """Detect communities in the plagiarism network"""
    
    def __init__(self, network: PlagiarismNetwork):
        self.network = network
        self.communities = []
    
    def detect_communities(self, algorithm: str = 'louvain') -> List[Community]:
        """Detect communities using various algorithms"""
        communities = []
        
        if algorithm == 'louvain':
            try:
                import community
                partition = community.best_partition(self.network.graph)
                
                # Group nodes by community
                comm_dict = defaultdict(list)
                for node, comm_id in partition.items():
                    comm_dict[comm_id].append(node)
                
                for comm_id, nodes in comm_dict.items():
                    if len(nodes) >= 3:
                        subgraph = self.network.graph.subgraph(nodes)
                        density = nx.density(subgraph)
                        
                        # Determine community type
                        node_types = [self.network.nodes[n].type for n in nodes]
                        if all(t == NodeType.STUDENT for t in node_types):
                            comm_type = "student_cluster"
                        elif all(t == NodeType.DOCUMENT for t in node_types):
                            comm_type = "document_cluster"
                        else:
                            comm_type = "mixed"
                        
                        # Calculate risk score
                        risk_score = 0.0
                        for edge in self.network.edges:
                            if edge.source in nodes and edge.target in nodes:
                                if edge.type == EdgeType.SIMILARITY:
                                    risk_score += edge.weight
                        
                        risk_score = min(1.0, risk_score / len(nodes))
                        
                        community = Community(
                            id=f"comm_{len(communities):04d}",
                            nodes=nodes,
                            size=len(nodes),
                            density=density,
                            type=comm_type,
                            risk_score=risk_score,
                            key_patterns=self._find_key_patterns(nodes)
                        )
                        communities.append(community)
                
            except ImportError:
                print("⚠️ python-louvain not installed. Install with: pip install python-louvain")
                # Fallback to simple connected components
                communities = self._simple_community_detection()
        
        self.communities = communities
        return communities
    
    def _simple_community_detection(self) -> List[Community]:
        """Simple community detection using connected components"""
        communities = []
        
        for component in nx.connected_components(self.network.graph):
            nodes = list(component)
            if len(nodes) >= 3:
                subgraph = self.network.graph.subgraph(nodes)
                density = nx.density(subgraph)
                
                community = Community(
                    id=f"comm_{len(communities):04d}",
                    nodes=nodes,
                    size=len(nodes),
                    density=density,
                    type="unknown",
                    risk_score=density * 0.5,
                    key_patterns=["connected_component"]
                )
                communities.append(community)
        
        return communities
    
    def _find_key_patterns(self, nodes: List[str]) -> List[str]:
        """Find key patterns in a community"""
        patterns = []
        
        # Check for high similarity
        similarity_count = 0
        for edge in self.network.edges:
            if edge.source in nodes and edge.target in nodes:
                if edge.type == EdgeType.SIMILARITY:
                    similarity_count += 1
        
        if similarity_count > len(nodes):
            patterns.append("high_similarity")
        
        # Check for dense collaboration
        student_count = sum(1 for n in nodes if self.network.nodes.get(n, Node).type == NodeType.STUDENT)
        if student_count > len(nodes) / 2:
            patterns.append("student_cluster")
        
        # Check for shared sources
        source_count = sum(1 for n in nodes if self.network.nodes.get(n, Node).type == NodeType.SOURCE)
        if source_count > 2:
            patterns.append("shared_sources")
        
        return patterns
    
    def get_community_summary(self) -> Dict:
        """Get summary of detected communities"""
        return {
            'total_communities': len(self.communities),
            'average_size': sum(c.size for c in self.communities) / len(self.communities) if self.communities else 0,
            'high_risk_communities': [c for c in self.communities if c.risk_score > 0.5],
            'community_types': Counter(c.type for c in self.communities)
        }

# ==================== Visualization ====================

class NetworkVisualizer:
    """Visualize the plagiarism network"""
    
    @staticmethod
    def generate_network_stats(network: PlagiarismNetwork) -> Dict:
        """Generate comprehensive network statistics"""
        stats = {
            'total_nodes': len(network.nodes),
            'total_edges': len(network.edges),
            'node_types': Counter(n.type.value for n in network.nodes.values()),
            'edge_types': Counter(e.type.value for e in network.edges),
            'avg_degree': sum(network.get_node_degree(n) for n in network.nodes) / len(network.nodes) if network.nodes else 0,
        }
        
        # Add graph metrics if graph is not empty
        if network.graph.nodes:
            stats.update({
                'density': nx.density(network.graph),
                'is_connected': nx.is_connected(network.graph) if network.graph.edges else False,
                'avg_clustering': nx.average_clustering(network.graph) if network.graph.edges else 0,
            })
            
            # Calculate diameter if graph is connected
            if stats['is_connected']:
                try:
                    stats['diameter'] = nx.diameter(network.graph)
                except:
                    stats['diameter'] = None
            
            # Centrality metrics
            if len(network.graph.nodes) > 1:
                try:
                    centrality = nx.eigenvector_centrality(network.graph, max_iter=1000)
                    stats['centrality_distribution'] = {
                        'min': min(centrality.values()),
                        'max': max(centrality.values()),
                        'mean': sum(centrality.values()) / len(centrality)
                    }
                except:
                    stats['centrality_distribution'] = None
        
        return stats
    
    @staticmethod
    def generate_subgraph_report(network: PlagiarismNetwork, node_ids: List[str]) -> Dict:
        """Generate report for a specific subgraph"""
        subgraph = network.get_subgraph(node_ids)
        
        return {
            'subgraph_size': len(subgraph.nodes),
            'subgraph_edges': len(subgraph.edges),
            'subgraph_density': nx.density(subgraph.graph) if subgraph.graph.nodes else 0,
            'node_details': [
                {
                    'id': nid,
                    'type': node.type.value,
                    'attributes': node.attributes
                }
                for nid, node in subgraph.nodes.items()
            ],
            'edge_details': [
                {
                    'source': e.source,
                    'target': e.target,
                    'type': e.type.value,
                    'weight': e.weight
                }
                for e in subgraph.edges
            ]
        }

# ==================== CLI Application ====================

class PlagiarismNetworkCLI:
    """Command-line interface for plagiarism network detection"""
    
    def __init__(self):
        self.network = PlagiarismNetwork()
        self.detector = AdvancedPatternDetector(self.network)
        self.community_detector = CommunityDetector(self.network)
        self.visualizer = NetworkVisualizer()
    
    def run(self):
        """Main application loop"""
        print("\n" + "=" * 60)
        print("🕸️ PLAGIARISM NETWORK & COLLABORATION DETECTION".center(60))
        print("=" * 60)
        
        while True:
            print("\n📋 MENU")
            print("-" * 40)
            print("1. 🏫 Add Student")
            print("2. 📄 Add Document")
            print("3. 🔗 Add Similarity Link")
            print("4. 🤝 Add Collaboration Link")
            print("5. 🔍 Run Network Analysis")
            print("6. 🎯 Detect Plagiarism Rings")
            print("7. 📊 View Network Statistics")
            print("8. 👥 Community Detection")
            print("9. 📈 Generate Report")
            print("10. 🚪 Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == "1":
                self._add_student()
            elif choice == "2":
                self._add_document()
            elif choice == "3":
                self._add_similarity()
            elif choice == "4":
                self._add_collaboration()
            elif choice == "5":
                self._run_analysis()
            elif choice == "6":
                self._detect_rings()
            elif choice == "7":
                self._view_stats()
            elif choice == "8":
                self._community_detection()
            elif choice == "9":
                self._generate_report()
            elif choice == "10":
                print("\n👋 Goodbye! Keeping academic integrity!")
                break
            else:
                print("❌ Invalid option")
    
    def _add_student(self):
        """Add a student node"""
        print("\n🏫 ADD STUDENT")
        name = input("Student name: ").strip()
        student_id = input("Student ID (optional): ").strip() or f"STU{random.randint(1000, 9999)}"
        institution = input("Institution: ").strip()
        
        node_id = self.network.add_node(
            NodeType.STUDENT,
            attributes={
                'name': name,
                'student_id': student_id,
                'institution': institution
            },
            metadata={'added_by': 'cli'}
        )
        
        print(f"✅ Student added! ID: {node_id}")
    
    def _add_document(self):
        """Add a document node"""
        print("\n📄 ADD DOCUMENT")
        title = input("Document title: ").strip()
        author_id = input("Student ID (author): ").strip()
        
        if author_id not in self.network.nodes:
            print("❌ Student not found. Please add student first.")
            return
        
        doc_id = input("Document ID (optional): ").strip() or f"DOC{random.randint(1000, 9999)}"
        content_length = input("Content length (characters): ").strip()
        submission_time = input("Submission time (YYYY-MM-DD HH:MM): ").strip()
        
        node_id = self.network.add_node(
            NodeType.DOCUMENT,
            attributes={
                'title': title,
                'doc_id': doc_id,
                'author': author_id,
                'content_length': int(content_length) if content_length else 0,
                'submission_time': submission_time
            }
        )
        
        # Link student to document
        self.network.add_edge(author_id, node_id, EdgeType.CO_AUTHORSHIP, weight=1.0)
        
        print(f"✅ Document added! ID: {node_id}")
    
    def _add_similarity(self):
        """Add similarity link between documents"""
        print("\n🔗 ADD SIMILARITY LINK")
        doc1 = input("First document ID: ").strip()
        doc2 = input("Second document ID: ").strip()
        
        if doc1 not in self.network.nodes or doc2 not in self.network.nodes:
            print("❌ One or both documents not found")
            return
        
        similarity = float(input("Similarity score (0-1): ").strip() or 0.5)
        similarity = max(0, min(1, similarity))
        
        self.network.add_edge(doc1, doc2, EdgeType.SIMILARITY, weight=similarity)
        print(f"✅ Similarity link added! Score: {similarity:.2%}")
    
    def _add_collaboration(self):
        """Add collaboration link between students"""
        print("\n🤝 ADD COLLABORATION LINK")
        student1 = input("First student ID: ").strip()
        student2 = input("Second student ID: ").strip()
        
        if student1 not in self.network.nodes or student2 not in self.network.nodes:
            print("❌ One or both students not found")
            return
        
        weight = float(input("Collaboration strength (0-1, default 0.5): ").strip() or 0.5)
        weight = max(0, min(1, weight))
        
        self.network.add_edge(student1, student2, EdgeType.CO_AUTHORSHIP, weight=weight)
        print(f"✅ Collaboration link added! Strength: {weight:.2%}")
    
    def _run_analysis(self):
        """Run full network analysis"""
        print("\n🔍 RUNNING NETWORK ANALYSIS...")
        
        analysis = self.detector.run_full_analysis()
        
        print("\n📊 ANALYSIS RESULTS")
        print("-" * 40)
        
        # Collaboration networks
        collab = analysis['collaboration_networks']
        print(f"\n🤝 Collaboration Networks: {len(collab)}")
        for net in collab[:5]:
            print(f"  • {net['size']} members, Density: {net['density']:.2%}")
            if net['is_suspicious']:
                print(f"    ⚠️ Suspicious pattern detected!")
        
        # Plagiarism rings
        rings = analysis['plagiarism_rings']
        print(f"\n🎯 Plagiarism Rings: {len(rings)}")
        for ring in rings[:5]:
            print(f"  • {ring['pattern_type']} with {len(ring['members'])} members")
            print(f"    Risk Level: {ring['risk_level']}")
        
        # Temporal patterns
        temporal = analysis['temporal_patterns']
        print(f"\n⏰ Temporal Patterns: {len(temporal)}")
        for pattern in temporal[:5]:
            if pattern['suspicious']:
                print(f"  • ⚠️ Suspicious pattern at {pattern['time_window']}")
                print(f"    {pattern['document_count']} documents, {pattern['avg_similarity']:.2%} similarity")
        
        # Source sharing
        source_sharing = analysis['source_sharing_networks']
        print(f"\n📚 Source Sharing Networks: {len(source_sharing)}")
        for net in source_sharing[:5]:
            print(f"  • {len(net['students'])} students, {len(net['sources'])} sources")
            print(f"    Risk Score: {net['risk_score']:.2%}")
        
        # Similarity networks
        sim_networks = analysis['similarity_networks']
        print(f"\n📊 Similarity Networks: {len(sim_networks)}")
        for net in sim_networks[:5]:
            print(f"  • {net['size']} documents, Density: {net['density']:.2%}")
            if net['is_dense']:
                print(f"    ⚠️ Dense similarity network!")
    
    def _detect_rings(self):
        """Detect plagiarism rings"""
        print("\n🎯 DETECTING PLAGIARISM RINGS...")
        
        rings = self.detector.detect_plagiarism_rings(min_members=2)
        
        if not rings:
            print("✅ No plagiarism rings detected")
            return
        
        print(f"\n🚨 Found {len(rings)} potential plagiarism rings:")
        
        for ring in rings:
            print(f"\n🔹 Ring {ring.id} - {ring.pattern_type}")
            print(f"   Members: {len(ring.members)}")
            print(f"   Collusion Level: {ring.collusion_level:.2%}")
            print(f"   Risk Level: {ring.risk_level}")
            print(f"   Centrality: {ring.centrality_score:.3f}")
            
            # Show members
            member_names = []
            for member_id in ring.members[:10]:
                node = self.network.get_node(member_id)
                if node and node.type == NodeType.STUDENT:
                    name = node.attributes.get('name', member_id)
                    member_names.append(name)
            
            if member_names:
                print(f"   Members: {', '.join(member_names[:5])}")
                if len(member_names) > 5:
                    print(f"   ... and {len(member_names) - 5} more")
            
            # Show evidence
            if ring.evidence:
                print(f"   Evidence: {len(ring.evidence)} items")
                for ev in ring.evidence[:2]:
                    print(f"     • {ev['details']}")
    
    def _view_stats(self):
        """View network statistics"""
        print("\n📊 NETWORK STATISTICS")
        print("-" * 40)
        
        stats = self.visualizer.generate_network_stats(self.network)
        
        print(f"\n📈 Basic Statistics:")
        print(f"  • Total Nodes: {stats['total_nodes']}")
        print(f"  • Total Edges: {stats['total_edges']}")
        print(f"  • Average Degree: {stats['avg_degree']:.2f}")
        
        print(f"\n📂 Node Types:")
        for node_type, count in stats['node_types'].items():
            print(f"  • {node_type}: {count}")
        
        print(f"\n🔗 Edge Types:")
        for edge_type, count in stats['edge_types'].items():
            print(f"  • {edge_type}: {count}")
        
        if stats.get('density') is not None:
            print(f"\n📐 Graph Metrics:")
            print(f"  • Density: {stats['density']:.3f}")
            print(f"  • Connected: {stats['is_connected']}")
            if stats.get('diameter'):
                print(f"  • Diameter: {stats['diameter']}")
            if stats.get('avg_clustering'):
                print(f"  • Avg Clustering: {stats['avg_clustering']:.3f}")
    
    def _community_detection(self):
        """Run community detection"""
        print("\n👥 COMMUNITY DETECTION")
        print("-" * 40)
        
        algorithm = input("Algorithm (louvain/simple, default louvain): ").strip() or 'louvain'
        communities = self.community_detector.detect_communities(algorithm)
        
        if not communities:
            print("❌ No communities detected")
            return
        
        summary = self.community_detector.get_community_summary()
        
        print(f"\n📊 Community Summary:")
        print(f"  • Total Communities: {summary['total_communities']}")
        print(f"  • Average Size: {summary['average_size']:.1f}")
        
        print(f"\n🎯 Community Types:")
        for comm_type, count in summary['community_types'].items():
            print(f"  • {comm_type}: {count}")
        
        if summary['high_risk_communities']:
            print(f"\n⚠️ High-Risk Communities: {len(summary['high_risk_communities'])}")
            for comm in summary['high_risk_communities']:
                print(f"  • Community {comm.id}: {comm.size} members, Risk Score: {comm.risk_score:.2%}")
    
    def _generate_report(self):
        """Generate comprehensive report"""
        print("\n📈 GENERATING REPORT...")
        
        report = {
            'generated_at': datetime.datetime.now().isoformat(),
            'network_stats': self.visualizer.generate_network_stats(self.network),
            'analysis': self.detector.run_full_analysis(),
            'communities': self.community_detector.get_community_summary()
        }
        
        filename = f"plagiarism_network_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report saved to {filename}")
        
        # Generate readable summary
        print("\n📋 REPORT SUMMARY")
        print("-" * 40)
        
        print(f"\n🔹 Network Size:")
        print(f"   • {report['network_stats']['total_nodes']} nodes, {report['network_stats']['total_edges']} edges")
        
        print(f"\n🔹 Plagiarism Rings: {len(report['analysis']['plagiarism_rings'])}")
        for ring in report['analysis']['plagiarism_rings'][:3]:
            print(f"   • {ring['pattern_type']} - {ring['risk_level']} risk")
        
        print(f"\n🔹 Communities: {report['communities']['total_communities']}")
        if report['communities']['high_risk_communities']:
            print(f"   ⚠️ {len(report['communities']['high_risk_communities'])} high-risk communities")
        
        print(f"\n🔹 Collaboration Networks: {len(report['analysis']['collaboration_networks'])}")
        suspicious = [n for n in report['analysis']['collaboration_networks'] if n.get('is_suspicious', False)]
        if suspicious:
            print(f"   ⚠️ {len(suspicious)} suspicious collaboration networks")

# ==================== Main Execution ====================

def main():
    """Main function to run the plagiarism network detection system"""
    cli = PlagiarismNetworkCLI()
    
    # Check for optional dependencies
    try:
        import networkx
        print("✅ NetworkX available")
    except ImportError:
        print("❌ NetworkX not installed. Install with: pip install networkx")
        return
    
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Upholding academic integrity!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
