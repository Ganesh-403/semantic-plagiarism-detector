# ───────────────────────────────────────────────────────────────────────────────
# ── SECTION: ADVANCED CLUSTERING AND TOPIC ANALYSIS (Issue #1984) ──────────
# ───────────────────────────────────────────────────────────────────────────────

# ── Imports for Clustering and Topic Analysis ─────────────────────────────
try:
    import scipy.cluster.hierarchy as sch
    from scipy.cluster.hierarchy import fcluster
    from scipy.spatial.distance import squareform
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.manifold import TSNE
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    CLUSTERING_AVAILABLE = True
except ImportError as e:
    CLUSTERING_AVAILABLE = False
    logger.warning(f"Clustering dependencies not available: {e}")


# ── Clustering Engine ──────────────────────────────────────────────────────
class SemanticClusterer:
    """
    Perform hierarchical and k-means clustering on document similarity matrices.
    Provides automatic optimal cluster detection and quality metrics.
    """

    def __init__(self, linkage: str = "ward", metric: str = "precomputed"):
        """
        Initialize the clusterer.

        Args:
            linkage: Linkage criterion for hierarchical clustering
            metric: Distance metric (use 'precomputed' for similarity matrices)
        """
        self.linkage = linkage
        self.metric = metric
        self.model = None
        self.labels = None
        self.distance_matrix = None
        self.cluster_metrics = {}
        self.is_fitted = False

    def fit_hierarchical(
        self, similarity_matrix: np.ndarray, n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit hierarchical clustering model.

        Args:
            similarity_matrix: Square similarity matrix
            n_clusters: Number of clusters (auto-detect if None)

        Returns:
            Cluster labels array
        """
        # Convert similarity to distance
        self.distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(self.distance_matrix, 0)

        # If n_clusters is None, use silhouette analysis to find optimal
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(
                self.distance_matrix,
                min_clusters=2,
                max_clusters=min(10, len(similarity_matrix) - 1),
            )

        # Fit hierarchical clustering
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage=self.linkage
        )
        self.labels = self.model.fit_predict(self.distance_matrix)
        self.is_fitted = True

        # Compute cluster metrics
        self._compute_metrics(similarity_matrix)

        return self.labels

    def fit_kmeans(
        self, embeddings: np.ndarray, n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit k-means clustering on embeddings.

        Args:
            embeddings: Document embedding matrix
            n_clusters: Number of clusters (auto-detect if None)

        Returns:
            Cluster labels array
        """
        # Standardize embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)

        # If n_clusters is None, use elbow method to find optimal
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters_kmeans(
                scaled_embeddings,
                min_clusters=2,
                max_clusters=min(10, len(embeddings) - 1),
            )

        # Fit k-means
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.model.fit_predict(scaled_embeddings)
        self.is_fitted = True

        return self.labels

    def _find_optimal_clusters(
        self, distance_matrix: np.ndarray, min_clusters: int = 2, max_clusters: int = 10
    ) -> int:
        """
        Find optimal number of clusters using silhouette score.
        """
        best_score = -1
        best_n = min_clusters

        for n in range(min_clusters, max_clusters + 1):
            try:
                clusterer = AgglomerativeClustering(
                    n_clusters=n, metric="precomputed", linkage=self.linkage
                )
                labels = clusterer.fit_predict(distance_matrix)

                # Silhouette score on distance matrix
                score = silhouette_score(distance_matrix, labels, metric="precomputed")

                if score > best_score:
                    best_score = score
                    best_n = n
            except Exception as e:
                logger.warning(f"Failed to compute silhouette for n={n}: {e}")
                continue

        return best_n

    def _find_optimal_clusters_kmeans(
        self, embeddings: np.ndarray, min_clusters: int = 2, max_clusters: int = 10
    ) -> int:
        """
        Find optimal number of clusters using elbow method.
        """
        inertias = []

        for n in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)

        # Find elbow point using curvature
        if len(inertias) < 3:
            return min_clusters

        # Calculate second derivative to find elbow
        diffs = np.diff(inertias)
        diffs2 = np.diff(diffs)
        elbow_idx = np.argmin(diffs2) + 1

        return min_clusters + elbow_idx

    def _compute_metrics(self, similarity_matrix: np.ndarray):
        """
        Compute cluster quality metrics.
        """
        if not self.is_fitted or self.labels is None:
            return

        n_clusters = len(np.unique(self.labels))
        n_samples = len(self.labels)

        # Silhouette score (using similarity matrix)
        try:
            distance_matrix = 1 - similarity_matrix
            np.fill_diagonal(distance_matrix, 0)
            sil_score = silhouette_score(
                distance_matrix, self.labels, metric="precomputed"
            )
        except Exception:
            sil_score = 0.0

        # Davies-Bouldin score
        try:
            db_score = davies_bouldin_score(similarity_matrix, self.labels)
        except Exception:
            db_score = 0.0

        # Intra-cluster similarity
        cluster_sims = []
        cluster_sizes = []

        for cluster_id in range(n_clusters):
            indices = np.where(self.labels == cluster_id)[0]
            cluster_sizes.append(len(indices))

            if len(indices) > 1:
                cluster_mat = similarity_matrix[np.ix_(indices, indices)]
                # Average similarity within cluster (excluding diagonal)
                mean_sim = (np.sum(cluster_mat) - len(indices)) / (
                    len(indices) * (len(indices) - 1)
                )
                cluster_sims.append(mean_sim)
            else:
                cluster_sims.append(1.0)

        # Inter-cluster similarity
        inter_cluster_sims = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                indices_i = np.where(self.labels == i)[0]
                indices_j = np.where(self.labels == j)[0]
                if len(indices_i) > 0 and len(indices_j) > 0:
                    inter_sim = np.mean(similarity_matrix[np.ix_(indices_i, indices_j)])
                    inter_cluster_sims.append(inter_sim)

        self.cluster_metrics = {
            "n_clusters": n_clusters,
            "n_samples": n_samples,
            "silhouette_score": sil_score,
            "davies_bouldin_score": db_score,
            "cluster_sizes": cluster_sizes,
            "intra_cluster_similarities": cluster_sims,
            "average_intra_similarity": np.mean(cluster_sims) if cluster_sims else 0.0,
            "average_inter_similarity": np.mean(inter_cluster_sims)
            if inter_cluster_sims
            else 0.0,
            "cluster_separation": np.mean(inter_cluster_sims) - np.mean(cluster_sims)
            if inter_cluster_sims
            else 0.0,
        }

    def get_cluster_membership(self, document_names: List[str]) -> Dict[int, List[str]]:
        """
        Get documents grouped by cluster.
        """
        if not self.is_fitted or self.labels is None:
            return {}

        cluster_members = {}
        for idx, label in enumerate(self.labels):
            if label not in cluster_members:
                cluster_members[label] = []
            if idx < len(document_names):
                cluster_members[label].append(document_names[idx])

        return cluster_members

    def get_suspicious_clusters(self, similarity_threshold: float = 0.70) -> List[int]:
        """
        Identify clusters with unusually high internal similarity (potential collusion).
        """
        if not self.is_fitted:
            return []

        suspicious = []
        for cluster_id, intra_sim in enumerate(
            self.cluster_metrics.get("intra_cluster_similarities", [])
        ):
            if (
                intra_sim > similarity_threshold
                and self.cluster_metrics["cluster_sizes"][cluster_id] >= 2
            ):
                suspicious.append(cluster_id)

        return suspicious

    def get_cluster_centroids(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Compute cluster centroids as average similarity profiles.
        """
        if not self.is_fitted or self.labels is None:
            return np.array([])

        n_clusters = len(np.unique(self.labels))
        centroids = np.zeros((n_clusters, similarity_matrix.shape[1]))

        for cluster_id in range(n_clusters):
            indices = np.where(self.labels == cluster_id)[0]
            if len(indices) > 0:
                centroids[cluster_id] = np.mean(similarity_matrix[indices, :], axis=0)

        return centroids


# ── Topic Modeling Engine ─────────────────────────────────────────────────
class TopicExtractor:
    """
    Extract latent topics from document chunks using NMF or LDA.
    """

    def __init__(
        self,
        n_topics: int = 10,
        max_features: int = 1000,
        method: str = "nmf",
        random_state: int = 42,
    ):
        """
        Initialize topic extractor.

        Args:
            n_topics: Number of topics to extract
            max_features: Maximum vocabulary size
            method: 'nmf' or 'lda'
            random_state: Random seed for reproducibility
        """
        self.n_topics = n_topics
        self.max_features = max_features
        self.method = method
        self.random_state = random_state
        self.vectorizer = None
        self.model = None
        self.doc_topic_dist = None
        self.topic_term_dist = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, documents: List[str]) -> np.ndarray:
        """
        Fit topic model on documents.

        Args:
            documents: List of document texts

        Returns:
            Document-topic distribution matrix
        """
        # Vectorize documents
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, stop_words="english", min_df=2, max_df=0.9
        )
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Fit topic model
        if self.method == "nmf":
            self.model = NMF(
                n_components=self.n_topics,
                random_state=self.random_state,
                init="nndsvd",
                beta_loss="frobenius",
                solver="cd",
            )
            self.doc_topic_dist = self.model.fit_transform(doc_term_matrix)
            self.topic_term_dist = self.model.components_
        elif self.method == "lda":
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=self.random_state,
                max_iter=100,
                learning_method="online",
            )
            self.doc_topic_dist = self.model.fit_transform(doc_term_matrix)
            self.topic_term_dist = self.model.components_
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted = True
        return self.doc_topic_dist

    def get_top_terms(self, n_terms: int = 10) -> Dict[int, List[str]]:
        """
        Get top terms for each topic.
        """
        if not self.is_fitted:
            return {}

        topic_terms = {}
        for topic_idx in range(self.n_topics):
            # Get term scores for this topic
            if self.method == "nmf":
                topic_weights = self.topic_term_dist[topic_idx, :]
            else:  # LDA
                topic_weights = self.topic_term_dist[topic_idx, :]

            # Get top terms by score
            top_indices = np.argsort(topic_weights)[-n_terms:][::-1]
            top_terms = [self.feature_names[idx] for idx in top_indices]
            top_scores = [topic_weights[idx] for idx in top_indices]

            topic_terms[topic_idx] = list(zip(top_terms, top_scores))

        return topic_terms

    def get_document_topics(
        self, threshold: float = 0.2
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get dominant topics for each document.
        """
        if not self.is_fitted or self.doc_topic_dist is None:
            return {}

        doc_topics = {}
        for doc_idx, doc_dist in enumerate(self.doc_topic_dist):
            # Get topics above threshold
            topics = [
                (topic_idx, float(score))
                for topic_idx, score in enumerate(doc_dist)
                if score > threshold
            ]
            doc_topics[f"Document_{doc_idx}"] = sorted(
                topics, key=lambda x: x[1], reverse=True
            )

        return doc_topics

    def get_topic_coherence(self, documents: List[str]) -> float:
        """
        Calculate average topic coherence (quality metric).
        """
        if not self.is_fitted:
            return 0.0

        try:
            # Simplified coherence: average pairwise similarity of top terms
            top_terms = self.get_top_terms(n_terms=10)

            # Vectorize terms
            term_vectors = {}
            for topic_idx, terms in top_terms.items():
                term_list = [t[0] for t in terms]
                # Create a set of terms for this topic
                term_vectors[topic_idx] = set(term_list)

            # Calculate Jaccard similarity between topics (lower is better)
            coherence_scores = []
            topic_keys = list(term_vectors.keys())
            for i in range(len(topic_keys)):
                for j in range(i + 1, len(topic_keys)):
                    set_i = term_vectors[topic_keys[i]]
                    set_j = term_vectors[topic_keys[j]]
                    intersection = len(set_i.intersection(set_j))
                    union = len(set_i.union(set_j))
                    if union > 0:
                        coherence_scores.append(1 - (intersection / union))

            return np.mean(coherence_scores) if coherence_scores else 0.0
        except Exception:
            return 0.0


# ── Pattern Evolution Tracker ─────────────────────────────────────────────
class PatternEvolutionTracker:
    """
    Track how plagiarism patterns evolve across multiple scans.
    """

    def __init__(self, max_history: int = 100):
        """
        Initialize pattern evolution tracker.

        Args:
            max_history: Maximum number of historical scans to keep
        """
        self.history = []
        self.max_history = max_history
        self.patterns = {}

    def add_scan(
        self,
        similarity_matrix: np.ndarray,
        document_names: List[str],
        timestamp: Optional[datetime] = None,
    ):
        """
        Add a new scan to the history.
        """
        if timestamp is None:
            timestamp = datetime.now()

        scan_data = {
            "timestamp": timestamp,
            "similarity_matrix": similarity_matrix,
            "document_names": document_names,
            "n_documents": len(document_names),
            "avg_similarity": np.mean(
                similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            ),
            "max_similarity": np.max(
                similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            ),
        }

        self.history.append(scan_data)

        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        # Detect patterns from this scan
        self._detect_patterns(scan_data)

    def _detect_patterns(self, scan_data: Dict):
        """
        Detect patterns in a scan.
        """
        sim_matrix = scan_data["similarity_matrix"]
        names = scan_data["document_names"]

        # Find document pairs with high similarity
        high_sim_pairs = []
        n = len(names)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > 0.70:  # High similarity threshold
                    high_sim_pairs.append((names[i], names[j], float(sim_matrix[i, j])))

        if high_sim_pairs:
            pattern_key = f"high_sim_pairs_{scan_data['timestamp'].strftime('%Y%m%d')}"
            self.patterns[pattern_key] = {
                "timestamp": scan_data["timestamp"],
                "pairs": high_sim_pairs,
                "n_pairs": len(high_sim_pairs),
            }

    def get_evolution_data(self) -> pd.DataFrame:
        """
        Get evolution data as DataFrame.
        """
        if not self.history:
            return pd.DataFrame()

        data = []
        for scan in self.history:
            data.append(
                {
                    "timestamp": scan["timestamp"],
                    "n_documents": scan["n_documents"],
                    "avg_similarity": scan["avg_similarity"],
                    "max_similarity": scan["max_similarity"],
                }
            )

        return pd.DataFrame(data)

    def get_emerging_patterns(self, lookback_days: int = 7) -> List[Dict]:
        """
        Identify emerging plagiarism patterns.
        """
        if not self.history:
            return []

        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_patterns = [p for p in self.patterns.values() if p["timestamp"] > cutoff]

        # Group by document pairs
        pair_counts = {}
        for pattern in recent_patterns:
            for doc_a, doc_b, sim in pattern["pairs"]:
                key = tuple(sorted([doc_a, doc_b]))
                if key not in pair_counts:
                    pair_counts[key] = {"count": 0, "sims": [], "timestamps": []}
                pair_counts[key]["count"] += 1
                pair_counts[key]["sims"].append(sim)
                pair_counts[key]["timestamps"].append(pattern["timestamp"])

        # Filter to emerging patterns (appearing multiple times)
        emerging = []
        for (doc_a, doc_b), data in pair_counts.items():
            if data["count"] >= 2:  # Appeared in multiple scans
                emerging.append(
                    {
                        "document_a": doc_a,
                        "document_b": doc_b,
                        "frequency": data["count"],
                        "avg_similarity": np.mean(data["sims"]),
                        "first_seen": min(data["timestamps"]),
                        "last_seen": max(data["timestamps"]),
                    }
                )

        return sorted(emerging, key=lambda x: x["frequency"], reverse=True)


# ── Visualization Functions for Clustering ──────────────────────────────


def plot_cluster_dendrogram(
    distance_matrix: np.ndarray,
    document_names: List[str],
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Generate dendrogram for hierarchical clustering visualization.
    """
    if not CLUSTERING_AVAILABLE:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "Clustering dependencies not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Convert to condensed distance matrix
    condensed_dist = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = sch.linkage(condensed_dist, method="ward")

    # Create dendrogram
    fig, ax = plt.subplots(figsize=figsize)

    # Plot dendrogram with labels
    dendro = sch.dendrogram(
        linkage_matrix,
        labels=document_names,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=10,
        orientation="top",
    )

    ax.set_title("Document Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Documents")
    ax.set_ylabel("Distance")
    plt.tight_layout()

    return fig


def plot_cluster_scatter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    document_names: List[str],
    method: str = "tsne",
    perplexity: int = 30,
) -> plt.Figure:
    """
    Generate 2D projection of documents with cluster coloring.
    """
    if not CLUSTERING_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(
            0.5,
            0.5,
            "Clustering dependencies not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Reduce dimensionality
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:  # PCA
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)

    coords = reducer.fit_transform(embeddings)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by cluster
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=100, alpha=0.7
    )

    # Add labels
    for i, name in enumerate(document_names):
        truncated = name[:20] + "..." if len(name) > 20 else name
        ax.annotate(
            truncated,
            (coords[i, 0], coords[i, 1]),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_title(f"Document Clusters ({method.upper()} Projection)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()

    return fig


def plot_cluster_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    document_names: List[str],
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Generate heatmap of similarity matrix with cluster coloring.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by cluster labels
    sorted_indices = np.argsort(labels)
    sorted_matrix = similarity_matrix[np.ix_(sorted_indices, sorted_indices)]
    sorted_names = [document_names[i] for i in sorted_indices]

    # Plot heatmap
    im = ax.imshow(sorted_matrix, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    # Set labels
    ax.set_xticks(range(len(sorted_names)))
    ax.set_yticks(range(len(sorted_names)))
    ax.set_xticklabels([name[:15] for name in sorted_names], rotation=90, fontsize=8)
    ax.set_yticklabels([name[:15] for name in sorted_names], fontsize=8)

    # Add cluster separation lines
    cluster_boundaries = []
    current_label = labels[sorted_indices[0]]
    for i, label in enumerate(labels[sorted_indices]):
        if label != current_label:
            cluster_boundaries.append(i)
            current_label = label

    for boundary in cluster_boundaries:
        ax.axhline(y=boundary - 0.5, color="black", linewidth=2)
        ax.axvline(x=boundary - 0.5, color="black", linewidth=2)

    ax.set_title("Document Similarity Matrix (Sorted by Clusters)")
    plt.colorbar(im, ax=ax, label="Similarity")
    plt.tight_layout()

    return fig


# ── UI Rendering Functions ──────────────────────────────────────────────


def render_clustering_tab(
    similarity_matrix: np.ndarray, document_names: List[str], embeddings: np.ndarray
):
    """
    Render the clustering analysis tab.
    """
    st.subheader("🔬 Semantic Document Clustering")

    if not CLUSTERING_AVAILABLE:
        st.error(
            "Clustering dependencies not available. Please install scikit-learn and scipy."
        )
        st.info("Run: pip install scikit-learn scipy umap-learn")
        return

    if len(document_names) < 3:
        st.info("At least 3 documents are required for clustering analysis.")
        return

    # Initialize clusterer
    clusterer = SemanticClusterer(linkage="ward")

    # Clustering options
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.number_input(
            "Number of Clusters (0 for auto-detect)",
            min_value=0,
            max_value=min(20, len(document_names) - 1),
            value=0,
            help="Set to 0 to automatically detect optimal number",
        )
        n_clusters = None if n_clusters == 0 else n_clusters

    with col2:
        linkage_method = st.selectbox(
            "Linkage Criterion",
            options=["ward", "complete", "average", "single"],
            index=0,
        )
        clusterer.linkage = linkage_method

    # Run clustering
    if st.button("🔬 Run Clustering Analysis", type="primary"):
        with st.spinner("Performing hierarchical clustering..."):
            # Fit clustering
            labels = clusterer.fit_hierarchical(
                similarity_matrix, n_clusters=n_clusters
            )

            # Store in session state
            st.session_state["cluster_labels"] = labels
            st.session_state["clusterer"] = clusterer

            # Display metrics
            st.success("✅ Clustering completed!")

            # Metrics dashboard
            metrics = clusterer.cluster_metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Number of Clusters", metrics["n_clusters"])
            col2.metric("Silhouette Score", f"{metrics['silhouette_score']:.3f}")
            col3.metric(
                "Davies-Bouldin Score", f"{metrics['davies_bouldin_score']:.3f}"
            )
            col4.metric(
                "Avg Intra-Cluster Similarity",
                f"{metrics['average_intra_similarity']:.3f}",
            )

            # Cluster sizes
            st.subheader("📊 Cluster Sizes")
            cluster_sizes = metrics["cluster_sizes"]
            size_data = pd.DataFrame(
                {
                    "Cluster": [f"Cluster {i}" for i in range(len(cluster_sizes))],
                    "Documents": cluster_sizes,
                }
            )
            st.bar_chart(size_data.set_index("Cluster"))

            # Suspicious clusters
            suspicious = clusterer.get_suspicious_clusters(similarity_threshold=0.70)
            if suspicious:
                st.warning(f"🚨 **{len(suspicious)} suspicious clusters detected**")
                for cluster_id in suspicious:
                    cluster_members = clusterer.get_cluster_membership(document_names)[
                        cluster_id
                    ]
                    st.markdown(
                        f"**Cluster {cluster_id}** ({len(cluster_members)} docs): {', '.join(cluster_members[:5])}{'...' if len(cluster_members) > 5 else ''}"
                    )
            else:
                st.success("✅ No suspicious clusters detected")

            # Visualizations
            st.subheader("📈 Cluster Visualizations")
            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                # Dendrogram
                if st.button("📊 Show Dendrogram", key="show_dendrogram"):
                    fig = plot_cluster_dendrogram(
                        clusterer.distance_matrix, document_names
                    )
                    st.pyplot(fig, use_container_width=True)

            with viz_col2:
                # Scatter plot
                if st.button("🔄 Show 2D Projection", key="show_scatter"):
                    if embeddings is not None and len(embeddings) > 0:
                        fig = plot_cluster_scatter(
                            embeddings, labels, document_names, method="tsne"
                        )
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.warning("Embeddings not available for projection")

            # Cluster similarity heatmap
            if st.button(
                "🗺️ Show Cluster Similarity Heatmap", key="show_cluster_heatmap"
            ):
                fig = plot_cluster_similarity_heatmap(
                    similarity_matrix, labels, document_names
                )
                st.pyplot(fig, use_container_width=True)

            # Cluster membership table
            with st.expander("📋 Full Cluster Membership"):
                membership = clusterer.get_cluster_membership(document_names)
                membership_data = []
                for cluster_id, docs in membership.items():
                    for doc in docs:
                        membership_data.append(
                            {"Cluster": f"Cluster {cluster_id}", "Document": doc}
                        )
                st.dataframe(pd.DataFrame(membership_data))


def render_topic_analysis_tab(document_chunks: Dict[str, List[str]]):
    """
    Render the topic analysis tab.
    """
    st.subheader("📚 Topic Analysis")

    if not CLUSTERING_AVAILABLE:
        st.error(
            "Topic modeling dependencies not available. Please install scikit-learn."
        )
        return

    if not document_chunks:
        st.info("No document chunks available for topic analysis.")
        return

    # Prepare documents
    all_chunks = []
    for doc_name, chunks in document_chunks.items():
        all_chunks.extend(chunks)

    if len(all_chunks) < 10:
        st.info("At least 10 text chunks are required for meaningful topic analysis.")
        return

    # Topic extraction options
    col1, col2, col3 = st.columns(3)
    with col1:
        n_topics = st.slider(
            "Number of Topics", min_value=3, max_value=20, value=8, step=1
        )
    with col2:
        max_features = st.slider(
            "Vocabulary Size", min_value=500, max_value=3000, value=1000, step=100
        )
    with col3:
        method = st.selectbox(
            "Model Method",
            options=["nmf", "lda"],
            index=0,
            format_func=lambda x: x.upper(),
        )

    if st.button("📊 Extract Topics", type="primary"):
        with st.spinner("Extracting topics from document chunks..."):
            # Initialize topic extractor
            extractor = TopicExtractor(
                n_topics=n_topics, max_features=max_features, method=method
            )

            # Fit model
            doc_topic_dist = extractor.fit(all_chunks)

            # Store in session state
            st.session_state["topic_extractor"] = extractor

            # Display results
            st.success("✅ Topic extraction completed!")

            # Topic coherence
            coherence = extractor.get_topic_coherence(all_chunks)
            st.metric(
                "Topic Coherence Score",
                f"{coherence:.3f}",
                help="Higher is better (max 1.0)",
            )

            # Top terms per topic
            st.subheader("📝 Top Terms per Topic")
            top_terms = extractor.get_top_terms(n_terms=10)

            topic_cols = st.columns(min(4, n_topics))
            for topic_idx, (col) in enumerate(topic_cols):
                if topic_idx < n_topics:
                    with col:
                        st.markdown(f"**Topic {topic_idx + 1}**")
                        terms = top_terms.get(topic_idx, [])
                        for term, score in terms[:5]:
                            st.markdown(f"- {term} ({score:.3f})")

            # Topic distribution bar chart
            st.subheader("📊 Topic Distribution")

            # Aggregate topic distribution across documents
            topic_weights = np.mean(doc_topic_dist, axis=0)
            topic_data = pd.DataFrame(
                {
                    "Topic": [f"Topic {i + 1}" for i in range(n_topics)],
                    "Weight": topic_weights,
                }
            )
            st.bar_chart(topic_data.set_index("Topic"))

            # Document-topic matrix
            with st.expander("📋 Document-Topic Distribution"):
                # Get document topics
                doc_topics = extractor.get_document_topics(threshold=0.1)

                # Create matrix
                doc_topic_matrix = []
                doc_names = list(document_chunks.keys())
                for idx, (doc_name, topics) in enumerate(doc_topics.items()):
                    if idx < len(doc_names):
                        row = {"Document": doc_names[idx]}
                        for topic_idx in range(n_topics):
                            row[f"Topic_{topic_idx + 1}"] = 0.0
                        for topic_idx, score in topics:
                            row[f"Topic_{topic_idx + 1}"] = score
                        doc_topic_matrix.append(row)

                st.dataframe(pd.DataFrame(doc_topic_matrix), use_container_width=True)


def render_evolution_tab():
    """
    Render the pattern evolution tab.
    """
    st.subheader("⏳ Pattern Evolution")

    if "pattern_tracker" not in st.session_state:
        st.session_state["pattern_tracker"] = PatternEvolutionTracker()
        st.info("Pattern tracker initialized. Add scans to see evolution.")
        return

    tracker = st.session_state["pattern_tracker"]

    # Check if we have history
    evolution_data = tracker.get_evolution_data()
    if evolution_data.empty:
        st.info("No scan history available. Run analyses to start tracking patterns.")
        return

    # Evolution metrics
    st.subheader("📈 Evolution Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Scans", len(tracker.history))
    col2.metric("Total Patterns Detected", len(tracker.patterns))
    col3.metric("Emerging Patterns (7 days)", len(tracker.get_emerging_patterns()))

    # Time series plot
    st.subheader("📊 Similarity Over Time")
    st.line_chart(
        evolution_data.set_index("timestamp")[["avg_similarity", "max_similarity"]],
        use_container_width=True,
    )

    # Emerging patterns
    emerging = tracker.get_emerging_patterns(lookback_days=7)
    if emerging:
        st.subheader("🚨 Emerging Plagiarism Patterns")

        for pattern in emerging:
            with st.expander(
                f"{pattern['document_a']} ↔ {pattern['document_b']} "
                f"(Frequency: {pattern['frequency']})",
                expanded=True,
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Frequency", pattern["frequency"])
                col2.metric("Avg Similarity", f"{pattern['avg_similarity']:.2%}")
                col3.metric("First Seen", pattern["first_seen"].strftime("%Y-%m-%d"))

                st.info(
                    "⚠️ This pair appears repeatedly in scans. Consider investigating further."
                )

    # Pattern history
    with st.expander("📋 Pattern History"):
        pattern_data = []
        for pattern_key, pattern_info in tracker.patterns.items():
            for doc_a, doc_b, sim in pattern_info["pairs"]:
                pattern_data.append(
                    {
                        "Timestamp": pattern_info["timestamp"],
                        "Document A": doc_a,
                        "Document B": doc_b,
                        "Similarity": sim,
                    }
                )

        if pattern_data:
            st.dataframe(pd.DataFrame(pattern_data), use_container_width=True)


# ── Integration Function ──────────────────────────────────────────────────


def integrate_clustering_analysis(
    similarity_matrix: np.ndarray,
    document_names: List[str],
    embeddings: np.ndarray,
    document_chunks: Dict[str, List[str]],
):
    """
    Integrate clustering and topic analysis into the main application.
    """
    if similarity_matrix is None or document_names is None:
        return

    # Create tabs for clustering, topics, and evolution
    cluster_tab, topic_tab, evolution_tab = st.tabs(
        ["🔬 Clustering", "📚 Topics", "⏳ Evolution"]
    )

    with cluster_tab:
        render_clustering_tab(similarity_matrix, document_names, embeddings)

    with topic_tab:
        render_topic_analysis_tab(document_chunks)

    with evolution_tab:
        render_evolution_tab()


# ── End of Clustering and Topic Analysis Section ────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
