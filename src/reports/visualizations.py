"""
Visualization utilities for report generation
Creates heatmaps, charts, and diagrams for plagiarism reports.
"""

import base64
import io
from typing import List, Dict, Any, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


class ReportVisualizer:
    """
    Creates visualizations for reports including heatmaps, charts, and diagrams.
    """
    
    def __init__(self):
        self._setup_style()
        self.color_palettes = {
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'inferno': plt.cm.inferno,
            'magma': plt.cm.magma,
            'coolwarm': plt.cm.coolwarm,
            'RdYlGn': plt.cm.RdYlGn,
            'Blues': plt.cm.Blues,
            'Greens': plt.cm.Greens,
            'Reds': plt.cm.Reds
        }
    
    def _setup_style(self):
        """Setup matplotlib style for consistent visuals."""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("viridis")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        plt.rcParams['savefig.bbox'] = 'tight'
    
    def create_heatmap(
        self,
        matrix: List[List[float]],
        labels: List[str],
        title: str = "Similarity Matrix",
        cmap: str = "RdYlGn_r",
        figsize: Tuple[int, int] = (12, 10),
        annotate: bool = True,
        annotate_fontsize: int = 8
    ) -> str:
        """
        Create a heatmap visualization of similarity scores.
        
        Args:
            matrix: 2D list of similarity scores
            labels: List of document names
            title: Chart title
            cmap: Color map name
            figsize: Figure size
            annotate: Whether to show values
            annotate_fontsize: Font size for annotations
        
        Returns:
            Base64 encoded image string
        """
        if not matrix or not labels:
            return ""
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(
            matrix,
            cmap=self.color_palettes.get(cmap, plt.cm.RdYlGn_r),
            vmin=0,
            vmax=1,
            interpolation='nearest'
        )
        
        # Set labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Similarity Score', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Add annotations
        if annotate:
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    text_color = "white" if matrix[i][j] > 0.5 else "black"
                    ax.text(
                        j, i, f"{matrix[i][j]:.2f}",
                        ha="center", va="center",
                        color=text_color,
                        fontsize=annotate_fontsize,
                        fontweight='bold'
                    )
        
        # Add title and format
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.tick_params(axis='both', which='both', length=0)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        return self._fig_to_base64(fig)
    
    def create_similarity_chart(
        self,
        scores: List[float],
        labels: List[str],
        title: str = "Similarity Scores",
        threshold: float = 0.5,
        figsize: Tuple[int, int] = (10, 6)
    ) -> str:
        """
        Create a bar chart of similarity scores.
        
        Args:
            scores: List of similarity scores
            labels: List of labels for each bar
            title: Chart title
            threshold: Threshold line value
            figsize: Figure size
        
        Returns:
            Base64 encoded image string
        """
        if not scores or not labels:
            return ""
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color bars based on threshold
        colors = [
            '#2ecc71' if s >= threshold else 
            '#f39c12' if s >= 0.3 else 
            '#e74c3c' 
            for s in scores
        ]
        
        bars = ax.bar(labels, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add threshold line
        ax.axhline(
            y=threshold,
            color='#3498db',
            linestyle='--',
            linewidth=2,
            label=f'Threshold ({threshold:.0%})'
        )
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.02,
                f'{score:.1%}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_xlabel('Document Pairs', fontsize=13)
        ax.set_ylabel('Similarity Score', fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        return self._fig_to_base64(fig)
    
    def create_distribution_chart(
        self,
        scores: List[float],
        title: str = "Score Distribution",
        figsize: Tuple[int, int] = (10, 6),
        bins: int = 20
    ) -> str:
        """
        Create a histogram of score distribution.
        
        Args:
            scores: List of similarity scores
            title: Chart title
            figsize: Figure size
            bins: Number of histogram bins
        
        Returns:
            Base64 encoded image string
        """
        if not scores:
            return ""
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create histogram
        n, bins_edges, patches = ax.hist(
            scores,
            bins=bins,
            color='#3498db',
            edgecolor='black',
            alpha=0.7
        )
        
        # Add mean and median lines
        mean_val = np.mean(scores)
        median_val = np.median(scores)
        
        ax.axvline(
            x=mean_val,
            color='#e74c3c',
            linestyle='--',
            linewidth=2,
            label=f'Mean: {mean_val:.1%}'
        )
        ax.axvline(
            x=median_val,
            color='#2ecc71',
            linestyle='--',
            linewidth=2,
            label=f'Median: {median_val:.1%}'
        )
        
        ax.set_xlabel('Similarity Score', fontsize=13)
        ax.set_ylabel('Frequency', fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        return self._fig_to_base64(fig)
    
    def create_severity_distribution(
        self,
        matches: List[Dict[str, Any]],
        title: str = "Severity Distribution",
        figsize: Tuple[int, int] = (8, 6)
    ) -> str:
        """
        Create a severity distribution chart.
        
        Args:
            matches: List of match dictionaries
            title: Chart title
            figsize: Figure size
        
        Returns:
            Base64 encoded image string
        """
        if not matches:
            return ""
        
        severities = {'High': 0, 'Medium': 0, 'Low': 0, 'None': 0}
        colors = {
            'High': '#e74c3c',
            'Medium': '#f39c12',
            'Low': '#f1c40f',
            'None': '#95a5a6'
        }
        
        for match in matches:
            score = match.get('hybrid_score', match.get('score', 0))
            if score >= 0.8:
                severities['High'] += 1
            elif score >= 0.6:
                severities['Medium'] += 1
            elif score >= 0.4:
                severities['Low'] += 1
            else:
                severities['None'] += 1
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(
            list(severities.keys()),
            list(severities.values()),
            color=[colors[k] for k in severities.keys()],
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )
        
        for bar, value in zip(bars, severities.values()):
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                bar.get_height() + 0.5,
                str(value),
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
        
        ax.set_ylabel('Count', fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        return self._fig_to_base64(fig)
    
    def create_summary_dashboard(
        self,
        stats: Dict[str, Any],
        matches: List[Dict[str, Any]],
        title: str = "Summary Dashboard",
        figsize: Tuple[int, int] = (10, 8)
    ) -> str:
        """
        Create a summary dashboard with key metrics.
        
        Args:
            stats: Statistics dictionary
            matches: List of match dictionaries
            title: Dashboard title
            figsize: Figure size
        
        Returns:
            Base64 encoded image string
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        # Build summary text
        summary_text = [
            "=" * 60,
            f"  {title}",
            "=" * 60,
            "",
            f"  📄 Total Documents: {stats.get('total_documents', 0)}",
            f"  🔄 Total Comparisons: {stats.get('total_comparisons', 0)}",
            f"  🔍 Total Matches Found: {len(matches)}",
            "",
            "-" * 60,
            "  📊 Similarity Statistics:",
            f"    • Average: {stats.get('avg_similarity', 0):.2%}",
            f"    • Median:  {stats.get('median_similarity', 0):.2%}",
            f"    • Maximum: {stats.get('max_similarity', 0):.2%}",
            f"    • Minimum: {stats.get('min_similarity', 0):.2%}",
            f"    • Std Dev: {stats.get('std_similarity', 0):.2%}",
            "",
            "-" * 60,
            "  ⚠️ Severity Distribution:",
            f"    • 🔴 High (≥80%):   {stats.get('high_severity_count', 0)}",
            f"    • 🟡 Medium (50-80%): {stats.get('medium_severity_count', 0)}",
            f"    • 🟢 Low (30-50%):   {stats.get('low_severity_count', 0)}",
            f"    • ⚪ Very Low (<30%): {stats.get('none_severity_count', 0)}",
            "",
            "=" * 60,
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60
        ]
        
        ax.text(
            0.1, 0.9,
            "\n".join(summary_text),
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            family='monospace',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.95,
                edgecolor='#3498db',
                linewidth=2
            )
        )
        
        ax.set_title("Summary Dashboard", fontsize=16, fontweight='bold', pad=20)
        
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        return image_base64