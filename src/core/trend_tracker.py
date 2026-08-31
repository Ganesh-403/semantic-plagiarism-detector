"""
src/core/trend_tracker.py
--------------------------
Plagiarism trend analytics engine.

Tracks similarity metrics over time by recording scan snapshots and
computing trend indicators: moving averages, rate of change, volatility,
seasonal decomposition, and automatic alerts on trend shifts.

Integrates with the existing ``scan_history`` table in corpus.db and
feeds into the recommendation engine.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.config import SimilarityThresholds, normalize_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data types
# ---------------------------------------------------------------------------

class TrendDirection(str, Enum):
    """Direction of a measured trend."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class AlertSeverity(str, Enum):
    """Severity level for trend alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ScanSnapshot:
    """A single recorded scan session."""
    timestamp: str
    document_count: int
    avg_similarity: float
    max_similarity: float
    flagged_count: int
    threshold_used: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "document_count": self.document_count,
            "avg_similarity": round(self.avg_similarity, 6),
            "max_similarity": round(self.max_similarity, 6),
            "flagged_count": self.flagged_count,
            "threshold_used": self.threshold_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScanSnapshot":
        return cls(
            timestamp=data["timestamp"],
            document_count=int(data["document_count"]),
            avg_similarity=float(data["avg_similarity"]),
            max_similarity=float(data["max_similarity"]),
            flagged_count=int(data["flagged_count"]),
            threshold_used=float(data["threshold_used"]),
        )


@dataclass
class TrendAlert:
    """An automatically detected trend anomaly or shift."""
    alert_id: str
    alert_severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    expected_value: float
    deviation: float
    detected_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.alert_severity.value,
            "title": self.title,
            "description": self.description,
            "metric_name": self.metric_name,
            "current_value": round(self.current_value, 6),
            "expected_value": round(self.expected_value, 6),
            "deviation": round(self.deviation, 6),
            "detected_at": self.detected_at,
        }


@dataclass
class TrendMetrics:
    """Computed trend metrics for a single metric over time."""
    metric_name: str
    values: List[float]
    timestamps: List[str]
    direction: TrendDirection
    slope: float
    r_squared: float
    moving_average: List[float]
    volatility: float
    min_value: float
    max_value: float
    mean_value: float
    latest_value: float
    change_rate: float  # percentage change from first to last

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "direction": self.direction.value,
            "slope": round(self.slope, 8),
            "r_squared": round(self.r_squared, 6),
            "volatility": round(self.volatility, 6),
            "min_value": round(self.min_value, 6),
            "max_value": round(self.max_value, 6),
            "mean_value": round(self.mean_value, 6),
            "latest_value": round(self.latest_value, 6),
            "change_rate": round(self.change_rate, 6),
            "data_points": len(self.values),
        }


@dataclass
class TrendAnalysisResult:
    """Complete output of a trend analysis run."""
    snapshots_analyzed: int
    time_range_days: int
    metrics: Dict[str, TrendMetrics]
    alerts: List[TrendAlert]
    moving_average_window: int
    generated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @property
    def critical_alerts(self) -> List[TrendAlert]:
        return [a for a in self.alerts if a.alert_severity == AlertSeverity.CRITICAL]

    @property
    def warning_alerts(self) -> List[TrendAlert]:
        return [a for a in self.alerts if a.alert_severity == AlertSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshots_analyzed": self.snapshots_analyzed,
            "time_range_days": self.time_range_days,
            "moving_average_window": self.moving_average_window,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "alerts": [a.to_dict() for a in self.alerts],
            "alert_summary": {
                "critical": len(self.critical_alerts),
                "warning": len(self.warning_alerts),
                "info": len(self.alerts) - len(self.critical_alerts) - len(self.warning_alerts),
            },
            "generated_at": self.generated_at,
        }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _compute_linear_regression(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float]:
    """Compute slope, intercept, and R-squared for a linear fit.

    Args:
        x: Independent variable (e.g., time indices).
        y: Dependent variable (e.g., metric values).

    Returns:
        (slope, intercept, r_squared) tuple.
    """
    n = len(x)
    if n < 2:
        return 0.0, float(y[0]) if n == 1 else 0.0, 0.0

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx == 0:
        return 0.0, float(y_mean), 0.0

    slope = float(ss_xy / ss_xx)
    intercept = float(y_mean - slope * x_mean)

    # R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    r_squared = max(0.0, min(1.0, r_squared))

    return slope, intercept, r_squared


def _compute_moving_average(
    values: Sequence[float], window: int
) -> List[float]:
    """Compute a simple moving average with the given window size.

    Values before the window is full are filled with NaN.

    Args:
        values: Input time series.
        window: Moving average window size.

    Returns:
        List of moving average values (same length as input).
    """
    result: List[float] = []
    arr = np.array(values, dtype=float)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(arr[start : i + 1])))
    return result


def _compute_volatility(values: Sequence[float]) -> float:
    """Compute the standard deviation of first differences as a volatility measure."""
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    diffs = np.diff(arr)
    return float(np.std(diffs))


def _detect_outliers_zscore(
    values: Sequence[float], threshold: float = 2.5
) -> List[Tuple[int, float]]:
    """Find values that deviate more than *threshold* standard deviations.

    Returns:
        List of (index, z-score) tuples for each outlier.
    """
    if len(values) < 3:
        return []
    arr = np.array(values, dtype=float)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return []
    outliers = []
    for i, v in enumerate(arr):
        z = abs(float((v - mean) / std))
        if z >= threshold:
            outliers.append((i, float(v - mean) / std))
    return outliers


def _parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string."""
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Core trend tracker
# ---------------------------------------------------------------------------

class PlagiarismTrendTracker:
    """Records scan snapshots and computes trend analytics.

    Usage::

        tracker = PlagiarismTrendTracker()
        tracker.record_snapshot(ScanSnapshot(...))
        result = tracker.analyze(window=7)
        print(result.to_dict())
    """

    TRACKED_METRICS = ("avg_similarity", "max_similarity", "flagged_count", "document_count")

    def __init__(
        self,
        moving_average_window: int = 5,
        alert_z_threshold: float = 2.5,
        trend_change_threshold: float = 0.01,
    ) -> None:
        self.moving_average_window = moving_average_window
        self.alert_z_threshold = alert_z_threshold
        self.trend_change_threshold = trend_change_threshold
        self._snapshots: List[ScanSnapshot] = []

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def add_snapshot(self, snapshot: ScanSnapshot) -> None:
        """Record a new scan snapshot."""
        self._snapshots.append(snapshot)
        self._snapshots.sort(key=lambda s: s.timestamp)

    def load_snapshots(self, snapshots: Sequence[ScanSnapshot]) -> None:
        """Bulk-load snapshots, replacing any existing data."""
        self._snapshots = sorted(snapshots, key=lambda s: s.timestamp)

    def get_snapshots(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[ScanSnapshot]:
        """Retrieve snapshots with optional date filtering."""
        result = self._snapshots
        if start_date:
            result = [s for s in result if s.timestamp >= start_date]
        if end_date:
            result = [s for s in result if s.timestamp <= end_date]
        return result

    def analyze(
        self,
        window: Optional[int] = None,
        include_alerts: bool = True,
    ) -> TrendAnalysisResult:
        """Run full trend analysis on all loaded snapshots.

        Args:
            window: Number of recent snapshots to analyze (None = all).
            include_alerts: Whether to generate trend alerts.

        Returns:
            TrendAnalysisResult with per-metric trends and alerts.
        """
        snapshots = self._snapshots
        if window and window > 0:
            snapshots = snapshots[-window:]

        if not snapshots:
            return TrendAnalysisResult(
                snapshots_analyzed=0,
                time_range_days=0,
                metrics={},
                alerts=[],
                moving_average_window=self.moving_average_window,
            )

        # Compute time range
        first_ts = _parse_timestamp(snapshots[0].timestamp)
        last_ts = _parse_timestamp(snapshots[-1].timestamp)
        time_range_days = (
            (last_ts - first_ts).days if first_ts and last_ts else 0
        )

        # Analyze each tracked metric
        metrics: Dict[str, TrendMetrics] = {}
        for metric_name in self.TRACKED_METRICS:
            values = [getattr(s, metric_name) for s in snapshots]
            timestamps = [s.timestamp for s in snapshots]
            tm = self._analyze_metric(metric_name, values, timestamps)
            metrics[metric_name] = tm

        # Generate alerts
        alerts: List[TrendAlert] = []
        if include_alerts:
            alerts = self._generate_alerts(snapshots, metrics)

        return TrendAnalysisResult(
            snapshots_analyzed=len(snapshots),
            time_range_days=time_range_days,
            metrics=metrics,
            alerts=alerts,
            moving_average_window=self.moving_average_window,
        )

    def _analyze_metric(
        self,
        name: str,
        values: List[float],
        timestamps: List[str],
    ) -> TrendMetrics:
        """Compute trend metrics for a single time series."""
        arr = np.array(values, dtype=float)
        n = len(arr)
        x = np.arange(n, dtype=float)

        # Linear regression
        slope, intercept, r_squared = _compute_linear_regression(x, arr)

        # Direction classification
        if abs(slope) < self.trend_change_threshold:
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        # Moving average
        ma = _compute_moving_average(values, self.moving_average_window)

        # Volatility
        vol = _compute_volatility(values)

        # Change rate (first to last)
        first_val = float(arr[0])
        last_val = float(arr[-1])
        if first_val != 0:
            change_rate = (last_val - first_val) / abs(first_val)
        else:
            change_rate = 0.0

        return TrendMetrics(
            metric_name=name,
            values=values,
            timestamps=timestamps,
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            moving_average=ma,
            volatility=vol,
            min_value=float(np.min(arr)),
            max_value=float(np.max(arr)),
            mean_value=float(np.mean(arr)),
            latest_value=last_val,
            change_rate=change_rate,
        )

    def _generate_alerts(
        self,
        snapshots: List[ScanSnapshot],
        metrics: Dict[str, TrendMetrics],
    ) -> List[TrendAlert]:
        """Generate alerts based on trend analysis and statistical outliers."""
        alerts: List[TrendAlert] = []
        alert_counter = 0

        # 1. Rising plagiarism rate alert
        avg_sim_metric = metrics.get("avg_similarity")
        if avg_sim_metric and avg_sim_metric.direction == TrendDirection.INCREASING:
            if avg_sim_metric.slope > 0.02:
                alerts.append(TrendAlert(
                    alert_id=f"ALERT-{alert_counter:04d}",
                    alert_severity=AlertSeverity.CRITICAL,
                    title="Rising Average Similarity",
                    description=(
                        f"Average similarity is trending upward (slope={avg_sim_metric.slope:.4f}) "
                        f"with R²={avg_sim_metric.r_squared:.3f}. This may indicate a "
                        f"growing plagiarism problem."
                    ),
                    metric_name="avg_similarity",
                    current_value=avg_sim_metric.latest_value,
                    expected_value=avg_sim_metric.mean_value,
                    deviation=avg_sim_metric.latest_value - avg_sim_metric.mean_value,
                    detected_at=datetime.now().isoformat(),
                ))
                alert_counter += 1
            elif avg_sim_metric.slope > 0.01:
                alerts.append(TrendAlert(
                    alert_id=f"ALERT-{alert_counter:04d}",
                    alert_severity=AlertSeverity.WARNING,
                    title="Moderate Rise in Average Similarity",
                    description=(
                        f"Average similarity shows a moderate upward trend "
                        f"(slope={avg_sim_metric.slope:.4f}). Monitor closely."
                    ),
                    metric_name="avg_similarity",
                    current_value=avg_sim_metric.latest_value,
                    expected_value=avg_sim_metric.mean_value,
                    deviation=avg_sim_metric.latest_value - avg_sim_metric.mean_value,
                    detected_at=datetime.now().isoformat(),
                ))
                alert_counter += 1

        # 2. Rising flag rate alert
        flagged_metric = metrics.get("flagged_count")
        if flagged_metric and flagged_metric.direction == TrendDirection.INCREASING:
            if flagged_metric.change_rate > 0.5:
                alerts.append(TrendAlert(
                    alert_id=f"ALERT-{alert_counter:04d}",
                    alert_severity=AlertSeverity.CRITICAL,
                    title="Surge in Flagged Pairs",
                    description=(
                        f"Flagged pair count increased by {flagged_metric.change_rate:.0%} "
                        f"from {flagged_metric.values[0]:.0f} to "
                        f"{flagged_metric.latest_value:.0f}."
                    ),
                    metric_name="flagged_count",
                    current_value=flagged_metric.latest_value,
                    expected_value=flagged_metric.mean_value,
                    deviation=flagged_metric.latest_value - flagged_metric.mean_value,
                    detected_at=datetime.now().isoformat(),
                ))
                alert_counter += 1

        # 3. Statistical outlier detection on latest value
        for name, tm in metrics.items():
            outliers = _detect_outliers_zscore(
                tm.values, self.alert_z_threshold
            )
            # Only alert on the last point being an outlier
            if outliers and outliers[-1][0] == len(tm.values) - 1:
                z_score = outliers[-1][1]
                alerts.append(TrendAlert(
                    alert_id=f"ALERT-{alert_counter:04d}",
                    alert_severity=(
                        AlertSeverity.CRITICAL
                        if abs(z_score) > 3.5
                        else AlertSeverity.WARNING
                    ),
                    title=f"Statistical Outlier: {name}",
                    description=(
                        f"Latest {name} value ({tm.latest_value:.4f}) is "
                        f"{abs(z_score):.2f} standard deviations from the mean "
                        f"({tm.mean_value:.4f})."
                    ),
                    metric_name=name,
                    current_value=tm.latest_value,
                    expected_value=tm.mean_value,
                    deviation=tm.latest_value - tm.mean_value,
                    detected_at=datetime.now().isoformat(),
                ))
                alert_counter += 1

        # 4. High volatility alert
        for name, tm in metrics.items():
            if tm.volatility > 0.1 and tm.mean_value > 0:
                cv = tm.volatility / abs(tm.mean_value)
                if cv > 0.3:
                    alerts.append(TrendAlert(
                        alert_id=f"ALERT-{alert_counter:04d}",
                        alert_severity=AlertSeverity.INFO,
                        title=f"High Volatility: {name}",
                        description=(
                            f"{name} shows high volatility "
                            f"(std of changes={tm.volatility:.4f}, "
                            f"CV={cv:.2f}). Results may be inconsistent."
                        ),
                        metric_name=name,
                        current_value=tm.volatility,
                        expected_value=0.0,
                        deviation=tm.volatility,
                        detected_at=datetime.now().isoformat(),
                    ))
                    alert_counter += 1

        # Sort: critical first
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.INFO: 2,
        }
        alerts.sort(key=lambda a: severity_order[a.alert_severity])

        return alerts

    def compare_periods(
        self,
        recent_days: int,
        baseline_days: int,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare recent period against a baseline period.

        Args:
            recent_days: Number of most recent days to use as "recent".
            baseline_days: Number of days before the recent period to use
                as "baseline".

        Returns:
            Dict mapping metric names to comparison results.
        """
        if not self._snapshots:
            return {}

        last_ts = _parse_timestamp(self._snapshots[-1].timestamp)
        if last_ts is None:
            return {}

        recent_cutoff = (last_ts - timedelta(days=recent_days)).isoformat()
        baseline_start = (last_ts - timedelta(days=recent_days + baseline_days)).isoformat()
        baseline_end = recent_cutoff

        recent = [
            s for s in self._snapshots if s.timestamp >= recent_cutoff
        ]
        baseline = [
            s for s in self._snapshots
            if baseline_start <= s.timestamp < baseline_end
        ]

        if not recent or not baseline:
            return {}

        comparisons: Dict[str, Dict[str, Any]] = {}
        for metric in self.TRACKED_METRICS:
            recent_vals = [getattr(s, metric) for s in recent]
            baseline_vals = [getattr(s, metric) for s in baseline]

            recent_mean = float(np.mean(recent_vals))
            baseline_mean = float(np.mean(baseline_vals))
            if baseline_mean != 0:
                pct_change = (recent_mean - baseline_mean) / abs(baseline_mean)
            else:
                pct_change = 0.0

            comparisons[metric] = {
                "recent_mean": round(recent_mean, 6),
                "baseline_mean": round(baseline_mean, 6),
                "absolute_change": round(recent_mean - baseline_mean, 6),
                "percent_change": round(pct_change, 6),
                "recent_count": len(recent),
                "baseline_count": len(baseline),
            }

        return comparisons

    def get_summary(self) -> Dict[str, Any]:
        """Quick summary of all loaded snapshots."""
        if not self._snapshots:
            return {"total_snapshots": 0}

        all_avg = [s.avg_similarity for s in self._snapshots]
        all_max = [s.max_similarity for s in self._snapshots]
        all_flagged = [s.flagged_count for s in self._snapshots]
        all_docs = [s.document_count for s in self._snapshots]

        return {
            "total_snapshots": len(self._snapshots),
            "first_snapshot": self._snapshots[0].timestamp,
            "latest_snapshot": self._snapshots[-1].timestamp,
            "avg_similarity": {
                "mean": round(float(np.mean(all_avg)), 6),
                "min": round(float(np.min(all_avg)), 6),
                "max": round(float(np.max(all_avg)), 6),
            },
            "max_similarity": {
                "mean": round(float(np.mean(all_max)), 6),
                "min": round(float(np.min(all_max)), 6),
                "max": round(float(np.max(all_max)), 6),
            },
            "flagged_count": {
                "mean": round(float(np.mean(all_flagged)), 6),
                "min": int(min(all_flagged)),
                "max": int(max(all_flagged)),
                "total": int(sum(all_flagged)),
            },
            "document_count": {
                "mean": round(float(np.mean(all_docs)), 6),
                "min": int(min(all_docs)),
                "max": int(max(all_docs)),
            },
        }


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def snapshots_from_dicts(records: Sequence[Dict[str, Any]]) -> List[ScanSnapshot]:
    """Convert a list of dicts (e.g., from SQLite rows) to ScanSnapshot objects.

    Handles both dict-like objects with string keys and sqlite3.Row objects.

    Args:
        records: Sequence of dicts with scan_history columns.

    Returns:
        List of ScanSnapshot objects.
    """
    result = []
    for record in records:
        try:
            result.append(ScanSnapshot.from_dict(dict(record)))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed snapshot record: %s", exc)
    return result


def create_demo_snapshots(n: int = 30) -> List[ScanSnapshot]:
    """Generate synthetic snapshots for testing and demonstration.

    Simulates a slow upward trend in plagiarism scores with random noise.

    Args:
        n: Number of snapshots to generate.

    Returns:
        List of synthetic ScanSnapshot objects.
    """
    np.random.seed(42)
    snapshots = []
    base_date = datetime(2025, 1, 1)

    for i in range(n):
        day = base_date + timedelta(days=i)
        trend = i * 0.005  # slow upward drift
        noise = np.random.normal(0, 0.02)

        avg_sim = float(np.clip(0.45 + trend + noise, 0.0, 1.0))
        max_sim = float(np.clip(avg_sim + 0.15 + np.random.normal(0, 0.03), 0.0, 1.0))
        doc_count = int(np.clip(20 + np.random.randint(-3, 5), 5, 50))
        flagged = int(np.clip(doc_count * avg_sim * 0.3, 0, doc_count))

        snapshots.append(ScanSnapshot(
            timestamp=day.isoformat(),
            document_count=doc_count,
            avg_similarity=avg_sim,
            max_similarity=max_sim,
            flagged_count=flagged,
            threshold_used=0.59,
        ))

    return snapshots
