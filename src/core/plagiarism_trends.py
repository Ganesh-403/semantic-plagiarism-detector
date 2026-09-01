"""
Plagiarism Trend Analytics Engine
=================================
Tracks plagiarism detection history over time, computes statistical
trends, and generates institutional analytics for administrative
reporting and policy decisions.

Provides:
  - Incident aggregation by time window (daily, weekly, monthly)
  - Trend detection with linear regression and moving averages
  - Statistical summaries (mean, median, std, percentiles)
  - Severity distribution analysis over time
  - Top offender tracking and repeat-offense detection
  - Exportable report generation (JSON, CSV)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── Enums & Constants ─────────────────────────────────────────────────────────

class TimeWindow(Enum):
    """Aggregation time windows for trend analysis."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class TrendDirection(Enum):
    """Detected trend directions."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


class SeverityLevel(Enum):
    """Plagiarism severity classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReportFormat(Enum):
    """Export report format options."""
    JSON = "json"
    CSV = "csv"


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlagiarismIncident:
    """A single plagiarism detection event."""
    incident_id: str
    document_name: str
    matched_against: str
    similarity_score: float
    severity: str
    detected_at: datetime
    chunk_count: int = 0
    max_chunk_similarity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesPoint:
    """A single data point in a time series."""
    timestamp: datetime
    value: float
    label: str = ""


@dataclass
class TrendResult:
    """Result of a trend analysis computation."""
    direction: TrendDirection
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    confidence: float
    forecast_values: List[float] = field(default_factory=list)
    forecast_timestamps: List[datetime] = field(default_factory=list)


@dataclass
class StatisticalSummary:
    """Statistical summary of a numeric dataset."""
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_25: float
    percentile_75: float
    percentile_90: float
    iqr: float


@dataclass
class SeverityDistribution:
    """Distribution of severity levels across a time period."""
    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0
    total: int = 0

    @property
    def high_rate(self) -> float:
        """Percentage of high+critical incidents."""
        if self.total == 0:
            return 0.0
        return ((self.high + self.critical) / self.total) * 100.0


@dataclass
class TrendWindow:
    """Aggregated data for a single time window."""
    start: datetime
    end: datetime
    window_label: str
    incident_count: int = 0
    avg_similarity: float = 0.0
    max_similarity: float = 0.0
    severity_dist: SeverityDistribution = field(default_factory=SeverityDistribution)
    unique_documents: int = 0
    unique_matchees: int = 0


@dataclass
class OffenderProfile:
    """Profile of a repeated plagiarism offender."""
    document_name: str
    incident_count: int
    avg_similarity: float
    max_similarity: float
    first_detected: datetime
    last_detected: datetime
    unique_matchees: List[str] = field(default_factory=list)
    severity_history: List[str] = field(default_factory=list)


@dataclass
class AnalyticsReport:
    """Complete analytics report output."""
    generated_at: datetime
    time_window: str
    total_incidents: int
    date_range_start: datetime
    date_range_end: datetime
    statistical_summary: StatisticalSummary
    severity_distribution: SeverityDistribution
    trend: TrendResult
    windows: List[TrendWindow]
    top_offenders: List[OffenderProfile]
    monthly_growth_rate: float
    repeat_offense_rate: float


# ── Linear Regression Helper ──────────────────────────────────────────────────

def _linear_regression(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Compute simple linear regression without numpy.

    Returns:
        (slope, intercept, r_squared, p_value)

    p_value is estimated via a two-tailed t-test approximation.
    """
    n = len(x_values)
    if n < 2:
        return 0.0, 0.0, 0.0, 1.0

    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n

    ss_xx = sum((x - x_mean) ** 2 for x in x_values)
    ss_yy = sum((y - y_mean) ** 2 for y in y_values)
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))

    if ss_xx == 0:
        return 0.0, y_mean, 0.0, 1.0

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # R-squared
    if ss_yy == 0:
        r_squared = 1.0 if ss_xy == 0 else 0.0
    else:
        r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

    # P-value approximation using t-distribution
    if n <= 2:
        p_value = 1.0
    else:
        residual_ss = ss_yy - slope * ss_xy
        if residual_ss <= 0:
            p_value = 0.0 if r_squared > 0.9 else 1.0
        else:
            se_slope = math.sqrt(residual_ss / (n - 2) / ss_xx) if ss_xx > 0 else float('inf')
            if se_slope == 0:
                p_value = 0.0 if slope != 0 else 1.0
            else:
                t_stat = abs(slope) / se_slope
                # Rough p-value from t-stat with n-2 degrees of freedom
                # Using normal approximation for large n
                df = n - 2
                if df >= 30:
                    # Normal approximation
                    z = t_stat
                    p_value = 2.0 * (1.0 - _norm_cdf(z))
                else:
                    p_value = _approx_t_pvalue(t_stat, df)

    return slope, intercept, r_squared, p_value


def _norm_cdf(x: float) -> float:
    """Approximate the standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _approx_t_pvalue(t: float, df: int) -> float:
    """Approximate two-tailed p-value from t-distribution."""
    # Use Cornish-Fisher expansion for small df
    x = df / (df + t ** 2)
    if x >= 1.0:
        return 1.0
    if x <= 0.0:
        return 0.0

    # Incomplete beta function approximation
    a = df / 2.0
    b = 0.5
    return float(min(1.0, max(0.0, _incomplete_beta(x, a, b))))


def _incomplete_beta(x: float, a: float, b: float, max_iter: int = 100) -> float:
    """Regularized incomplete beta function via continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use series expansion
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(
        math.log(x) * a + math.log(1.0 - x) * b - lbeta
    ) / a

    # Lentz's algorithm for continued fraction
    f_val = 1.0
    c_val = 1.0
    d_val = 0.0

    for i in range(max_iter):
        m = i // 2 + 1
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 2) * (a + 2 * m - 1))
        else:
            num = -(a + m - 1) * (a + b + m - 1) * x / ((a + 2 * m - 1) * (a + 2 * m))

        d_val = 1.0 + num * d_val
        if abs(d_val) < 1e-30:
            d_val = 1e-30
        d_val = 1.0 / d_val

        c_val = 1.0 + num / c_val
        if abs(c_val) < 1e-30:
            c_val = 1e-30

        f_val *= c_val * d_val
        if abs(c_val * d_val - 1.0) < 1e-8:
            break

    return front * (f_val - 1.0)


# ── Core Analytics Engine ─────────────────────────────────────────────────────

class PlagiarismTrendAnalytics:
    """Main engine for plagiarism trend analysis and reporting.

    Accumulates incidents and provides analytical queries over them.
    Can be used standalone or integrated with the Streamlit dashboard.
    """

    def __init__(self, default_window: TimeWindow = TimeWindow.MONTHLY):
        self._incidents: List[PlagiarismIncident] = []
        self._default_window = default_window
        self._window_cache: Dict[str, List[TrendWindow]] = {}

    @property
    def incident_count(self) -> int:
        """Return total number of tracked incidents."""
        return len(self._incidents)

    def add_incident(self, incident: PlagiarismIncident) -> None:
        """Record a new plagiarism incident."""
        self._incidents.append(incident)
        self._window_cache.clear()
        logger.debug("Recorded incident %s for document '%s'", incident.incident_id, incident.document_name)

    def add_incidents(self, incidents: Sequence[PlagiarismIncident]) -> None:
        """Record multiple plagiarism incidents in batch."""
        for inc in incidents:
            self._incidents.append(inc)
        self._window_cache.clear()
        logger.info("Batch-recorded %d incidents", len(incidents))

    def load_from_dicts(self, records: List[Dict[str, Any]]) -> int:
        """Load incidents from a list of dictionaries.

        Expected keys: incident_id, document_name, matched_against,
        similarity_score, severity, detected_at (ISO string).
        Optional: chunk_count, max_chunk_similarity, metadata.

        Returns:
            Number of incidents loaded.
        """
        loaded = 0
        for rec in records:
            try:
                detected = rec.get("detected_at", "")
                if isinstance(detected, str):
                    detected = datetime.fromisoformat(detected.replace("Z", "+00:00"))

                incident = PlagiarismIncident(
                    incident_id=str(rec.get("incident_id", f"auto_{loaded}")),
                    document_name=str(rec.get("document_name", "unknown")),
                    matched_against=str(rec.get("matched_against", "unknown")),
                    similarity_score=float(rec.get("similarity_score", 0.0)),
                    severity=str(rec.get("severity", "low")),
                    detected_at=detected,
                    chunk_count=int(rec.get("chunk_count", 0)),
                    max_chunk_similarity=float(rec.get("max_chunk_similarity", 0.0)),
                    metadata=rec.get("metadata", {}),
                )
                self._incidents.append(incident)
                loaded += 1
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning("Skipping malformed incident record: %s", exc)
        self._window_cache.clear()
        return loaded

    def get_date_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Return (earliest, latest) detection timestamps."""
        if not self._incidents:
            return None, None
        dates = [inc.detected_at for inc in self._incidents]
        return min(dates), max(dates)

    # ── Aggregation ────────────────────────────────────────────────────────

    def aggregate_by_window(
        self,
        window: Optional[TimeWindow] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[TrendWindow]:
        """Aggregate incidents into time windows.

        Args:
            window: Time window size. Defaults to self._default_window.
            start_date: Optional filter for incident start.
            end_date: Optional filter for incident end.

        Returns:
            List of TrendWindow objects sorted by start time.
        """
        window = window or self._default_window
        cache_key = f"{window.value}_{start_date}_{end_date}"

        if cache_key in self._window_cache:
            return self._window_cache[cache_key]

        # Filter incidents by date range
        filtered = self._incidents
        if start_date:
            filtered = [i for i in filtered if i.detected_at >= start_date]
        if end_date:
            filtered = [i for i in filtered if i.detected_at <= end_date]

        if not filtered:
            result = []
            self._window_cache[cache_key] = result
            return result

        # Build window buckets
        buckets: Dict[datetime, List[PlagiarismIncident]] = defaultdict(list)
        for inc in filtered:
            bucket_start = self._get_window_start(inc.detected_at, window)
            buckets[bucket_start].append(inc)

        # Sort buckets and build TrendWindow objects
        sorted_starts = sorted(buckets.keys())
        windows: List[TrendWindow] = []

        for i, start in enumerate(sorted_starts):
            end = self._get_window_end(start, window)
            bucket = buckets[start]

            scores = [inc.similarity_score for inc in bucket]
            severities = [inc.severity.lower() for inc in bucket]
            sev_counts = Counter(severities)

            sev_dist = SeverityDistribution(
                low=sev_counts.get("low", 0),
                medium=sev_counts.get("medium", 0),
                high=sev_counts.get("high", 0),
                critical=sev_counts.get("critical", 0),
                total=len(bucket),
            )

            tw = TrendWindow(
                start=start,
                end=end,
                window_label=self._format_window_label(start, window),
                incident_count=len(bucket),
                avg_similarity=statistics.mean(scores) if scores else 0.0,
                max_similarity=max(scores) if scores else 0.0,
                severity_dist=sev_dist,
                unique_documents=len(set(inc.document_name for inc in bucket)),
                unique_matchees=len(set(inc.matched_against for inc in bucket)),
            )
            windows.append(tw)

        self._window_cache[cache_key] = windows
        return windows

    def _get_window_start(self, dt: datetime, window: TimeWindow) -> datetime:
        """Compute the start of the time window containing dt."""
        if window == TimeWindow.DAILY:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        elif window == TimeWindow.WEEKLY:
            days_since_monday = dt.weekday()
            return (dt - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        elif window == TimeWindow.MONTHLY:
            return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif window == TimeWindow.QUARTERLY:
            quarter_month = ((dt.month - 1) // 3) * 3 + 1
            return dt.replace(month=quarter_month, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif window == TimeWindow.YEARLY:
            return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return dt

    def _get_window_end(self, start: datetime, window: TimeWindow) -> datetime:
        """Compute the end of a time window given its start."""
        if window == TimeWindow.DAILY:
            return start + timedelta(days=1)
        elif window == TimeWindow.WEEKLY:
            return start + timedelta(weeks=1)
        elif window == TimeWindow.MONTHLY:
            if start.month == 12:
                return start.replace(year=start.year + 1, month=1)
            return start.replace(month=start.month + 1)
        elif window == TimeWindow.QUARTERLY:
            quarter_month = ((start.month - 1) // 3) * 3 + 1
            next_q_month = quarter_month + 3
            if next_q_month > 12:
                return start.replace(year=start.year + 1, month=next_q_month - 12)
            return start.replace(month=next_q_month)
        elif window == TimeWindow.YEARLY:
            return start.replace(year=start.year + 1)
        return start

    def _format_window_label(self, start: datetime, window: TimeWindow) -> str:
        """Format a human-readable label for a window start."""
        if window == TimeWindow.DAILY:
            return start.strftime("%Y-%m-%d")
        elif window == TimeWindow.WEEKLY:
            return f"W{start.isocalendar()[1]:02d}-{start.year}"
        elif window == TimeWindow.MONTHLY:
            return start.strftime("%b %Y")
        elif window == TimeWindow.QUARTERLY:
            q = (start.month - 1) // 3 + 1
            return f"Q{q} {start.year}"
        elif window == TimeWindow.YEARLY:
            return str(start.year)
        return start.isoformat()

    # ── Trend Analysis ─────────────────────────────────────────────────────

    def compute_trend(
        self,
        window: Optional[TimeWindow] = None,
        forecast_periods: int = 3,
    ) -> TrendResult:
        """Compute trend analysis with linear regression and forecasting.

        Args:
            window: Time window for aggregation.
            forecast_periods: Number of future periods to forecast.

        Returns:
            TrendResult with direction, regression stats, and forecasts.
        """
        windows = self.aggregate_by_window(window)

        if len(windows) < 3:
            return TrendResult(
                direction=TrendDirection.INSUFFICIENT_DATA,
                slope=0.0,
                intercept=0.0,
                r_squared=0.0,
                p_value=1.0,
                confidence=0.0,
            )

        # Use incident counts as the time series
        x_values = list(range(len(windows)))
        y_values = [w.incident_count for w in windows]

        slope, intercept, r_squared, p_value = _linear_regression(x_values, y_values)

        # Determine trend direction
        confidence = (1.0 - p_value) * 100.0
        if p_value < 0.05 and abs(slope) > 0.01:
            direction = TrendDirection.INCREASING if slope > 0 else TrendDirection.DECREASING
        else:
            direction = TrendDirection.STABLE

        # Generate forecasts
        forecast_values = []
        forecast_timestamps = []
        last_window = windows[-1]
        window_enum = window or self._default_window

        for i in range(1, forecast_periods + 1):
            future_x = len(windows) + i - 1
            predicted = max(0.0, slope * future_x + intercept)
            forecast_values.append(round(predicted, 2))
            forecast_start = self._get_window_end(last_window.start, window_enum)
            # Shift forecast start by i windows
            for _ in range(i):
                forecast_start = self._get_window_end(forecast_start, window_enum)
            forecast_timestamps.append(forecast_start)

        return TrendResult(
            direction=direction,
            slope=round(slope, 4),
            intercept=round(intercept, 4),
            r_squared=round(r_squared, 4),
            p_value=round(p_value, 6),
            confidence=round(confidence, 2),
            forecast_values=forecast_values,
            forecast_timestamps=forecast_timestamps,
        )

    def compute_similarity_trend(
        self,
        window: Optional[TimeWindow] = None,
    ) -> TrendResult:
        """Compute trend for average similarity scores over time."""
        windows = self.aggregate_by_window(window)

        if len(windows) < 3:
            return TrendResult(
                direction=TrendDirection.INSUFFICIENT_DATA,
                slope=0.0,
                intercept=0.0,
                r_squared=0.0,
                p_value=1.0,
                confidence=0.0,
            )

        x_values = list(range(len(windows)))
        y_values = [w.avg_similarity for w in windows]

        slope, intercept, r_squared, p_value = _linear_regression(x_values, y_values)
        confidence = (1.0 - p_value) * 100.0

        if p_value < 0.05 and abs(slope) > 0.001:
            direction = TrendDirection.INCREASING if slope > 0 else TrendDirection.DECREASING
        else:
            direction = TrendDirection.STABLE

        return TrendResult(
            direction=direction,
            slope=round(slope, 4),
            intercept=round(intercept, 4),
            r_squared=round(r_squared, 4),
            p_value=round(p_value, 6),
            confidence=round(confidence, 2),
        )

    # ── Statistical Summaries ──────────────────────────────────────────────

    def compute_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> StatisticalSummary:
        """Compute statistical summary of similarity scores."""
        filtered = self._incidents
        if start_date:
            filtered = [i for i in filtered if i.detected_at >= start_date]
        if end_date:
            filtered = [i for i in filtered if i.detected_at <= end_date]

        scores = sorted([inc.similarity_score for inc in filtered])

        if not scores:
            return StatisticalSummary(
                count=0, mean=0.0, median=0.0, std_dev=0.0,
                min_value=0.0, max_value=0.0,
                percentile_25=0.0, percentile_75=0.0, percentile_90=0.0,
                iqr=0.0,
            )

        n = len(scores)
        mean_val = statistics.mean(scores)
        median_val = statistics.median(scores)
        std_val = statistics.stdev(scores) if n > 1 else 0.0

        def percentile(data: List[float], p: float) -> float:
            """Compute the p-th percentile."""
            k = (len(data) - 1) * p / 100.0
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            return data[f] * (c - k) + data[c] * (k - f)

        p25 = percentile(scores, 25)
        p75 = percentile(scores, 75)
        p90 = percentile(scores, 90)

        return StatisticalSummary(
            count=n,
            mean=round(mean_val, 4),
            median=round(median_val, 4),
            std_dev=round(std_val, 4),
            min_value=round(scores[0], 4),
            max_value=round(scores[-1], 4),
            percentile_25=round(p25, 4),
            percentile_75=round(p75, 4),
            percentile_90=round(p90, 4),
            iqr=round(p75 - p25, 4),
        )

    def compute_severity_distribution(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> SeverityDistribution:
        """Compute overall severity distribution."""
        filtered = self._incidents
        if start_date:
            filtered = [i for i in filtered if i.detected_at >= start_date]
        if end_date:
            filtered = [i for i in filtered if i.detected_at <= end_date]

        counts = Counter(inc.severity.lower() for inc in filtered)
        total = len(filtered)

        return SeverityDistribution(
            low=counts.get("low", 0),
            medium=counts.get("medium", 0),
            high=counts.get("high", 0),
            critical=counts.get("critical", 0),
            total=total,
        )

    # ── Offender Analysis ──────────────────────────────────────────────────

    def get_top_offenders(self, top_n: int = 10) -> List[OffenderProfile]:
        """Identify documents with the most plagiarism incidents."""
        doc_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "scores": [],
            "matchees": set(),
            "first": None,
            "last": None,
            "severities": [],
        })

        for inc in self._incidents:
            d = doc_data[inc.document_name]
            d["count"] += 1
            d["scores"].append(inc.similarity_score)
            d["matchees"].add(inc.matched_against)
            d["severities"].append(inc.severity.lower())
            if d["first"] is None or inc.detected_at < d["first"]:
                d["first"] = inc.detected_at
            if d["last"] is None or inc.detected_at > d["last"]:
                d["last"] = inc.detected_at

        profiles = []
        for doc_name, data in doc_data.items():
            profiles.append(OffenderProfile(
                document_name=doc_name,
                incident_count=data["count"],
                avg_similarity=round(statistics.mean(data["scores"]), 4),
                max_similarity=round(max(data["scores"]), 4),
                first_detected=data["first"],
                last_detected=data["last"],
                unique_matchees=sorted(data["matchees"]),
                severity_history=data["severities"],
            ))

        profiles.sort(key=lambda p: (-p.incident_count, -p.max_similarity))
        return profiles[:top_n]

    def get_repeat_offense_rate(self) -> float:
        """Compute the fraction of documents with 2+ incidents."""
        if not self._incidents:
            return 0.0
        doc_counts = Counter(inc.document_name for inc in self._incidents)
        repeat_docs = sum(1 for c in doc_counts.values() if c >= 2)
        return round(repeat_docs / len(doc_counts), 4) if doc_counts else 0.0

    # ── Growth Rate ────────────────────────────────────────────────────────

    def compute_monthly_growth_rate(self) -> float:
        """Compute the average month-over-month growth rate in incidents."""
        windows = self.aggregate_by_window(TimeWindow.MONTHLY)
        if len(windows) < 2:
            return 0.0

        counts = [w.incident_count for w in windows]
        growth_rates = []
        for i in range(1, len(counts)):
            prev = counts[i - 1]
            curr = counts[i]
            if prev > 0:
                growth_rates.append((curr - prev) / prev)
            elif curr > 0:
                growth_rates.append(1.0)
            else:
                growth_rates.append(0.0)

        return round(statistics.mean(growth_rates), 4) if growth_rates else 0.0

    # ── Moving Average ─────────────────────────────────────────────────────

    def compute_moving_average(
        self,
        window: Optional[TimeWindow] = None,
        period: int = 3,
    ) -> List[TimeSeriesPoint]:
        """Compute a simple moving average of incident counts."""
        windows = self.aggregate_by_window(window)
        if not windows:
            return []

        counts = [w.incident_count for w in windows]
        result: List[TimeSeriesPoint] = []

        for i in range(len(counts)):
            start_idx = max(0, i - period + 1)
            window_slice = counts[start_idx:i + 1]
            avg = statistics.mean(window_slice)
            result.append(TimeSeriesPoint(
                timestamp=windows[i].start,
                value=round(avg, 2),
                label=windows[i].window_label,
            ))

        return result

    # ── Full Report ────────────────────────────────────────────────────────

    def generate_report(
        self,
        window: Optional[TimeWindow] = None,
        forecast_periods: int = 3,
    ) -> AnalyticsReport:
        """Generate a complete analytics report."""
        windows = self.aggregate_by_window(window)
        stats = self.compute_statistics()
        sev_dist = self.compute_severity_distribution()
        trend = self.compute_trend(window, forecast_periods)
        top_offenders = self.get_top_offenders()
        growth_rate = self.compute_monthly_growth_rate()
        repeat_rate = self.get_repeat_offense_rate()

        date_range = self.get_date_range()
        start = date_range[0] or datetime.now(timezone.utc)
        end = date_range[1] or datetime.now(timezone.utc)

        return AnalyticsReport(
            generated_at=datetime.now(timezone.utc),
            time_window=(window or self._default_window).value,
            total_incidents=len(self._incidents),
            date_range_start=start,
            date_range_end=end,
            statistical_summary=stats,
            severity_distribution=sev_dist,
            trend=trend,
            windows=windows,
            top_offenders=top_offenders,
            monthly_growth_rate=growth_rate,
            repeat_offense_rate=repeat_rate,
        )

    # ── Export ──────────────────────────────────────────────────────────────

    def export_json(self, report: Optional[AnalyticsReport] = None) -> str:
        """Export the analytics report as a JSON string."""
        if report is None:
            report = self.generate_report()
        return json.dumps(self._report_to_dict(report), indent=2, default=str)

    def export_csv(self, report: Optional[AnalyticsReport] = None) -> str:
        """Export the analytics report as a CSV string."""
        if report is None:
            report = self.generate_report()

        output = io.StringIO()
        writer = csv.writer(output)

        # Header section
        writer.writerow(["Plagiarism Trend Analytics Report"])
        writer.writerow(["Generated At", str(report.generated_at)])
        writer.writerow(["Time Window", report.time_window])
        writer.writerow(["Total Incidents", report.total_incidents])
        writer.writerow(["Date Range", f"{report.date_range_start} to {report.date_range_end}"])
        writer.writerow([])

        # Statistical Summary
        writer.writerow(["=== Statistical Summary ==="])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Count", report.statistical_summary.count])
        writer.writerow(["Mean", report.statistical_summary.mean])
        writer.writerow(["Median", report.statistical_summary.median])
        writer.writerow(["Std Dev", report.statistical_summary.std_dev])
        writer.writerow(["Min", report.statistical_summary.min_value])
        writer.writerow(["Max", report.statistical_summary.max_value])
        writer.writerow(["P25", report.statistical_summary.percentile_25])
        writer.writerow(["P75", report.statistical_summary.percentile_75])
        writer.writerow(["P90", report.statistical_summary.percentile_90])
        writer.writerow([])

        # Severity Distribution
        writer.writerow(["=== Severity Distribution ==="])
        writer.writerow(["Low", report.severity_distribution.low])
        writer.writerow(["Medium", report.severity_distribution.medium])
        writer.writerow(["High", report.severity_distribution.high])
        writer.writerow(["Critical", report.severity_distribution.critical])
        writer.writerow(["Total", report.severity_distribution.total])
        writer.writerow(["High+Crit Rate (%)", round(report.severity_distribution.high_rate, 2)])
        writer.writerow([])

        # Time Windows
        writer.writerow(["=== Time Window Breakdown ==="])
        writer.writerow(["Window", "Incidents", "Avg Sim", "Max Sim", "Unique Docs"])
        for tw in report.windows:
            writer.writerow([
                tw.window_label,
                tw.incident_count,
                tw.avg_similarity,
                tw.max_similarity,
                tw.unique_documents,
            ])
        writer.writerow([])

        # Trend
        writer.writerow(["=== Trend Analysis ==="])
        writer.writerow(["Direction", report.trend.direction.value])
        writer.writerow(["Slope", report.trend.slope])
        writer.writerow(["R-squared", report.trend.r_squared])
        writer.writerow(["P-value", report.trend.p_value])
        writer.writerow(["Confidence (%)", report.trend.confidence])
        writer.writerow(["Forecast", ", ".join(str(v) for v in report.trend.forecast_values)])
        writer.writerow([])

        # Top Offenders
        writer.writerow(["=== Top Offenders ==="])
        writer.writerow(["Document", "Incidents", "Avg Sim", "Max Sim", "First Detected", "Last Detected"])
        for off in report.top_offenders:
            writer.writerow([
                off.document_name,
                off.incident_count,
                off.avg_similarity,
                off.max_similarity,
                str(off.first_detected),
                str(off.last_detected),
            ])

        return output.getvalue()

    def _report_to_dict(self, report: AnalyticsReport) -> Dict[str, Any]:
        """Convert an AnalyticsReport to a serializable dictionary."""
        return {
            "generated_at": str(report.generated_at),
            "time_window": report.time_window,
            "total_incidents": report.total_incidents,
            "date_range": {
                "start": str(report.date_range_start),
                "end": str(report.date_range_end),
            },
            "statistical_summary": {
                "count": report.statistical_summary.count,
                "mean": report.statistical_summary.mean,
                "median": report.statistical_summary.median,
                "std_dev": report.statistical_summary.std_dev,
                "min": report.statistical_summary.min_value,
                "max": report.statistical_summary.max_value,
                "percentiles": {
                    "p25": report.statistical_summary.percentile_25,
                    "p75": report.statistical_summary.percentile_75,
                    "p90": report.statistical_summary.percentile_90,
                },
                "iqr": report.statistical_summary.iqr,
            },
            "severity_distribution": {
                "low": report.severity_distribution.low,
                "medium": report.severity_distribution.medium,
                "high": report.severity_distribution.high,
                "critical": report.severity_distribution.critical,
                "total": report.severity_distribution.total,
                "high_rate": round(report.severity_distribution.high_rate, 2),
            },
            "trend": {
                "direction": report.trend.direction.value,
                "slope": report.trend.slope,
                "intercept": report.trend.intercept,
                "r_squared": report.trend.r_squared,
                "p_value": report.trend.p_value,
                "confidence": report.trend.confidence,
                "forecast_values": report.trend.forecast_values,
                "forecast_timestamps": [str(t) for t in report.trend.forecast_timestamps],
            },
            "windows": [
                {
                    "label": tw.window_label,
                    "start": str(tw.start),
                    "end": str(tw.end),
                    "incident_count": tw.incident_count,
                    "avg_similarity": tw.avg_similarity,
                    "max_similarity": tw.max_similarity,
                    "severity": {
                        "low": tw.severity_dist.low,
                        "medium": tw.severity_dist.medium,
                        "high": tw.severity_dist.high,
                        "critical": tw.severity_dist.critical,
                    },
                    "unique_documents": tw.unique_documents,
                    "unique_matchees": tw.unique_matchees,
                }
                for tw in report.windows
            ],
            "top_offenders": [
                {
                    "document": off.document_name,
                    "incident_count": off.incident_count,
                    "avg_similarity": off.avg_similarity,
                    "max_similarity": off.max_similarity,
                    "first_detected": str(off.first_detected),
                    "last_detected": str(off.last_detected),
                    "unique_matchees": off.unique_matchees,
                }
                for off in report.top_offenders
            ],
            "monthly_growth_rate": report.monthly_growth_rate,
            "repeat_offense_rate": report.repeat_offense_rate,
        }
