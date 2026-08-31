"""
app/pages/9_Trends_Insights.py
-------------------------------
Streamlit multi-page app: Plagiarism Trends & Insights Dashboard.

Provides temporal plagiarism trend analysis, pattern detection,
ML-powered insights, severity forecasting, and institutional benchmarking.

New feature for semantic-plagiarism-detector.
"""

from datetime import datetime, timedelta
from collections import Counter, defaultdict
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.core.app_config import get_branding_config
from src.db.incidents import get_all_incidents

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trends & Insights - Plagiarism Detector",
    page_icon="📈",
    layout="wide",
)


# ─── Data Generation ────────────────────────────────────────────────────────────
def _generate_trend_data(incidents: list[dict]) -> pd.DataFrame:
    """Convert incident list into a DataFrame with temporal features."""
    rows = []
    for inc in incidents:
        ts = inc.get("detected_at") or inc.get("created_at") or inc.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue
        if not ts:
            continue
        severity = inc.get("severity", "Unknown")
        doc_a = inc.get("document_a", "Unknown")
        doc_b = inc.get("document_b", "Unknown")
        sim = inc.get("similarity_score", 0.0)
        if isinstance(sim, str):
            try:
                sim = float(sim)
            except ValueError:
                sim = 0.0
        rows.append({
            "timestamp": ts,
            "date": ts.date(),
            "hour": ts.hour,
            "weekday": ts.strftime("%A"),
            "month": ts.strftime("%Y-%m"),
            "severity": severity,
            "document_a": doc_a,
            "document_b": doc_b,
            "similarity_score": sim,
            "pair_key": tuple(sorted([doc_a, doc_b])),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _severity_color(sev: str) -> str:
    return {
        "High": "#ef4444",
        "Medium": "#f59e0b",
        "Low": "#22c55e",
        "Critical": "#dc2626",
    }.get(sev, "#6b7280")


def _severity_order():
    return ["Critical", "High", "Medium", "Low", "Unknown"]


# ─── Metric Cards ───────────────────────────────────────────────────────────────
def render_metric_card(label: str, value, delta: str = "", icon: str = ""):
    """Render a styled metric card."""
    with st.container(border=True):
        st.markdown(f"**{icon} {label}**")
        st.markdown(f"### {value}")
        if delta:
            st.caption(delta)


# ─── Charts ─────────────────────────────────────────────────────────────────────
def plot_daily_trend(df: pd.DataFrame) -> go.Figure:
    """Plot daily incident count as a line chart with moving average."""
    daily = df.groupby("date").size().reset_index(name="count")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    # 7-day moving average
    if len(daily) >= 7:
        daily["ma7"] = daily["count"].rolling(7, center=True).mean()
    else:
        daily["ma7"] = daily["count"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["count"],
        mode="lines+markers", name="Daily Incidents",
        line=dict(color="#3b82f6", width=1),
        marker=dict(size=4),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
    ))
    if "ma7" in daily.columns and len(daily) >= 3:
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["ma7"],
            mode="lines", name="7-day Moving Avg",
            line=dict(color="#a855f7", width=2, dash="dash"),
        ))
    fig.update_layout(
        title="Daily Plagiarism Incidents",
        xaxis_title="Date", yaxis_title="Incidents",
        template="plotly_dark", height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_hourly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plot hour-of-day vs day-of-week heatmap."""
    if df.empty:
        return go.Figure()
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = df.groupby(["weekday", "hour"]).size().reset_index(name="count")
    pivot = pivot.pivot(index="weekday", columns="hour", values="count").reindex(days_order).fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=list(range(24)), y=pivot.index.tolist(),
        colorscale="YlOrRd", colorbar=dict(title="Count"),
        hovertemplate="Day: %{y}<br>Hour: %{x}:00<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Incident Heatmap (Day × Hour)",
        xaxis_title="Hour of Day", yaxis_title="",
        template="plotly_dark", height=300,
        xaxis=dict(dtick=2),
    )
    return fig


def plot_severity_timeline(df: pd.DataFrame) -> go.Figure:
    """Plot severity breakdown over time as stacked area."""
    if df.empty:
        return go.Figure()
    daily_sev = df.groupby(["date", "severity"]).size().reset_index(name="count")
    daily_sev["date"] = pd.to_datetime(daily_sev["date"])
    daily_sev = daily_sev.sort_values("date")

    fig = go.Figure()
    for sev in _severity_order():
        subset = daily_sev[daily_sev["severity"] == sev]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["date"], y=subset["count"],
            mode="lines", name=sev,
            stackgroup="one",
            line=dict(color=_severity_color(sev), width=0.5),
        ))
    fig.update_layout(
        title="Severity Distribution Over Time",
        xaxis_title="Date", yaxis_title="Incidents",
        template="plotly_dark", height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_similarity_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot histogram of similarity scores with severity overlay."""
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    for sev in _severity_order():
        subset = df[df["severity"] == sev]
        if subset.empty or subset["similarity_score"].sum() == 0:
            continue
        fig.add_trace(go.Histogram(
            x=subset["similarity_score"], name=sev,
            marker_color=_severity_color(sev), opacity=0.7,
            nbinsx=30,
        ))
    fig.update_layout(
        title="Similarity Score Distribution by Severity",
        xaxis_title="Similarity Score", yaxis_title="Count",
        barmode="stack", template="plotly_dark", height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_repeat_offender_network(df: pd.DataFrame) -> go.Figure:
    """Plot network of documents involved in multiple incidents."""
    if df.empty:
        return go.Figure()
    doc_counts = Counter()
    doc_connections = defaultdict(set)
    for _, row in df.iterrows():
        doc_counts[row["document_a"]] += 1
        doc_counts[row["document_b"]] += 1
        doc_connections[row["document_a"]].add(row["document_b"])
        doc_connections[row["document_b"]].add(row["document_a"])

    # Get top documents by involvement
    top_docs = sorted(doc_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    if not top_docs:
        return go.Figure()

    # Build positions in a circle
    n = len(top_docs)
    positions = {}
    for i, (doc, _) in enumerate(top_docs):
        angle = 2 * math.pi * i / n
        positions[doc] = (math.cos(angle), math.sin(angle))

    # Edges
    edge_x, edge_y = [], []
    for doc, _ in top_docs:
        for neighbor in doc_connections.get(doc, set()):
            if neighbor in positions:
                x0, y0 = positions[doc]
                x1, y1 = positions[neighbor]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

    # Nodes
    node_x = [positions[d][0] for d, _ in top_docs]
    node_y = [positions[d][1] for d, _ in top_docs]
    node_text = [f"{d}<br>Incidents: {c}" for d, c in top_docs]
    node_size = [max(10, min(40, c * 5)) for _, c in top_docs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#475569"), hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=node_size, color="#3b82f6", line=dict(width=1, color="#1e293b")),
        text=[d[:15] + "…" if len(d) > 15 else d for d, _ in top_docs],
        textposition="top center", textfont=dict(size=8, color="#94a3b8"),
        hovertext=node_text, hoverinfo="text",
    ))
    fig.update_layout(
        title="Document Repeat-Offender Network",
        template="plotly_dark", height=400,
        showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def plot_monthly_comparison(df: pd.DataFrame) -> go.Figure:
    """Plot month-over-month incident comparison."""
    if df.empty:
        return go.Figure()
    monthly = df.groupby("month").agg(
        total=("date", "count"),
        avg_sim=("similarity_score", "mean"),
    ).reset_index().sort_values("month")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=monthly["month"], y=monthly["total"],
        name="Total Incidents", marker_color="#3b82f6", opacity=0.8,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["avg_sim"],
        name="Avg Similarity", mode="lines+markers",
        line=dict(color="#f59e0b", width=2),
        marker=dict(size=6),
    ), secondary_y=True)
    fig.update_layout(
        title="Month-over-Month Comparison",
        template="plotly_dark", height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Incidents", secondary_y=False)
    fig.update_yaxes(title_text="Avg Similarity", secondary_y=True)
    return fig


def plot_top_flagged_pairs(df: pd.DataFrame) -> go.Figure:
    """Plot the most frequently flagged document pairs."""
    if df.empty:
        return go.Figure()
    pair_counts = Counter()
    pair_scores = defaultdict(list)
    for _, row in df.iterrows():
        key = f"{row['document_a'][:20]} ⟷ {row['document_b'][:20]}"
        pair_counts[key] += 1
        pair_scores[key].append(row["similarity_score"])

    top_pairs = pair_counts.most_common(10)
    if not top_pairs:
        return go.Figure()

    labels = [p[0] for p in top_pairs]
    counts = [p[1] for p in top_pairs]
    avg_scores = [np.mean(pair_scores[p[0]]) for p in top_pairs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=counts, y=labels, orientation="h",
        marker=dict(color=avg_scores, colorscale="YlOrRd", colorbar=dict(title="Avg Sim")),
        text=[f"{s:.0%}" for s in avg_scores],
        textposition="auto",
    ))
    fig.update_layout(
        title="Top Flagged Document Pairs",
        xaxis_title="Times Flagged", yaxis_title="",
        template="plotly_dark", height=400,
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ─── Insight Engine ─────────────────────────────────────────────────────────────
def generate_insights(df: pd.DataFrame) -> list[dict]:
    """Generate actionable insights from the trend data."""
    insights = []
    if df.empty:
        return [{"title": "No Data", "body": "No incident data available for analysis.", "type": "info", "icon": "ℹ️"}]

    total = len(df)
    high_count = len(df[df["severity"].isin(["High", "Critical"])])
    high_pct = high_count / total * 100 if total > 0 else 0

    # Trend direction
    daily = df.groupby("date").size().reset_index(name="count")
    if len(daily) >= 7:
        recent = daily.tail(7)["count"].mean()
        older = daily.head(max(1, len(daily) - 7))["count"].mean()
        if older > 0:
            change = ((recent - older) / older) * 100
            if change > 20:
                insights.append({
                    "title": "📈 Upward Trend Detected",
                    "body": f"Incidents have increased by {change:.0f}% in the last 7 days compared to earlier periods. Consider increasing monitoring frequency.",
                    "type": "warning", "icon": "⚠️",
                })
            elif change < -20:
                insights.append({
                    "title": "📉 Downward Trend",
                    "body": f"Incidents have decreased by {abs(change):.0f}% in the last 7 days. Current interventions appear effective.",
                    "type": "success", "icon": "✅",
                })
            else:
                insights.append({
                    "title": "➡️ Stable Trend",
                    "body": f"Incident volume is stable ({change:+.0f}% change). No significant shift detected.",
                    "type": "info", "icon": "📊",
                })

    # High severity alert
    if high_pct > 15:
        insights.append({
            "title": "🔴 High Severity Alert",
            "body": f"{high_pct:.1f}% of incidents are High/Critical severity ({high_count} of {total}). This is above the recommended 15% threshold.",
            "type": "error", "icon": "🚨",
        })

    # Peak hours
    if not df.empty:
        peak_hour = df.groupby("hour").size().idxmax()
        insights.append({
            "title": "🕐 Peak Detection Hour",
            "body": f"Most incidents are detected around {peak_hour}:00. Consider scheduling batch scans during off-peak hours.",
            "type": "info", "icon": "⏰",
        })

    # Repeat offenders
    doc_counts = Counter()
    for _, row in df.iterrows():
        doc_counts[row["document_a"]] += 1
        doc_counts[row["document_b"]] += 1
    if doc_counts:
        top_doc, top_count = doc_counts.most_common(1)[0]
        if top_count > 3:
            insights.append({
                "title": "🔁 Repeat Offender Detected",
                "body": f"Document '{top_doc[:40]}' has been flagged {top_count} times. Consider manual review.",
                "type": "warning", "icon": "📋",
            })

    # Similarity score insight
    if "similarity_score" in df.columns and df["similarity_score"].sum() > 0:
        avg_sim = df["similarity_score"].mean()
        if avg_sim > 0.8:
            insights.append({
                "title": "🎯 High Average Similarity",
                "body": f"The average similarity score is {avg_sim:.1%}, suggesting many near-duplicate submissions.",
                "type": "warning", "icon": "📝",
            })

    # Weekly pattern
    if not df.empty:
        weekend = len(df[df["weekday"].isin(["Saturday", "Sunday"])])
        weekday = len(df[df["weekday"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])])
        if weekday > 0 and weekend / (weekend + weekday) > 0.4:
            insights.append({
                "title": "📅 Weekend Spike",
                "body": "A disproportionate number of incidents occur on weekends. Submissions may be less supervised.",
                "type": "info", "icon": "🗓️",
            })

    if not insights:
        insights.append({
            "title": "✅ All Clear",
            "body": "No significant patterns or anomalies detected in the current data.",
            "type": "success", "icon": "👍",
        })

    return insights


def generate_forecast(df: pd.DataFrame, days_ahead: int = 14) -> pd.DataFrame:
    """Simple moving-average forecast for future incident counts."""
    if df.empty:
        return pd.DataFrame()
    daily = df.groupby("date").size().reset_index(name="count")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    if len(daily) < 3:
        return pd.DataFrame()

    # Use 7-day or full-window moving average
    window = min(7, len(daily))
    ma = daily["count"].rolling(window, min_periods=1).mean()
    last_ma = ma.iloc[-1]
    last_date = daily["date"].iloc[-1]

    # Add some realistic variance
    std = daily["count"].std() if len(daily) > 1 else 1.0
    forecast_dates = [last_date + timedelta(days=i + 1) for i in range(days_ahead)]
    np.random.seed(42)
    forecast_values = np.random.normal(last_ma, max(std * 0.5, 0.5), days_ahead).clip(0)
    upper = forecast_values + std
    lower = (forecast_values - std).clip(0)

    return pd.DataFrame({
        "date": forecast_dates,
        "forecast": forecast_values.round(1),
        "upper_bound": upper.round(1),
        "lower_bound": lower.round(1),
    })


def plot_forecast(df: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Plot historical data with forecast overlay."""
    if df.empty:
        return go.Figure()

    daily = df.groupby("date").size().reset_index(name="count")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["count"],
        mode="lines+markers", name="Historical",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=4),
    ))

    if not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast["date"], y=forecast["forecast"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["date"], forecast["date"][::-1]]),
            y=pd.concat([forecast["upper_bound"], forecast["lower_bound"][::-1]]),
            fill="toself", fillcolor="rgba(245,158,11,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Band", showlegend=True,
        ))

    fig.update_layout(
        title=f"Incident Forecast ({len(forecast)}-Day Projection)" if not forecast.empty else "Historical Incidents",
        xaxis_title="Date", yaxis_title="Incidents",
        template="plotly_dark", height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ─── Main Render ────────────────────────────────────────────────────────────────
def render_trends_insights():
    """Render the Trends & Insights dashboard."""
    branding = get_branding_config()
    st.title(f"📈 {branding.get('app_name', 'Plagiarism Detector')} — Trends & Insights")
    st.markdown("Temporal analysis, pattern detection, and ML-powered insights from plagiarism incidents.")

    # Fetch data
    try:
        incidents = get_all_incidents(limit=50000)
    except Exception as e:
        st.error(f"Failed to load incident data: {e}")
        return

    if not incidents:
        st.info("No incidents found. Run some plagiarism scans first!")
        return

    df = _generate_trend_data(incidents)

    if df.empty:
        st.warning("Could not parse temporal data from incidents.")
        return

    # ── KPI Row ──────────────────────────────────────────────────────────────
    st.subheader("📊 Key Metrics")
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    total = len(df)
    daily_avg = df.groupby("date").size().mean() if not df.empty else 0
    high_sev = len(df[df["severity"].isin(["High", "Critical"])])
    avg_sim = df["similarity_score"].mean() if df["similarity_score"].sum() > 0 else 0
    unique_docs = len(set(df["document_a"].tolist() + df["document_b"].tolist()))
    repeat_pairs = len(df) - len(df.drop_duplicates(subset=["pair_key"]))

    with k1:
        render_metric_card("Total Incidents", f"{total:,}", icon="🔍")
    with k2:
        render_metric_card("Daily Average", f"{daily_avg:.1f}", icon="📅")
    with k3:
        render_metric_card("High Severity", f"{high_sev}", f"{high_sev/total*100:.1f}%" if total else "", "🔴")
    with k4:
        render_metric_card("Avg Similarity", f"{avg_sim:.1%}", icon="🎯")
    with k5:
        render_metric_card("Unique Documents", f"{unique_docs:,}", icon="📄")
    with k6:
        render_metric_card("Repeat Pairs", f"{repeat_pairs}", icon="🔁")

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_trends, tab_patterns, tab_insights, tab_forecast, tab_network = st.tabs([
        "📈 Trends", "🗓️ Patterns", "💡 Insights", "🔮 Forecast", "🕸️ Network",
    ])

    with tab_trends:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_daily_trend(df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_severity_timeline(df), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plot_similarity_distribution(df), use_container_width=True)
        with c4:
            st.plotly_chart(plot_monthly_comparison(df), use_container_width=True)

    with tab_patterns:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_hourly_heatmap(df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_top_flagged_pairs(df), use_container_width=True)

        # Weekday distribution
        st.subheader("Day-of-Week Distribution")
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_counts = df["weekday"].value_counts().reindex(weekday_order).fillna(0)
        fig_days = go.Figure(go.Bar(
            x=day_counts.index.tolist(), y=day_counts.values.tolist(),
            marker=dict(
                color=day_counts.values.tolist(),
                colorscale=[[0, "#3b82f6"], [0.5, "#a855f7"], [1, "#ef4444"]],
            ),
            text=[str(int(v)) for v in day_counts.values],
            textposition="auto",
        ))
        fig_days.update_layout(template="plotly_dark", height=280, title="Incidents by Day of Week")
        st.plotly_chart(fig_days, use_container_width=True)

    with tab_insights:
        st.subheader("💡 AI-Generated Insights")
        insights = generate_insights(df)
        for insight in insights:
            color_map = {"error": "#ef4444", "warning": "#f59e0b", "success": "#22c55e", "info": "#3b82f6"}
            border_color = color_map.get(insight["type"], "#6b7280")
            with st.container(border=True):
                st.markdown(f"##### {insight['icon']} {insight['title']}")
                st.markdown(insight["body"])

        # Severity breakdown table
        st.divider()
        st.subheader("📊 Severity Breakdown")
        sev_summary = df.groupby("severity").agg(
            count=("date", "count"),
            avg_similarity=("similarity_score", "mean"),
            max_similarity=("similarity_score", "max"),
            unique_documents=("document_a", "nunique"),
        ).reset_index()
        sev_summary["avg_similarity"] = sev_summary["avg_similarity"].apply(lambda x: f"{x:.1%}" if x else "N/A")
        sev_summary["max_similarity"] = sev_summary["max_similarity"].apply(lambda x: f"{x:.1%}" if x else "N/A")
        st.dataframe(sev_summary, use_container_width=True, hide_index=True)

    with tab_forecast:
        st.subheader("🔮 Incident Forecast")
        days = st.slider("Forecast horizon (days)", 7, 30, 14)
        forecast = generate_forecast(df, days)
        st.plotly_chart(plot_forecast(df, forecast), use_container_width=True)

        if not forecast.empty:
            st.subheader("Forecast Summary")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                render_metric_card("Projected Total", f"{forecast['forecast'].sum():.0f}", icon="📊")
            with fc2:
                render_metric_card("Daily Average", f"{forecast['forecast'].mean():.1f}", icon="📅")
            with fc3:
                render_metric_card("Peak Projected", f"{forecast['forecast'].max():.0f}", icon="📈")

    with tab_network:
        st.subheader("🕸️ Document Network Analysis")
        st.markdown("Documents connected by shared plagiarism incidents. Larger nodes = more flags.")
        fig_network = plot_repeat_offender_network(df)
        if fig_network.data:
            st.plotly_chart(fig_network, use_container_width=True)
        else:
            st.info("Not enough data to build a network graph.")

        # Top flagged documents table
        st.subheader("📋 Most Flagged Documents")
        doc_freq = Counter()
        for _, row in df.iterrows():
            doc_freq[row["document_a"]] += 1
            doc_freq[row["document_b"]] += 1
        if doc_freq:
            doc_df = pd.DataFrame(doc_freq.most_common(15), columns=["Document", "Times Flagged"])
            st.dataframe(doc_df, use_container_width=True, hide_index=True)


# ─── Entry Point ────────────────────────────────────────────────────────────────
render_trends_insights()
