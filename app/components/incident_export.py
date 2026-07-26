from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from src.db.incidents import (DEFAULT_DB_PATH, get_all_incidents,
                              incidents_to_csv, sync_flagged_incidents,
                              update_review_status)


def render_incident_export_panel(
    flags: Sequence[Mapping[str, Any]],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    st.subheader("🚨 Plagiarism Incident Log")
    st.caption(
        "Current warnings are synchronized into a persistent incident log. "
        "The first flagged date and review status are retained."
    )
    incidents = sync_flagged_incidents(flags, db_path)
    if not incidents:
        st.info("No plagiarism incidents are currently available for export.")
        return

    total = len(incidents)
    pending = sum(i["review_status"] == "Pending" for i in incidents)
    c1, c2, c3 = st.columns(3)
    c1.metric("All incidents", total)
    c2.metric("Pending review", pending)
    c3.metric("Resolved", total - pending)

    status_filter = st.selectbox(
        "Review-status filter",
        ["All", "Pending", "Resolved"],
        key="incident_status_filter",
    )
    visible = [
        i
        for i in incidents
        if status_filter == "All" or i["review_status"] == status_filter
    ]
    rows = [
        {
            "Incident ID": i["incident_id"],
            "Document A": i["document_a"],
            "Document B": i["document_b"],
            "Similarity": f"{i['similarity_score']:.1%}",
            "Threshold at Flag": f"{i['threshold_at_time_of_flag']:.1%}",
            "Severity": i["severity_rank"],
            "Review Status": i["review_status"],
            "Date Flagged": i["date_flagged"],
        }
        for i in visible
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Update review status")
    incident_id = st.selectbox(
        "Incident",
        [i["incident_id"] for i in incidents],
        key="incident_review_id",
    )
    current = next(i for i in incidents if i["incident_id"] == incident_id)
    status = st.selectbox(
        "Status",
        ["Pending", "Resolved"],
        index=0 if current["review_status"] == "Pending" else 1,
        key="incident_review_status",
    )
    if st.button("Save review status", type="primary"):
        update_review_status(incident_id, status, db_path)
        st.success(f"✅ {incident_id} marked as {status}.")
        st.rerun()

    filename = (
        "plagiarism_incidents_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    )
    st.download_button(
        "⬇️ Download All Flagged Incidents CSV",
        data=incidents_to_csv(get_all_incidents(db_path)),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )
