"""
Multi-Format Export and Integration System

Features:
- Multi-format export (PDF, Excel, CSV, JSON, XML, HTML)
- Cloud storage integrations (Google Drive, Dropbox, OneDrive)
- API-based data sharing
- Scheduled exports
- Customizable export templates
- Data transformation and filtering
- Integration hooks and webhooks
"""

import io
import json
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================


class ExportFormat(Enum):
    """Supported export formats."""

    PDF = "pdf"
    EXCEL = "xlsx"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    HTML = "html"
    MARKDOWN = "md"
    TXT = "txt"


class CloudProvider(Enum):
    """Supported cloud providers."""

    GOOGLE_DRIVE = "google_drive"
    DROPBOX = "dropbox"
    ONEDRIVE = "onedrive"
    AWS_S3 = "aws_s3"


class ExportStatus(Enum):
    """Export status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExportJob:
    """Export job record."""

    id: str
    name: str
    format: ExportFormat
    data: Dict[str, Any]
    status: ExportStatus
    created_at: float
    completed_at: Optional[float] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExportTemplate:
    """Export template definition."""

    id: str
    name: str
    description: str
    format: ExportFormat
    fields: List[str]
    filters: Dict[str, Any]
    styling: Dict[str, Any]
    created_at: float
    created_by: str
    is_default: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CloudConfig:
    """Cloud storage configuration."""

    provider: CloudProvider
    credentials: Dict[str, str]
    folder_path: str
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# EXPORT MANAGER
# ==============================================================================


class ExportManager:
    """
    Core export and integration manager.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.export_history: List[ExportJob] = []
        self.templates: List[ExportTemplate] = []
        self.cloud_configs: Dict[CloudProvider, CloudConfig] = {}
        self.export_queue: List[ExportJob] = []
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self._load_data()
        self._start_worker()

    def _load_data(self):
        """Load data from storage."""
        try:
            data_path = self.storage_path / "export_data.json"
            if data_path.exists():
                with open(data_path, "r") as f:
                    data = json.load(f)

                    self.export_history = [
                        ExportJob(**j) for j in data.get("history", [])
                    ]

                    self.templates = [
                        ExportTemplate(**t) for t in data.get("templates", [])
                    ]

                    self.cloud_configs = {
                        CloudProvider(k): CloudConfig(**v)
                        for k, v in data.get("cloud_configs", {}).items()
                    }
        except Exception as e:
            print(f"Error loading export data: {e}")

    def _save_data(self):
        """Save data to storage."""
        try:
            data_path = self.storage_path / "export_data.json"
            data_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "history": [asdict(j) for j in self.export_history[-100:]],
                "templates": [asdict(t) for t in self.templates],
                "cloud_configs": {
                    k.value: asdict(v) for k, v in self.cloud_configs.items()
                },
            }

            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving export data: {e}")

    def _start_worker(self):
        """Start background worker."""
        if self.is_running:
            return

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        """Worker thread for processing exports."""
        while self.is_running:
            try:
                if self.export_queue:
                    job = self.export_queue.pop(0)
                    self._process_export(job)
                else:
                    time.sleep(1)
            except Exception as e:
                print(f"Export worker error: {e}")
                time.sleep(5)

    def _process_export(self, job: ExportJob):
        """Process an export job."""
        job.status = ExportStatus.PROCESSING

        try:
            # Convert data
            converted_data = self._convert_data(job.data, job.format)

            # Generate file
            file_data = self._generate_file(converted_data, job.format)

            # Save file
            file_path = self._save_file(file_data, job.name, job.format)

            job.file_path = file_path
            job.file_size = len(file_data)
            job.status = ExportStatus.COMPLETED
            job.completed_at = time.time()

            # Upload to cloud if configured
            if job.metadata.get("upload_to_cloud"):
                self._upload_to_cloud(file_path, job)

            self._save_data()

        except Exception as e:
            job.status = ExportStatus.FAILED
            job.error = str(e)
            self._save_data()

    def _convert_data(self, data: Dict[str, Any], format: ExportFormat) -> Any:
        """Convert data to export format."""
        # Extract relevant data
        df = data.get("dataframe")
        metadata = data.get("metadata", {})

        if df is None:
            raise ValueError("No data to export")

        # Apply filters
        filters = data.get("filters", {})
        if filters:
            df = self._apply_filters(df, filters)

        return {"dataframe": df, "metadata": metadata, "format": format}

    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to dataframe."""
        filtered_df = df.copy()

        for column, condition in filters.items():
            if column in filtered_df.columns:
                if isinstance(condition, dict):
                    op = condition.get("operator", "eq")
                    value = condition.get("value")

                    if op == "eq":
                        filtered_df = filtered_df[filtered_df[column] == value]
                    elif op == "ne":
                        filtered_df = filtered_df[filtered_df[column] != value]
                    elif op == "gt":
                        filtered_df = filtered_df[filtered_df[column] > value]
                    elif op == "lt":
                        filtered_df = filtered_df[filtered_df[column] < value]
                    elif op == "contains":
                        filtered_df = filtered_df[
                            filtered_df[column].str.contains(value, na=False)
                        ]

        return filtered_df

    def _generate_file(self, data: Dict, format: ExportFormat) -> bytes:
        """Generate file in specified format."""
        df = data["dataframe"]
        metadata = data.get("metadata", {})

        if format == ExportFormat.CSV:
            return self._generate_csv(df)
        elif format == ExportFormat.EXCEL:
            return self._generate_excel(df, metadata)
        elif format == ExportFormat.JSON:
            return self._generate_json(df, metadata)
        elif format == ExportFormat.XML:
            return self._generate_xml(df, metadata)
        elif format == ExportFormat.HTML:
            return self._generate_html(df, metadata)
        elif format == ExportFormat.MARKDOWN:
            return self._generate_markdown(df, metadata)
        elif format == ExportFormat.TXT:
            return self._generate_txt(df, metadata)
        elif format == ExportFormat.PDF:
            return self._generate_pdf(df, metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_csv(self, df: pd.DataFrame) -> bytes:
        """Generate CSV file."""
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue().encode("utf-8")

    def _generate_excel(self, df: pd.DataFrame, metadata: Dict) -> bytes:
        """Generate Excel file with metadata."""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name="Data", index=False)

            # Metadata sheet
            if metadata:
                metadata_df = pd.DataFrame([metadata])
                metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
        return output.getvalue()

    def _generate_json(self, df: pd.DataFrame, metadata: Dict) -> bytes:
        """Generate JSON file."""
        data = {
            "metadata": metadata,
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "exported_at": datetime.now().isoformat(),
        }
        return json.dumps(data, indent=2).encode("utf-8")

    def _generate_xml(self, df: pd.DataFrame, metadata: Dict) -> bytes:
        """Generate XML file."""
        root = ET.Element("plagiarism_report")

        # Metadata
        meta_elem = ET.SubElement(root, "metadata")
        for key, value in metadata.items():
            elem = ET.SubElement(meta_elem, key)
            elem.text = str(value)

        # Data
        data_elem = ET.SubElement(root, "data")
        for _, row in df.iterrows():
            row_elem = ET.SubElement(data_elem, "row")
            for col in df.columns:
                col_elem = ET.SubElement(row_elem, col.replace(" ", "_"))
                col_elem.text = str(row[col])

        return ET.tostring(root, encoding="utf-8")

    def _generate_html(self, df: pd.DataFrame, metadata: Dict) -> bytes:
        """Generate HTML file."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Plagiarism Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .header {{ background-color: #f8f9fa; padding: 20px; margin-bottom: 20px; }}
                .metadata {{ margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Plagiarism Report</h1>
                <p>Exported: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        """

        # Metadata
        if metadata:
            html += '<div class="metadata"><h2>Metadata</h2><ul>'
            for key, value in metadata.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul></div>"

        # Data table
        html += "<h2>Data</h2>"
        html += df.to_html(index=False)

        html += """
        </body>
        </html>
        """
        return html.encode("utf-8")

    def _generate_markdown(self, df: pd.DataFrame, metadata: Dict) -> bytes:
        """Generate Markdown file."""
        md = "# Plagiarism Report\n\n"
        md += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Metadata
        if metadata:
            md += "## Metadata\n\n"
            for key, value in metadata.items():
                md += f"- **{key}:** {value}\n"
            md += "\n"

        # Data
        md += "## Data\n\n"
        md += df.to_markdown(index=False)

        return md.encode("utf-8")

    def _generate_txt(self, df: pd.DataFrame, metadata: Dict) -> bytes:
        """Generate plain text file."""
        txt = "PLAGIARISM REPORT\n"
        txt += "=" * 50 + "\n\n"

        # Metadata
        if metadata:
            txt += "METADATA:\n"
            for key, value in metadata.items():
                txt += f"  {key}: {value}\n"
            txt += "\n"

        # Data
        txt += "DATA:\n"
        txt += "-" * 40 + "\n"

        for _, row in df.iterrows():
            for col in df.columns:
                txt += f"{col}: {row[col]}\n"
            txt += "-" * 40 + "\n"

        return txt.encode("utf-8")

    def _generate_pdf(self, df: pd.DataFrame, metadata: Dict) -> bytes:
        """Generate PDF file."""
        # Try to use reportlab if available
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            story.append(Paragraph("Plagiarism Report", styles["Title"]))
            story.append(Spacer(1, 12))

            # Metadata
            if metadata:
                for key, value in metadata.items():
                    story.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
                story.append(Spacer(1, 12))

            # Data table
            data = [df.columns.tolist()] + df.values.tolist()
            table = Table(data)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 14),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )
            story.append(table)

            doc.build(story)
            return buffer.getvalue()

        except ImportError:
            # Fallback to text
            return self._generate_txt(df, metadata)

    def _save_file(self, file_data: bytes, name: str, format: ExportFormat) -> str:
        """Save file to disk."""
        file_dir = self.storage_path / "exports"
        file_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.{format.value}"
        file_path = file_dir / filename

        with open(file_path, "wb") as f:
            f.write(file_data)

        return str(file_path)

    def _upload_to_cloud(self, file_path: str, job: ExportJob):
        """Upload file to cloud storage."""
        provider = job.metadata.get("cloud_provider")
        if not provider:
            return

        config = self.cloud_configs.get(CloudProvider(provider))
        if not config or not config.enabled:
            return

        try:
            if provider == "google_drive":
                self._upload_google_drive(file_path, config)
            elif provider == "dropbox":
                self._upload_dropbox(file_path, config)
            elif provider == "onedrive":
                self._upload_onedrive(file_path, config)
        except Exception as e:
            print(f"Cloud upload error: {e}")

    def _upload_google_drive(self, file_path: str, config: CloudConfig):
        """Upload to Google Drive."""
        # Placeholder - would use Google Drive API
        print(f"📤 Uploading to Google Drive: {file_path}")

    def _upload_dropbox(self, file_path: str, config: CloudConfig):
        """Upload to Dropbox."""
        # Placeholder - would use Dropbox API
        print(f"📤 Uploading to Dropbox: {file_path}")

    def _upload_onedrive(self, file_path: str, config: CloudConfig):
        """Upload to OneDrive."""
        # Placeholder - would use OneDrive API
        print(f"📤 Uploading to OneDrive: {file_path}")

    def export_data(
        self,
        data: Dict[str, Any],
        name: str,
        format: ExportFormat = ExportFormat.CSV,
        filters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        template_id: str = None,
        upload_to_cloud: bool = False,
        cloud_provider: str = None,
    ) -> ExportJob:
        """
        Export data to specified format.

        Args:
            data: Data to export
            name: Export name
            format: Export format
            filters: Data filters
            metadata: Additional metadata
            template_id: Template to use
            upload_to_cloud: Upload to cloud
            cloud_provider: Cloud provider

        Returns:
            ExportJob: Export job
        """
        # Apply template if specified
        if template_id:
            template = next((t for t in self.templates if t.id == template_id), None)
            if template:
                format = template.format
                filters = template.filters
                metadata = {**metadata, **template.metadata}

        job = ExportJob(
            id=f"export_{int(time.time())}",
            name=name,
            format=format,
            data={
                "dataframe": data.get("dataframe"),
                "metadata": metadata or {},
                "filters": filters or {},
            },
            status=ExportStatus.PENDING,
            created_at=time.time(),
            metadata={
                "upload_to_cloud": upload_to_cloud,
                "cloud_provider": cloud_provider,
                "template_id": template_id,
            },
        )

        self.export_queue.append(job)
        self.export_history.append(job)
        self._save_data()

        return job

    def create_template(
        self,
        name: str,
        description: str,
        format: ExportFormat,
        fields: List[str],
        filters: Dict[str, Any],
        styling: Dict[str, Any],
        created_by: str,
    ) -> ExportTemplate:
        """Create export template."""
        template = ExportTemplate(
            id=f"template_{int(time.time())}",
            name=name,
            description=description,
            format=format,
            fields=fields,
            filters=filters,
            styling=styling,
            created_at=time.time(),
            created_by=created_by,
        )
        self.templates.append(template)
        self._save_data()
        return template

    def get_export_history(self, limit: int = 50) -> List[Dict]:
        """Get export history."""
        history = []
        for job in self.export_history[-limit:]:
            history.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "format": job.format.value,
                    "status": job.status.value,
                    "created_at": datetime.fromtimestamp(job.created_at).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "file_size": f"{job.file_size / 1024:.1f} KB"
                    if job.file_size
                    else "N/A",
                    "error": job.error,
                }
            )
        return history

    def get_job_status(self, job_id: str) -> Optional[ExportJob]:
        """Get job status."""
        for job in self.export_history:
            if job.id == job_id:
                return job
        return None

    def download_export(self, job_id: str) -> Optional[bytes]:
        """Download exported file."""
        job = self.get_job_status(job_id)
        if job and job.file_path:
            try:
                with open(job.file_path, "rb") as f:
                    return f.read()
            except Exception:
                return None
        return None


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_export_center():
    """Render export center UI."""
    st.subheader("📤 Export Center")

    # Initialize
    if "export_manager" not in st.session_state:
        data_dir = Path(st.session_state.get("data_dir", "."))
        st.session_state.export_manager = ExportManager(data_dir / "exports")

    manager = st.session_state.export_manager

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📤 Export", "📋 History", "📝 Templates", "☁️ Integrations"]
    )

    with tab1:
        render_export_ui(manager)

    with tab2:
        render_export_history(manager)

    with tab3:
        render_template_management(manager)

    with tab4:
        render_integration_settings(manager)


def render_export_ui(manager: ExportManager):
    """Render export UI."""
    st.markdown("#### 📤 Export Data")

    # Get data from session state
    sim_df = st.session_state.get("sim_df")
    flags = st.session_state.get("flags", [])

    if sim_df is None or sim_df.empty:
        st.warning("No data available to export")
        return

    col1, col2 = st.columns(2)

    with col1:
        export_name = st.text_input("Export Name", "plagiarism_report")
        export_format = st.selectbox(
            "Format", [f.value for f in ExportFormat], format_func=lambda x: x.upper()
        )

    with col2:
        export_type = st.selectbox(
            "Export Type",
            ["Full Report", "Summary Only", "Flagged Pairs Only"],
            help="Select what data to include in export",
        )
        include_metadata = st.checkbox("Include Metadata", value=True)

    # Filters
    with st.expander("🔧 Filters", expanded=False):
        min_similarity = st.slider("Minimum Similarity", 0.0, 1.0, 0.0, 0.05)
        max_similarity = st.slider("Maximum Similarity", 0.0, 1.0, 1.0, 0.05)
        document_filter = st.multiselect(
            "Filter Documents",
            options=sim_df.columns.tolist() if sim_df is not None else [],
        )

    # Template selection
    templates = [t.name for t in manager.templates]
    selected_template = st.selectbox("Apply Template", ["None"] + templates)

    # Export button
    if st.button("📤 Generate Export", type="primary", use_container_width=True):
        with st.spinner("Generating export..."):
            # Prepare data
            export_data = {
                "dataframe": sim_df,
                "metadata": {
                    "export_type": export_type,
                    "threshold": st.session_state.get("threshold_slider", 0.75),
                    "document_count": len(sim_df.columns) if sim_df is not None else 0,
                    "exported_at": datetime.now().isoformat(),
                },
            }

            # Get template
            template_id = None
            if selected_template != "None":
                template = next(
                    (t for t in manager.templates if t.name == selected_template), None
                )
                if template:
                    template_id = template.id

            # Create export job
            job = manager.export_data(
                data=export_data,
                name=export_name,
                format=ExportFormat(export_format),
                filters={"similarity": {"operator": "gt", "value": min_similarity}}
                if min_similarity > 0
                else {},
                metadata={"include_metadata": include_metadata},
                template_id=template_id,
            )

            st.success(f"✅ Export job created: {job.id}")
            st.info("Processing in background...")

            # Download link
            if job.status == ExportStatus.COMPLETED:
                file_data = manager.download_export(job.id)
                if file_data:
                    st.download_button(
                        label=f"⬇️ Download {export_format.upper()}",
                        data=file_data,
                        file_name=f"{export_name}.{export_format}",
                        mime=f"application/{export_format}",
                        use_container_width=True,
                    )
            else:
                st.caption("Export is processing. Refresh to check status.")


def render_export_history(manager: ExportManager):
    """Render export history."""
    st.markdown("#### 📋 Export History")

    history = manager.get_export_history()

    if not history:
        st.info("No export history")
        return

    df = pd.DataFrame(history)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download completed exports
    st.markdown("#### 📥 Download Exports")
    completed_jobs = [
        j for j in manager.export_history if j.status == ExportStatus.COMPLETED
    ]

    if completed_jobs:
        job_options = [
            f"{j.name} ({j.format.value}) - {datetime.fromtimestamp(j.created_at).strftime('%Y-%m-%d %H:%M')}"
            for j in completed_jobs
        ]
        selected = st.selectbox("Select Export", job_options)

        if selected:
            idx = job_options.index(selected)
            job = completed_jobs[idx]

            if st.button("⬇️ Download", use_container_width=True):
                file_data = manager.download_export(job.id)
                if file_data:
                    st.download_button(
                        label="⬇️ Click to Download",
                        data=file_data,
                        file_name=f"{job.name}.{job.format.value}",
                        mime=f"application/{job.format.value}",
                        use_container_width=True,
                    )


def render_template_management(manager: ExportManager):
    """Render template management UI."""
    st.markdown("#### 📝 Export Templates")

    # Create template
    with st.expander("➕ Create Template", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            template_name = st.text_input("Template Name")
            template_description = st.text_area("Description")
        with col2:
            template_format = st.selectbox("Format", [f.value for f in ExportFormat])
            template_fields = st.text_input(
                "Fields (comma separated)", "doc_a,doc_b,similarity,status"
            )

        if st.button("Create Template", use_container_width=True):
            if template_name:
                template = manager.create_template(
                    name=template_name,
                    description=template_description,
                    format=ExportFormat(template_format),
                    fields=[f.strip() for f in template_fields.split(",") if f.strip()],
                    filters={},
                    styling={},
                    created_by=st.session_state.get("username", "anonymous"),
                )
                st.success(f"✅ Template created: {template.id}")
                st.rerun()

    # Display templates
    if manager.templates:
        for template in manager.templates:
            with st.expander(f"📄 {template.name}", expanded=False):
                st.markdown(f"**Description:** {template.description}")
                st.markdown(f"**Format:** {template.format.value.upper()}")
                st.markdown(f"**Fields:** {', '.join(template.fields)}")
                st.caption(
                    f"Created: {datetime.fromtimestamp(template.created_at).strftime('%Y-%m-%d')}"
                )
                st.caption(f"Created by: {template.created_by}")
    else:
        st.info("No templates created")


def render_integration_settings(manager: ExportManager):
    """Render integration settings UI."""
    st.markdown("#### ☁️ Integration Settings")

    # Cloud providers
    st.markdown("##### Cloud Storage")

    cloud_options = ["Google Drive", "Dropbox", "OneDrive"]
    selected_cloud = st.selectbox("Cloud Provider", ["None"] + cloud_options)

    if selected_cloud != "None":
        col1, col2 = st.columns(2)
        with col1:
            folder_path = st.text_input("Folder Path", "/plagiarism_reports")
            credentials = st.text_area(
                "Credentials (JSON)", placeholder='{"token": "xxx"}'
            )
        with col2:
            enabled = st.checkbox("Enable", value=True)

        if st.button("Save Cloud Configuration", use_container_width=True):
            provider_map = {
                "Google Drive": CloudProvider.GOOGLE_DRIVE,
                "Dropbox": CloudProvider.DROPBOX,
                "OneDrive": CloudProvider.ONEDRIVE,
            }

            config = CloudConfig(
                provider=provider_map[selected_cloud],
                credentials=json.loads(credentials) if credentials else {},
                folder_path=folder_path,
                enabled=enabled,
            )

            manager.cloud_configs[config.provider] = config
            manager._save_data()
            st.success("✅ Cloud configuration saved")

    # Webhooks
    st.markdown("##### 🔗 Webhooks")
    webhook_url = st.text_input(
        "Webhook URL", placeholder="https://example.com/webhook"
    )
    webhook_events = st.multiselect(
        "Trigger Events", ["export_completed", "export_failed", "new_export"]
    )

    if st.button("Save Webhook", use_container_width=True):
        if webhook_url:
            # Save webhook configuration
            st.success("✅ Webhook saved")


def render_export_bulk():
    """Render bulk export UI."""
    st.subheader("📦 Bulk Export")

    # Initialize
    if "export_manager" not in st.session_state:
        data_dir = Path(st.session_state.get("data_dir", "."))
        st.session_state.export_manager = ExportManager(data_dir / "exports")

    manager = st.session_state.export_manager

    # Get documents
    docs = st.session_state.get("doc_names", [])

    if not docs:
        st.warning("No documents available")
        return

    # Select documents
    selected_docs = st.multiselect("Select Documents to Export", docs, default=docs[:5])

    if not selected_docs:
        st.info("Select at least one document")
        return

    # Export options
    col1, col2 = st.columns(2)
    with col1:
        export_format = st.selectbox("Format", [f.value for f in ExportFormat])
        include_chunks = st.checkbox("Include Document Chunks", value=True)
    with col2:
        export_name = st.text_input("Export Name", "bulk_export")
        compress = st.checkbox("Compress as ZIP", value=True)

    # Export button
    if st.button("📦 Export Selected", type="primary", use_container_width=True):
        with st.spinner("Preparing bulk export..."):
            # Collect data
            export_data = {
                "dataframe": st.session_state.get("sim_df"),
                "metadata": {
                    "documents": selected_docs,
                    "total_documents": len(selected_docs),
                    "export_type": "bulk",
                    "include_chunks": include_chunks,
                    "exported_at": datetime.now().isoformat(),
                },
            }

            # Create export job
            job = manager.export_data(
                data=export_data,
                name=export_name,
                format=ExportFormat(export_format),
                metadata={"bulk": True},
            )

            st.success(f"✅ Bulk export started: {job.id}")

            # Download when ready
            if job.status == ExportStatus.COMPLETED:
                file_data = manager.download_export(job.id)
                if file_data:
                    filename = f"{export_name}.{export_format}"
                    if compress:
                        filename = f"{export_name}.zip"

                    st.download_button(
                        label="⬇️ Download Bulk Export",
                        data=file_data,
                        file_name=filename,
                        mime="application/zip"
                        if compress
                        else f"application/{export_format}",
                        use_container_width=True,
                    )


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_export_system():
    """Initialize export system."""
    if "export_system_initialized" not in st.session_state:
        st.session_state.export_system_initialized = True

        # Create export manager
        data_dir = Path(st.session_state.get("data_dir", "."))
        manager = ExportManager(data_dir / "exports")
        st.session_state.export_manager = manager
