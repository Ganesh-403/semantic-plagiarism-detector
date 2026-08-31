import csv
import io
from datetime import datetime

# Software version constant
VERSION = "1.4.2"


def export_to_csv(results, threshold=0.8, include_metadata=True):
    """
    Exports plagiarism scan results to a CSV string with optional metadata headers.
    """
    output = io.StringIO()

    if include_metadata:
        scan_date = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        output.write(f"# Plagiarism Report\n")
        output.write(f"# Scan Date: {scan_date}\n")
        output.write(f"# Threshold Used: {threshold}\n")
        output.write(f"# Software Version: {VERSION}\n")
        output.write(f"#\n")  # Blank separator comment line

    writer = csv.writer(output)

    # Standard CSV column headers
    writer.writerow(["doc_a", "doc_b", "similarity_score"])

    # Data rows
    for row in results:
        writer.writerow(
            [row.get("doc_a"), row.get("doc_b"), row.get("similarity_score")]
        )

    return output.getvalue()
