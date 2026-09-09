"""Evaluate the complete pip-audit report against reviewed applicability evidence.

No findings are removed from the report. Assessments expire, apply to one exact
package version, and cease to apply when reviewed call sites or upstream fixes change.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def source_evidence(root: Path, package: str) -> dict[str, str]:
    evidence = {}
    for directory in ("src", "app", "scripts"):
        for path in (root / directory).rglob("*.py"):
            if path.name == "audit_dependencies.py":
                continue
            text = path.read_text(encoding="utf-8-sig")
            if package in text:
                evidence[path.relative_to(root).as_posix()] = hashlib.sha256(text.encode()).hexdigest()
    return evidence


def evaluate(report, assessments, *, root=ROOT, today=None):
    today = today or date.today()
    blocked, reviewed = [], []
    if not isinstance(report.get("dependencies"), list) or not report["dependencies"]:
        return ["Audit did not inspect any dependencies"], []
    for dependency in report["dependencies"]:
        if dependency.get("skip_reason"):
            blocked.append(f"{dependency['name']}: audit incomplete: {dependency['skip_reason']}")
            continue
        for finding in dependency.get("vulns", []):
            key = f"{dependency['name']}=={dependency['version']}: {finding['id']}"
            applicable = None
            for assessment in assessments:
                if (
                    assessment["package"] == dependency["name"]
                    and assessment["version"] == dependency["version"]
                    and assessment["vulnerability"] in {finding["id"], *finding.get("aliases", [])}
                    and assessment["status"] == "not_affected"
                    and today <= date.fromisoformat(assessment["review_by"])
                    and not finding.get("fix_versions")
                    and assessment["source_sha256"] == source_evidence(root, assessment["package"])
                ):
                    applicable = assessment
                    break
            if applicable:
                reviewed.append(f"{key}: not affected — {applicable['justification']}")
            else:
                blocked.append(key)
    return blocked, reviewed


def installed_inventory(distributions=None):
    """Pin every installed distribution, mapping only the official Torch CPU suffix.

    CPU wheels share PyTorch's public release/advisories. PyPI cannot resolve the
    +cpu index suffix. Preserve the installed version beside the audited version;
    unknown local builds remain unmodified and fail closed if they cannot be audited.
    """
    if distributions is None:
        distributions = [(d.metadata["Name"], d.version) for d in importlib.metadata.distributions()]
    inventory = {}
    for name, installed_version in distributions:
        name = canonicalize_name(name)
        parsed = Version(installed_version)
        audit_version = parsed.public if name == "torch" and parsed.local == "cpu" else installed_version
        item = {"name": name, "installed_version": installed_version, "audited_version": audit_version}
        if name in inventory and inventory[name] != item:
            raise ValueError(f"Conflicting installed versions for {name}")
        inventory[name] = item
    return [inventory[name] for name in sorted(inventory)]


def audit_installed(output):
    """Audit the full installed closure without asking pip to resolve it again."""
    inventory = installed_inventory()
    with tempfile.TemporaryDirectory(prefix="dependency-audit-") as directory:
        requirements = Path(directory) / "installed.txt"
        requirements.write_text("".join(f"{item['name']}=={item['audited_version']}\n" for item in inventory), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "pip_audit", "--no-deps", "--disable-pip",
             "-r", str(requirements), "--format=json", "--output", str(output)],
            check=False,
        )
    if result.returncode not in (0, 1) or not output.exists():
        return None
    report = json.loads(output.read_text(encoding="utf-8"))
    report["installed_inventory"] = inventory
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # A tool result must account for every pinned package, including skipped ones.
    inspected = {canonicalize_name(item["name"]) for item in report.get("dependencies", [])}
    expected = {item["name"] for item in inventory}
    if inspected != expected:
        raise ValueError(f"Incomplete dependency inventory: missing={sorted(expected-inspected)}, unexpected={sorted(inspected-expected)}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Remove stale reports so a failed scan cannot reuse a previous clean result.
    args.output.unlink(missing_ok=True)
    report = audit_installed(args.output)
    if report is None:
        return 2
    assessments = json.loads((ROOT / "security/dependency-assessments.json").read_text())["assessments"]
    blocked, reviewed = evaluate(report, assessments)
    for item in reviewed:
        print(item)
    for item in blocked:
        print(f"BLOCKED: {item}")
    print(f"Dependency audit: {len(blocked)} blocking; {len(reviewed)} reviewed not-affected findings. Full report: {args.output}")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
