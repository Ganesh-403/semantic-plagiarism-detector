"""Audit CPU wheels without losing installed versions or transitive packages."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import audit_dependencies as audit

@pytest.mark.parametrize("package,version,expected", [
    ("torch", "2.14.0+cpu", "2.14.0"),
    ("torch", "2.14.0", "2.14.0"),
    ("torch", "2.14.0+custom", "2.14.0+custom"),
    ("other", "1.0+cpu", "1.0+cpu"),
])
def test_only_official_cpu_variant_maps_to_public_advisories(package, version, expected):
    assert audit.installed_inventory([(package, version)]) == [
        {"name": package, "installed_version": version, "audited_version": expected}]

def test_duplicate_metadata_is_deterministic_and_conflicts_fail():
    assert len(audit.installed_inventory([("Some_Pkg", "1.0"), ("some-pkg", "1.0")])) == 1
    with pytest.raises(ValueError, match="Conflicting"):
        audit.installed_inventory([("some-pkg", "1.0"), ("some-pkg", "2.0")])

@pytest.fixture
def inventory(monkeypatch):
    result = audit.installed_inventory([("torch", "2.14.0+cpu"), ("setuptools", "79.0.1")])
    monkeypatch.setattr(audit, "installed_inventory", lambda: result)
    return result

def test_full_inventory_and_findings_survive_audit(tmp_path, monkeypatch, inventory):
    output = tmp_path / "audit.json"
    def run(command, **kwargs):
        assert "--no-deps" in command and "--disable-pip" in command
        requirements = Path(command[command.index("-r") + 1]).read_text()
        assert requirements.splitlines() == ["setuptools==79.0.1", "torch==2.14.0"]
        output.write_text(json.dumps({"dependencies": [
            {"name": "setuptools", "version": "79.0.1", "vulns": [{"id": "unfixed", "fix_versions": ["83.0.0"]}]},
            {"name": "torch", "version": "2.14.0", "vulns": [{"id": "torch-finding"}]},
        ]}))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(audit.subprocess, "run", run)
    report = audit.audit_installed(output)
    assert report["installed_inventory"] == inventory
    assert json.loads(output.read_text()) == report
    blocked, reviewed = audit.evaluate(report, [])
    assert len(blocked) == 2 and not reviewed

@pytest.mark.parametrize("dependencies", [[], [{"name": "extra", "version": "1", "vulns": []}]])
def test_missing_or_unexpected_packages_fail_closed(tmp_path, monkeypatch, inventory, dependencies):
    output = tmp_path / "audit.json"
    def run(*args, **kwargs):
        output.write_text(json.dumps({"dependencies": dependencies}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(audit.subprocess, "run", run)
    with pytest.raises(ValueError, match="Incomplete"):
        audit.audit_installed(output)

def test_audit_tool_failure_cannot_appear_clean(tmp_path, monkeypatch, inventory):
    monkeypatch.setattr(audit.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=2))
    assert audit.audit_installed(tmp_path / "missing.json") is None
