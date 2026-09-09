"""Exercise the scanner's clone, history and velocity behavior."""
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from src.core.git_repository_scanner import GitRepositoryScanner


@pytest.fixture
def scanner():
    value = GitRepositoryScanner("https://github.com/test/repo.git")
    yield value
    value.cleanup()
    assert not Path(value.temp_dir).exists()


@patch("src.core.git_repository_scanner.pygit2")
def test_clone_repository(pygit2, scanner):
    scanner.clone_repository()
    pygit2.clone_repository.assert_called_once_with(scanner.repo_url, scanner.temp_dir, callbacks=None)
    assert scanner.repo is pygit2.clone_repository.return_value


def test_requires_clone(scanner):
    for method in (scanner.traverse_commits, scanner.analyze_code_velocity):
        with pytest.raises(ValueError, match="not cloned"):
            method()


def test_empty_history_has_zero_commits(scanner):
    scanner.repo = MagicMock()
    scanner.repo.walk.return_value = []
    assert scanner.analyze_code_velocity() == {"total_commits": 0, "velocity": [], "suspicious_commits": []}


def test_velocity_and_history(scanner):
    author = SimpleNamespace(name="Test Author", email="author@example.org")
    commits = [SimpleNamespace(id=str(i), author=author, message="Update", commit_time=t) for i,t in enumerate([100,100,200])]
    scanner.repo = MagicMock()
    scanner.repo.walk.side_effect = lambda *args: iter(commits)
    scanner.repo.diff.return_value.stats.insertions = 100
    scanner.repo.diff.return_value.stats.deletions = 3
    assert [row["hash"] for row in scanner.traverse_commits()] == ["0", "1", "2"]
    result = scanner.analyze_code_velocity()
    assert result["total_commits"] == 3
    assert [row["lines_per_second"] for row in result["velocity"]] == [100,1]
    assert len(result["suspicious_commits"]) == 1
