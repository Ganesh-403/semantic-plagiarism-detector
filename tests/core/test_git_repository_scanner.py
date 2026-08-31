import pytest
import os
from unittest.mock import patch, MagicMock
from src.core.git_repository_scanner import GitRepositoryScanner, EnterpriseGitPlagiarismVelocityAnalyzer

@patch("src.core.git_repository_scanner.pygit2")
def test_clone_repository(mock_pygit2):
    mock_repo = MagicMock()
    mock_pygit2.clone_repository.return_value = mock_repo
    
    scanner = GitRepositoryScanner("https://github.com/test/repo.git")
    scanner.clone_repository()
    
    mock_pygit2.clone_repository.assert_called_once()
    assert scanner.repo == mock_repo

def test_analyze_git_velocity_pass():
    analyzer = EnterpriseGitPlagiarismVelocityAnalyzer()
    assert analyzer.analyze_git_velocity_pass_10() is True
    assert analyzer.analyze_git_velocity_pass_250() is True
