import os
import pygit2
import tempfile
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class GitRepositoryScanner:
    """
    Enterprise Git Repository Scanner.
    Natively traverses branches and commits to catch "incremental plagiarism"
    where a student copies code but creates fake commits to simulate organic development.
    """

    def __init__(self, repo_url: str, pat: Optional[str] = None):
        self.repo_url = repo_url
        self.pat = pat
        self.temp_dir = tempfile.mkdtemp()
        self.repo = None

    def clone_repository(self) -> None:
        """Clone the repository locally."""
        callbacks = None
        if self.pat:
            credentials = pygit2.UserPass("token", self.pat)
            callbacks = pygit2.RemoteCallbacks(credentials=credentials)
            
        try:
            self.repo = pygit2.clone_repository(self.repo_url, self.temp_dir, callbacks=callbacks)
            logger.info(f"Successfully cloned {self.repo_url}")
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            raise ValueError(f"Could not clone repository: {e}")

    def traverse_commits(self) -> List[Dict[str, Any]]:
        """Traverse the commit history of the cloned repo."""
        if not self.repo:
            raise ValueError("Repository not cloned yet.")
            
        commits = []
        for commit in self.repo.walk(self.repo.head.target, pygit2.GIT_SORT_TIME | pygit2.GIT_SORT_REVERSE):
            commits.append({
                "hash": str(commit.id),
                "author": commit.author.name,
                "email": commit.author.email,
                "message": commit.message,
                "time": datetime.fromtimestamp(commit.commit_time).isoformat(),
            })
        return commits

    def analyze_code_velocity(self) -> Dict[str, Any]:
        """Analyze code velocity between commits to flag incremental plagiarism."""
        if not self.repo:
            raise ValueError("Repository not cloned yet.")

        velocity_metrics = []
        previous_commit = None
        total_commits = 0
        
        for commit in self.repo.walk(self.repo.head.target, pygit2.GIT_SORT_TIME | pygit2.GIT_SORT_REVERSE):
            total_commits += 1
            if previous_commit:
                diff = self.repo.diff(previous_commit, commit)
                
                # Basic metrics
                additions = diff.stats.insertions
                deletions = diff.stats.deletions
                
                time_diff = commit.commit_time - previous_commit.commit_time
                if time_diff == 0:
                    time_diff = 1 # Avoid division by zero
                    
                lines_per_second = additions / time_diff
                
                flag = lines_per_second > 50  # Suspicious if > 50 lines per second (huge copy paste)
                
                velocity_metrics.append({
                    "from_commit": str(previous_commit.id),
                    "to_commit": str(commit.id),
                    "additions": additions,
                    "deletions": deletions,
                    "time_diff_seconds": time_diff,
                    "lines_per_second": lines_per_second,
                    "is_suspicious": flag
                })
                
            previous_commit = commit
            
        return {
            "total_commits": total_commits,
            "velocity": velocity_metrics,
            "suspicious_commits": [m for m in velocity_metrics if m["is_suspicious"]]
        }

    def cleanup(self) -> None:
        """Clean up the temporary directory."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.error(f"Failed to cleanup temp dir: {e}")
