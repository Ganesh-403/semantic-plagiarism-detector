"""
Background Clustering Router for Semantic Plagiarism Detector.
Offloads heavy clustering tasks (KMeans, Agglomerative) to FastAPI background tasks
to prevent Streamlit UI timeouts. (Issue #2811)
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uuid
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/clustering", tags=["clustering"])

# In-memory task store (Production mein ise Redis se replace kiya ja sakta hai)
task_store: Dict[str, Dict[str, Any]] = {}

class ClusteringRequest(BaseModel):
    vectors: List[List[float]]
    n_clusters: int
    method: str  # "kmeans" ya "agglomerative"

def perform_clustering(task_id: str, vectors: List[List[float]], n_clusters: int, method: str):
    """Background task function to perform heavy clustering computation."""
    try:
        task_store[task_id]["status"] = "processing"
        logger.info(f"Task {task_id}: Starting {method} clustering with {n_clusters} clusters.")
        
        # Import inside function to avoid blocking main thread on startup
        import numpy as np
        from sklearn.cluster import KMeans, AgglomerativeClustering
        
        X = np.array(vectors)
        if method.lower() == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters)
        
        # Heavy computation yahan background mein hoga
        labels = model.fit_predict(X).tolist()
        
        task_store[task_id]["status"] = "completed"
        task_store[task_id]["result"] = {"labels": labels}
        logger.info(f"Task {task_id}: Completed successfully.")
        
    except Exception as e:
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["error"] = str(e)
        logger.error(f"Task {task_id}: Failed with error {e}")

@router.post("/")
async def start_clustering(request: ClusteringRequest, background_tasks: BackgroundTasks):
    """Endpoint to trigger background clustering task."""
    task_id = str(uuid.uuid4())
    task_store[task_id] = {"status": "pending", "result": None, "error": None}
    
    # Background task queue mein daal dein
    background_tasks.add_task(
        perform_clustering,
        task_id,
        request.vectors,
        request.n_clusters,
        request.method
    )
    return {"task_id": task_id, "status": "pending"}

@router.get("/status/{task_id}")
async def get_cluster_status(task_id: str):
    """Endpoint to poll the status of a clustering task."""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_store[task_id]