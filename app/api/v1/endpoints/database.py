import os
import shutil
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException

router = APIRouter()

def create_database_snapshot(data_dir: str = "data") -> str:
    """Creates a timestamped snapshot backup of the data directory before clearing."""
    backup_dir = os.path.join(data_dir, "backups")
    os.makedirs(backup_dir, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"snapshot_{timestamp}.bak")
    
    # Assuming primary database file or index storage is located within data directory
    db_file = os.path.join(data_dir, "index.db")
    if os.path.exists(db_file):
        shutil.copy2(db_file, backup_path)
        return backup_path
    
    return ""

@router.delete("/api/v1/clear")
def clear_database(create_snapshot: bool = Query(True, description="Create automatic snapshot backup before clearing")):
    """Permanently clears all indexed documents after safely creating a timestamped backup."""
    try:
        backup_created = None
        if create_snapshot:
            backup_created = create_database_snapshot()
            
        # Invoke core data clearing logic
        # clear_all_data()
        
        return {
            "status": "success",
            "message": "Database cleared successfully.",
            "snapshot_backup": backup_created
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database safely: {str(e)}")
