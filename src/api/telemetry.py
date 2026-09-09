from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import json


router = APIRouter()

class TelemetryConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def process_telemetry(self, data: dict):
        # In a real environment, save this to SQLite or Redis
        pass

manager = TelemetryConnectionManager()

@router.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            telemetry = json.loads(data)
            await manager.process_telemetry(telemetry)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
