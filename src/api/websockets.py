# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
src/api/websockets.py
---------------------
FastAPI WebSocket router for real-time collaborative document review.

Manages WebSocket connections, room assignments per document, and
broadcasts annotation events to all connected instructors in the
same review session.
"""

import json
import logging
from typing import Any, Dict, List, Set

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from src.db.annotations_db import (
    create_annotation,
    delete_annotation,
    get_annotations_for_document,
    resolve_annotation,
)
from src.models.annotations import (
    AnnotationCreate,
    AnnotationRecord,
    AnnotationType,
    WebSocketMessage,
    WebSocketMessageType,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections organized by document rooms."""

    def __init__(self):
        # Maps room_id (document_id) to a set of active WebSocket connections
        self.active_connections: dict[str, set[WebSocket]] = {}
        # Maps websocket to user_id for tracking who is in the room
        self.connection_users: dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, room_id: str, user_id: str):
        """Accept a new WebSocket connection and add it to the room."""
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = set()
        self.active_connections[room_id].add(websocket)
        self.connection_users[websocket] = user_id
        logger.info("User %s connected to room %s", user_id, room_id)

    def disconnect(self, websocket: WebSocket, room_id: str):
        """Remove a WebSocket connection from the room."""
        if room_id in self.active_connections:
            self.active_connections[room_id].discard(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]

        user_id = self.connection_users.pop(websocket, "unknown")
        logger.info("User %s disconnected from room %s", user_id, room_id)

    async def broadcast(self, room_id: str, message: WebSocketMessage):
        """Send a message to all connected clients in a specific room."""
        if room_id not in self.active_connections:
            return

        payload = message.model_dump_json()
        disconnected = set()

        for connection in self.active_connections[room_id]:
            try:
                await connection.send_text(payload)
            except Exception as e:
                logger.warning("Failed to send message to client: %s", e)
                disconnected.add(connection)

        # Clean up broken connections
        for conn in disconnected:
            self.disconnect(conn, room_id)

    def get_active_users(self, room_id: str) -> list[str]:
        """Get a list of active user IDs in a room."""
        if room_id not in self.active_connections:
            return []
        return [
            self.connection_users[conn]
            for conn in self.active_connections[room_id]
            if conn in self.connection_users
        ]


# Global manager instance
manager = ConnectionManager()


@router.websocket("/ws/review/{document_id}")
async def review_websocket_endpoint(websocket: WebSocket, document_id: str):
    """WebSocket endpoint for collaborative document review.

    Clients connect here to receive real-time updates when other
    instructors add highlights or comments to the same document.
    """
    # In a real app, user_id would be extracted from a JWT token in the query params
    # For this implementation, we expect it as a query parameter: ?user_id=xxx
    user_id = websocket.query_params.get("user_id", "anonymous")

    await manager.connect(websocket, document_id, user_id)

    # Send initial state: existing annotations and active users
    try:
        existing_annotations = get_annotations_for_document(document_id)
        initial_msg = WebSocketMessage(
            type=WebSocketMessageType.NEW_ANNOTATION,
            payload={
                "annotations": [a.model_dump() for a in existing_annotations],
                "initial_load": True,
            },
            room_id=document_id,
            sender_id="system",
        )
        await websocket.send_text(initial_msg.model_dump_json())

        # Broadcast user joined
        join_msg = WebSocketMessage(
            type=WebSocketMessageType.USER_JOINED,
            payload={
                "user_id": user_id,
                "active_users": manager.get_active_users(document_id),
            },
            room_id=document_id,
            sender_id=user_id,
        )
        await manager.broadcast(document_id, join_msg)

    except Exception as e:
        logger.error("Error sending initial state: %s", e)

    try:
        while True:
            # Receive message from client
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)

            msg_type = data.get("type")
            payload = data.get("payload", {})

            if msg_type == WebSocketMessageType.NEW_ANNOTATION.value:
                # Validate and persist the new annotation
                try:
                    ann_data = AnnotationCreate(**payload)
                    record = AnnotationRecord(**ann_data.model_dump())

                    if create_annotation(record):
                        broadcast_msg = WebSocketMessage(
                            type=WebSocketMessageType.NEW_ANNOTATION,
                            payload=record.model_dump(),
                            room_id=document_id,
                            sender_id=user_id,
                        )
                        await manager.broadcast(document_id, broadcast_msg)
                    else:
                        raise ValueError("Database insertion failed")

                except (ValidationError, ValueError) as e:
                    error_msg = WebSocketMessage(
                        type=WebSocketMessageType.ERROR,
                        payload={"message": str(e)},
                        room_id=document_id,
                        sender_id="system",
                    )
                    await websocket.send_text(error_msg.model_dump_json())

            elif msg_type == WebSocketMessageType.RESOLVE.value:
                ann_id = payload.get("annotation_id")
                if resolve_annotation(ann_id):
                    broadcast_msg = WebSocketMessage(
                        type=WebSocketMessageType.UPDATE_ANNOTATION,
                        payload={"id": ann_id, "is_resolved": True},
                        room_id=document_id,
                        sender_id=user_id,
                    )
                    await manager.broadcast(document_id, broadcast_msg)

            elif msg_type == WebSocketMessageType.DELETE.value:
                ann_id = payload.get("annotation_id")
                if delete_annotation(ann_id):
                    broadcast_msg = WebSocketMessage(
                        type=WebSocketMessageType.DELETE_ANNOTATION,
                        payload={"id": ann_id},
                        room_id=document_id,
                        sender_id=user_id,
                    )
                    await manager.broadcast(document_id, broadcast_msg)

            elif msg_type == WebSocketMessageType.CURSOR_MOVE.value:
                # Ephemeral cursor updates are broadcast but not persisted
                broadcast_msg = WebSocketMessage(
                    type=WebSocketMessageType.CURSOR_MOVE,
                    payload=payload,
                    room_id=document_id,
                    sender_id=user_id,
                )
                await manager.broadcast(document_id, broadcast_msg)

    except WebSocketDisconnect:
        manager.disconnect(websocket, document_id)
        # Broadcast user left
        leave_msg = WebSocketMessage(
            type=WebSocketMessageType.USER_LEFT,
            payload={
                "user_id": user_id,
                "active_users": manager.get_active_users(document_id),
            },
            room_id=document_id,
            sender_id=user_id,
        )
        await manager.broadcast(document_id, leave_msg)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        manager.disconnect(websocket, document_id)
