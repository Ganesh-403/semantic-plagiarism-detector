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
tests/api/test_websockets.py
----------------------------
Unit tests for the WebSocket connection manager and broadcast logic.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.websockets import ConnectionManager, WebSocketMessage, WebSocketMessageType


class TestConnectionManager:
    """Test suite for the WebSocket ConnectionManager."""

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self):
        """Verify connections are added and removed correctly."""
        manager = ConnectionManager()
        ws = AsyncMock()

        await manager.connect(ws, "room_1", "user_1")
        assert ws in manager.active_connections["room_1"]
        assert manager.connection_users[ws] == "user_1"

        manager.disconnect(ws, "room_1")
        assert ws not in manager.active_connections.get("room_1", set())
        assert ws not in manager.connection_users

    @pytest.mark.asyncio
    async def test_broadcast_to_room(self):
        """Verify messages are broadcast to all clients in a room."""
        manager = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        await manager.connect(ws1, "room_1", "user_1")
        await manager.connect(ws2, "room_1", "user_2")

        msg = WebSocketMessage(
            type=WebSocketMessageType.NEW_ANNOTATION,
            payload={"test": "data"},
            room_id="room_1",
            sender_id="user_1",
        )

        await manager.broadcast("room_1", msg)

        assert ws1.send_text.call_count == 1
        assert ws2.send_text.call_count == 1

        # Verify payload structure
        sent_data = json.loads(ws1.send_text.call_args[0][0])
        assert sent_data["type"] == "new_annotation"
        assert sent_data["payload"]["test"] == "data"

    @pytest.mark.asyncio
    async def test_broadcast_handles_broken_connection(self):
        """Verify broken connections are cleaned up during broadcast."""
        manager = ConnectionManager()
        ws_good = AsyncMock()
        ws_bad = AsyncMock()
        ws_bad.send_text.side_effect = Exception("Connection closed")

        await manager.connect(ws_good, "room_1", "user_1")
        await manager.connect(ws_bad, "room_1", "user_2")

        msg = WebSocketMessage(
            type=WebSocketMessageType.USER_JOINED,
            payload={},
            room_id="room_1",
            sender_id="system",
        )

        await manager.broadcast("room_1", msg)

        # Good connection received message
        assert ws_good.send_text.call_count == 1
        # Bad connection was removed
        assert ws_bad not in manager.active_connections.get("room_1", set())

    def test_get_active_users(self):
        """Verify active users list is returned correctly."""
        manager = ConnectionManager()
        # Manually inject for sync test
        ws1 = MagicMock()
        ws2 = MagicMock()
        manager.active_connections["room_1"] = {ws1, ws2}
        manager.connection_users[ws1] = "alice"
        manager.connection_users[ws2] = "bob"

        users = manager.get_active_users("room_1")
        assert set(users) == {"alice", "bob"}
