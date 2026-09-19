"""
Session Memory Manager for Multimodal Food Agent.

Maintains multi-turn conversational context, image uploads, classification outputs,
and retrieved nutritional data per session in a thread-safe in-memory store.
"""
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SessionState:
    """Represents the complete state of an ongoing user interaction session."""
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    last_image_b64: Optional[str] = None
    last_classification: Optional[Dict[str, Any]] = None
    last_nutrition: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session state to serializable dictionary."""
        return asdict(self)


class SessionMemory:
    """
    Thread-safe in-memory session manager.
    Tracks dialogue history, images, and nutritional context across multiple turns.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionState] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionState:
        """
        Retrieve an existing session by ID or create a new session if not found or omitted.

        Args:
            session_id: Optional existing session ID string.

        Returns:
            SessionState instance.
        """
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]

            sid = session_id.strip() if (session_id and session_id.strip()) else uuid.uuid4().hex
            new_session = SessionState(session_id=sid)
            self._sessions[sid] = new_session
            return new_session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Get session state by ID if it exists.

        Args:
            session_id: ID of the session to look up.

        Returns:
            SessionState if found, else None.
        """
        with self._lock:
            return self._sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Append a conversation message to the session history.

        Args:
            session_id: Session identifier.
            role: 'user' or 'assistant'.
            content: Text message content.
        """
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)

            session = self._sessions[session_id]
            session.messages.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            session.updated_at = datetime.now(timezone.utc).isoformat()

    def update_session_data(
        self,
        session_id: str,
        classification: Optional[Dict[str, Any]] = None,
        nutrition: Optional[Dict[str, Any]] = None,
        image_b64: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update cached food classification, nutrition facts, or image reference for a session.
        """
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)

            session = self._sessions[session_id]
            if classification is not None:
                session.last_classification = classification
            if nutrition is not None:
                session.last_nutrition = nutrition
            if image_b64 is not None:
                session.last_image_b64 = image_b64
            if metadata is not None:
                session.metadata.update(metadata)
            session.updated_at = datetime.now(timezone.utc).isoformat()

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Retrieve messages for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of recent messages to return.

        Returns:
            List of message dicts [{"role": "...", "content": "..."}].
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            msgs = session.messages
            if limit and limit > 0:
                msgs = msgs[-limit:]
            return list(msgs)

    def clear_session(self, session_id: str) -> bool:
        """
        Delete a session from memory.

        Args:
            session_id: Target session ID.

        Returns:
            True if session existed and was removed, False otherwise.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> List[str]:
        """Return a list of all active session IDs."""
        with self._lock:
            return list(self._sessions.keys())


# Singleton instance for default memory store across the application
default_session_memory = SessionMemory()
