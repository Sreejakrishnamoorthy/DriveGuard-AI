"""
Database Service.

Handles all MongoDB persistence: sessions, risk_events, alerts, and
analysis_results. The application MUST continue working if MongoDB is
unavailable — every method checks connection state and degrades
gracefully (logs a warning, returns None/empty/False) instead of raising,
so callers never need special-case error handling for a down database.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from src.utils.config import Config
from src.utils.logger import logger

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    logger.warning("pymongo not installed. Database features will be unavailable.")

# MongoDB's sort-descending value is the plain integer -1 at the wire
# protocol level (pymongo.DESCENDING is just this constant). Defining it
# directly here means sort ordering still works correctly even when
# using a Mongo-compatible test double (e.g. mongomock) in an
# environment where pymongo itself isn't installed, instead of silently
# raising NameError from inside a broad except block.
DESCENDING = -1


class DatabaseService:
    """
    MongoDB persistence layer for sessions, risk events, alerts, and
    analysis results.

    Connection is attempted once at construction with a short server
    selection timeout, so a missing/unreachable MongoDB fails fast rather
    than hanging the application. If unavailable, `is_connected` is False
    and every method becomes a safe no-op.
    """

    def __init__(
        self,
        mongodb_uri: Optional[str] = None,
        database_name: Optional[str] = None,
        server_selection_timeout_ms: int = 3000,
        client: Optional[Any] = None,
    ) -> None:
        """
        Initialize the database service.

        Args:
            mongodb_uri: MongoDB connection string. Defaults to Config.MONGODB_URI.
            database_name: Database name. Defaults to Config.MONGODB_DATABASE.
            server_selection_timeout_ms: How long to wait before deciding
                MongoDB is unreachable.
            client: Optionally inject an existing Mongo-compatible client
                (e.g. a mongomock.MongoClient for testing) instead of
                creating a real pymongo connection.
        """
        self.mongodb_uri = mongodb_uri or Config.MONGODB_URI
        self.database_name = database_name or Config.MONGODB_DATABASE
        self.is_connected = False
        self.db = None
        self._client = None

        if client is not None:
            # Injected client (e.g. mongomock) — treat as connected without
            # a network ping, since test doubles don't implement one.
            self._client = client
            self.db = client[self.database_name]
            self.is_connected = True
            self._init_collections()
            logger.info("DatabaseService: using injected client (test/mock mode).")
            return

        if not PYMONGO_AVAILABLE:
            logger.warning("DatabaseService: pymongo unavailable — running without persistence.")
            return

        try:
            self._client = MongoClient(
                self.mongodb_uri, serverSelectionTimeoutMS=server_selection_timeout_ms
            )
            # MongoClient() doesn't actually connect until first use — ping
            # explicitly so we know NOW whether MongoDB is reachable, not
            # on the first real query from a request handler.
            self._client.admin.command("ping")
            self.db = self._client[self.database_name]
            self.is_connected = True
            self._init_collections()
            logger.info(f"DatabaseService: connected to MongoDB at {self.mongodb_uri}")

        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(
                f"DatabaseService: MongoDB unavailable ({e}). "
                "Application will continue without persistence."
            )
            self.is_connected = False
        except Exception as e:
            logger.error(f"DatabaseService: unexpected error connecting to MongoDB: {e}", exc_info=True)
            self.is_connected = False

    def _init_collections(self) -> None:
        """Set up collection handles once connected."""
        self.sessions = self.db["sessions"]
        self.risk_events = self.db["risk_events"]
        self.alerts = self.db["alerts"]
        self.analysis_results = self.db["analysis_results"]

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, session_data: Optional[Dict] = None) -> Optional[str]:
        """
        Create a new session record.

        Args:
            session_data: Optional initial fields to merge in (e.g. driver_status)

        Returns:
            The new session_id, or None if the database is unavailable.
        """
        if not self.is_connected:
            logger.warning("DatabaseService: create_session skipped — not connected.")
            return None

        session_id = str(uuid.uuid4())
        doc = {
            "session_id": session_id,
            "start_time": self._now_iso(),
            "end_time": None,
            "risk_score": 0,
            "risk_level": "LOW",
            "driver_status": "UNKNOWN",
            "road_status": "UNKNOWN",
            "alerts": [],
            "timestamps": {"created": self._now_iso()},
        }
        if session_data:
            doc.update(session_data)
            doc["session_id"] = session_id  # never let caller override the generated id

        try:
            self.sessions.insert_one(doc)
            logger.info(f"DatabaseService: session created ({session_id})")
            return session_id
        except Exception as e:
            logger.error(f"DatabaseService: failed to create session: {e}", exc_info=True)
            return None

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a single session by ID. Returns None if not found or unavailable."""
        if not self.is_connected:
            return None
        try:
            doc = self.sessions.find_one({"session_id": session_id}, {"_id": 0})
            return doc
        except Exception as e:
            logger.error(f"DatabaseService: failed to get session {session_id}: {e}", exc_info=True)
            return None

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        """List recent sessions, most recent first. Returns [] if unavailable."""
        if not self.is_connected:
            return []
        try:
            cursor = self.sessions.find({}, {"_id": 0}).sort("start_time", DESCENDING).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"DatabaseService: failed to list sessions: {e}", exc_info=True)
            return []

    def update_session(self, session_id: str, updates: Dict) -> bool:
        """Update fields on an existing session. Returns True if a document was modified."""
        if not self.is_connected:
            return False
        try:
            result = self.sessions.update_one({"session_id": session_id}, {"$set": updates})
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"DatabaseService: failed to update session {session_id}: {e}", exc_info=True)
            return False

    def end_session(self, session_id: str, final_risk_score: int, final_risk_level: str) -> bool:
        """Mark a session as ended with its final risk snapshot."""
        return self.update_session(
            session_id,
            {
                "end_time": self._now_iso(),
                "risk_score": final_risk_score,
                "risk_level": final_risk_level,
            },
        )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def insert_alert(self, alert: Dict, session_id: Optional[str] = None) -> bool:
        """Persist a single alert. Returns True on success."""
        if not self.is_connected:
            return False
        try:
            doc = dict(alert)
            if session_id:
                doc["session_id"] = session_id
            self.alerts.insert_one(doc)
            if session_id:
                self.sessions.update_one(
                    {"session_id": session_id}, {"$push": {"alerts": alert}}
                )
            return True
        except Exception as e:
            logger.error(f"DatabaseService: failed to insert alert: {e}", exc_info=True)
            return False

    def get_alerts(self, session_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recent alerts, optionally filtered by session. Returns [] if unavailable."""
        if not self.is_connected:
            return []
        try:
            query = {"session_id": session_id} if session_id else {}
            cursor = self.alerts.find(query, {"_id": 0}).sort("timestamp", DESCENDING).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"DatabaseService: failed to get alerts: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Risk events / analysis results
    # ------------------------------------------------------------------

    def insert_risk_event(self, event: Dict, session_id: Optional[str] = None) -> bool:
        """Persist a risk-score snapshot event."""
        if not self.is_connected:
            return False
        try:
            doc = dict(event)
            doc["timestamp"] = doc.get("timestamp", self._now_iso())
            if session_id:
                doc["session_id"] = session_id
            self.risk_events.insert_one(doc)
            return True
        except Exception as e:
            logger.error(f"DatabaseService: failed to insert risk event: {e}", exc_info=True)
            return False

    def get_risk_events(self, session_id: Optional[str] = None, limit: int = 200) -> List[Dict]:
        """Get recent risk-score events, optionally filtered by session."""
        if not self.is_connected:
            return []
        try:
            query = {"session_id": session_id} if session_id else {}
            cursor = (
                self.risk_events.find(query, {"_id": 0}).sort("timestamp", DESCENDING).limit(limit)
            )
            return list(cursor)
        except Exception as e:
            logger.error(f"DatabaseService: failed to get risk events: {e}", exc_info=True)
            return []

    def insert_analysis_result(self, result: Dict, session_id: Optional[str] = None) -> bool:
        """Persist a raw analysis result (road + driver combined output)."""
        if not self.is_connected:
            return False
        try:
            doc = dict(result)
            doc["timestamp"] = doc.get("timestamp", self._now_iso())
            if session_id:
                doc["session_id"] = session_id
            self.analysis_results.insert_one(doc)
            return True
        except Exception as e:
            logger.error(f"DatabaseService: failed to insert analysis result: {e}", exc_info=True)
            return False

    def close(self) -> None:
        """Close the MongoDB connection, if open."""
        if self._client is not None:
            try:
                self._client.close()
                logger.info("DatabaseService: connection closed.")
            except Exception as e:
                logger.error(f"DatabaseService: error closing connection: {e}", exc_info=True)
