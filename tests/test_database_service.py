"""
Tests for the MongoDB database service: graceful fallback when
unavailable, and correct CRUD behavior when connected (verified against
mongomock, an in-memory MongoDB-compatible test double).
"""

import time

from backend.services.database_service import DatabaseService


class TestDatabaseFallback:
    """
    MongoDB unavailable must degrade gracefully, never raise or hang.

    Uses 192.0.2.1 (RFC 5737 TEST-NET-1 — reserved for documentation/
    testing, guaranteed never to have a real service listening) instead
    of "localhost:27017", since a real MongoDB may legitimately be
    running on localhost in some environments (e.g. a developer who set
    it up per the README's optional MongoDB Setup section), which would
    make these "unavailable" tests fail for the wrong reason.
    """

    UNREACHABLE_URI = "mongodb://192.0.2.1:27017"

    def test_unreachable_database_fails_fast(self):
        start = time.time()
        db = DatabaseService(mongodb_uri=self.UNREACHABLE_URI, server_selection_timeout_ms=1000)
        elapsed = time.time() - start
        assert db.is_connected is False
        assert elapsed < 3.0

    def test_all_methods_safe_when_disconnected(self):
        db = DatabaseService(mongodb_uri=self.UNREACHABLE_URI, server_selection_timeout_ms=500)
        assert db.create_session() is None
        assert db.get_session("fake-id") is None
        assert db.list_sessions() == []
        assert db.update_session("fake-id", {}) is False
        assert db.insert_alert({"type": "TEST"}) is False
        assert db.get_alerts() == []
        assert db.insert_risk_event({"risk_score": 50}) is False
        assert db.insert_analysis_result({"foo": "bar"}) is False


class TestDatabaseCRUD:
    """Real CRUD logic, verified against mongomock (behaves like real MongoDB)."""

    def test_create_and_get_session(self, mongomock_client):
        db = DatabaseService(database_name="test_db", client=mongomock_client)
        sid = db.create_session({"driver_status": "NORMAL"})
        assert sid is not None

        session = db.get_session(sid)
        assert session["session_id"] == sid
        assert session["driver_status"] == "NORMAL"
        assert session["risk_level"] == "LOW"  # default

    def test_update_session(self, mongomock_client):
        db = DatabaseService(database_name="test_db2", client=mongomock_client)
        sid = db.create_session()
        assert db.update_session(sid, {"risk_score": 75, "risk_level": "HIGH"}) is True
        session = db.get_session(sid)
        assert session["risk_score"] == 75

    def test_list_sessions_returns_all(self, mongomock_client):
        db = DatabaseService(database_name="test_db3", client=mongomock_client)
        db.create_session()
        db.create_session()
        sessions = db.list_sessions()
        assert len(sessions) == 2

    def test_insert_alert_links_to_session(self, mongomock_client):
        db = DatabaseService(database_name="test_db4", client=mongomock_client)
        sid = db.create_session()
        ok = db.insert_alert(
            {"type": "DROWSINESS", "severity": "HIGH", "message": "test", "timestamp": "now"},
            session_id=sid,
        )
        assert ok is True
        alerts = db.get_alerts(session_id=sid)
        assert len(alerts) == 1
        session = db.get_session(sid)
        assert len(session["alerts"]) == 1

    def test_risk_events(self, mongomock_client):
        db = DatabaseService(database_name="test_db5", client=mongomock_client)
        sid = db.create_session()
        assert db.insert_risk_event({"risk_score": 55, "risk_level": "MEDIUM"}, session_id=sid) is True
        events = db.get_risk_events(session_id=sid)
        assert len(events) == 1

    def test_end_session(self, mongomock_client):
        db = DatabaseService(database_name="test_db6", client=mongomock_client)
        sid = db.create_session()
        assert db.end_session(sid, 80, "HIGH") is True
        session = db.get_session(sid)
        assert session["end_time"] is not None
        assert session["risk_score"] == 80

    def test_get_nonexistent_session_returns_none(self, mongomock_client):
        db = DatabaseService(database_name="test_db7", client=mongomock_client)
        assert db.get_session("does-not-exist") is None
