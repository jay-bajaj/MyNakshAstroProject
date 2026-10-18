from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import app


def test_chat_uses_user_and_session_for_checkpoint_thread(monkeypatch) -> None:
    graph = Mock()
    graph.invoke.return_value = {
        "response": "A tailored response",
        "retrieved_context": {"sun_sign": "Leo"},
        "context_used": ["user_profile", "career_goal"],
    }
    monkeypatch.setattr("app.main.chat_graph", graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={
                "user_id": "user-1",
                "session_id": "session-2",
                "message": "What should I focus on?",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "response": "A tailored response",
        "user_id": "user-1",
        "session_id": "session-2",
        "sun_sign": "Leo",
        "context_used": ["user_profile", "career_goal"],
    }
    assert graph.invoke.call_args.kwargs["config"]["configurable"]["thread_id"] == (
        "user-1:session-2"
    )


def test_chat_rejects_empty_session_id(monkeypatch) -> None:
    monkeypatch.setattr("app.main.chat_graph", Mock())
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"user_id": "user-1", "session_id": "", "message": "Hello"},
        )

    assert response.status_code == 422