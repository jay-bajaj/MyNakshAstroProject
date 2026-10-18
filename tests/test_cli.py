from app.cli import run_chat


class RecordingGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict]] = []

    def invoke(self, state: dict, config: dict) -> dict:
        self.calls.append((state, config))
        return {"response": f"Reply {len(self.calls)}"}


def test_cli_reuses_user_and_session_for_followups() -> None:
    graph = RecordingGraph()
    messages = iter(["Hello", "Why do you say that?", "exit"])
    output: list[str] = []

    run_chat(
        graph,
        user_id="rahul",
        session_id="session-123",
        input_fn=lambda _: next(messages),
        output_fn=output.append,
    )

    assert len(graph.calls) == 2
    assert all(call[0]["user_id"] == "rahul" for call in graph.calls)
    assert all(call[0]["session_id"] == "session-123" for call in graph.calls)
    assert all(
        call[1]["configurable"]["thread_id"] == "rahul:session-123"
        for call in graph.calls
    )
    assert output[1:3] == ["", "MyNaksh: Reply 1"]
    assert output[3:5] == ["", "MyNaksh: Reply 2"]
