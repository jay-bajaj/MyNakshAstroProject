from langchain_core.messages import HumanMessage

from app.brain.neo4j_client import Neo4jClient
from app.config import Settings
from app.graph.workflow import create_workflow


class RecordingProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return f"Reply {len(self.prompts)}"


def test_workflow_keeps_messages_for_the_same_session() -> None:
    provider = RecordingProvider()
    workflow = create_workflow(
        provider=provider,
        neo4j=Neo4jClient(Settings(neo4j_uri=None)),
    )
    config = {"configurable": {"thread_id": "session-1"}}

    def invoke(message: str) -> dict:
        return workflow.invoke(
            {
                "user_id": "user-1",
                "session_id": "session-1",
                "messages": [HumanMessage(content=message)],
                "selected_categories": ["career"],
                "needs_profile": False,
                "retrieved_context": {},
                "context_used": [],
                "response": "",
            },
            config=config,
        )

    first_result = invoke("First question")
    second_result = invoke("Follow-up question")

    assert first_result["response"] == "Reply 1"
    assert second_result["response"] == "Reply 2"
    assert "human: First question" in provider.prompts[1]
    assert "ai: Reply 1" in provider.prompts[1]
    assert "career" in provider.prompts[1]
    assert "conversation_history" in second_result["context_used"]