"""Deterministic end-to-end chat scenarios without external services."""

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

import app.main as main
from app.graph.workflow import build_chat_graph
from app.llm.factory import StubLLMProvider
from app.schemas import ExtractedFact, MemoryExtractionResult, QueryAnalysis


class InMemoryNeo4j:
    """Small in-memory implementation of the Neo4j methods used by the graph."""

    def __init__(self) -> None:
        self.profiles: dict[str, dict] = {}
        self.memories: dict[tuple[str, str], dict] = {}
        self.exchanges: list[tuple[str, str, str]] = []
        self.disconnected = False

    def initialize_schema(self) -> None:
        pass

    def close(self) -> None:
        pass

    def get_selective_context(self, user_id: str, categories: list[str]) -> dict:
        if self.disconnected:
            raise ConnectionError("Neo4j is unavailable")
        return {
            "profile": dict(self.profiles.get(user_id, {})),
            "memories": [
                dict(memory)
                for (stored_user_id, _), memory in self.memories.items()
                if stored_user_id == user_id and memory["category"] in categories
            ],
        }

    def upsert_profile(self, user_id: str, profile: dict) -> None:
        if self.disconnected:
            raise ConnectionError("Neo4j is unavailable")
        self.profiles.setdefault(user_id, {}).update(profile)

    def upsert_memories(self, user_id: str, facts: list[dict]) -> None:
        if self.disconnected:
            raise ConnectionError("Neo4j is unavailable")
        for fact in facts:
            self.memories[(user_id, fact["key"])] = dict(fact)

    def record_exchange(self, user_id: str, message: str, reply: str) -> None:
        if self.disconnected:
            raise ConnectionError("Neo4j is unavailable")
        self.exchanges.append((user_id, message, reply))


class DeterministicModel:
    def __init__(self) -> None:
        self.generation_inputs: list[list] = []

    def with_structured_output(self, schema):
        return DeterministicStructuredOutput(schema)

    def invoke(self, messages):
        self.generation_inputs.append(messages)
        latest_human = next(
            message for message in reversed(messages) if isinstance(message, HumanMessage)
        )
        text = latest_human.content.casefold()

        if "why do you say that" in text:
            response = (
                "I suggested updating your resume because it supports your plan to "
                "switch jobs in 2027."
            )
        elif "what should i focus" in text and "career" in text:
            response = (
                "Focus on updating your resume and targeting roles for your 2027 job switch."
            )
        elif "what do you remember" in text and "career" in text:
            response = "I remember your career goal: switch jobs in 2027."
        elif "birth chart" in text or "birth details" in text:
            response = "Please share your birth date, exact birth time, and birthplace."
        elif "my name is rahul" in text:
            response = "Thanks, Rahul. I have your profile and career goal."
        else:
            response = "For your career, choose one practical next step and act on it."
        return AIMessage(content=response)


class DeterministicStructuredOutput:
    def __init__(self, schema) -> None:
        self.schema = schema

    def invoke(self, messages):
        human_message = next(
            message for message in reversed(messages) if isinstance(message, HumanMessage)
        )
        text = human_message.content
        normalized = text.casefold()

        if self.schema is QueryAnalysis:
            categories = []
            if any(term in normalized for term in ("career", "job", "goal")):
                categories.append("career")
            if "health" in normalized:
                categories.append("health")
            if any(term in normalized for term in ("birth chart", "astrology", "born")):
                categories.append("astrology")
            if not categories:
                categories.append("general")
            return QueryAnalysis(
                needs_profile=any(
                    term in normalized for term in ("birth chart", "astrology", "born")
                ),
                relevant_categories=categories,
                is_followup="why do you say that" in normalized,
            )

        profile_updates = None
        if "my name is rahul" in normalized:
            profile_updates = {
                "name": "Rahul",
                "dob": "1995-08-15",
                "pob": "Delhi",
            }

        facts = []
        if "switch jobs" in normalized:
            timeframe = "2026" if "2026" in normalized else "2027"
            facts.append(
                ExtractedFact(
                    relation_type="HAS_GOAL",
                    category="career",
                    key="career_goal",
                    value="Switch jobs",
                    timeframe=timeframe,
                )
            )
        return MemoryExtractionResult(
            should_store=bool(profile_updates or facts),
            profile_updates=profile_updates,
            facts=facts,
        )


class DisconnectedNeo4j(InMemoryNeo4j):
    def __init__(self) -> None:
        super().__init__()
        self.disconnected = True


def make_test_client(monkeypatch, brain: InMemoryNeo4j) -> TestClient:
    model = DeterministicModel()
    monkeypatch.setattr(main, "brain", brain)
    monkeypatch.setattr(
        main,
        "chat_graph",
        build_chat_graph(
            brain,
            provider=StubLLMProvider(),
            llm=model,
        ),
    )
    client = TestClient(main.app)
    client.deterministic_model = model
    return client


def post_chat(client: TestClient, message: str, user_id="rahul", session_id="session-1"):
    return client.post(
        "/chat",
        json={"user_id": user_id, "session_id": session_id, "message": message},
    )


def test_scenario_1_new_user_profile_and_memory_creation(monkeypatch) -> None:
    brain = InMemoryNeo4j()
    with make_test_client(monkeypatch, brain) as client:
        response = post_chat(
            client,
            "My name is Rahul, born 15 Aug 1995 in Delhi. "
            "I want to switch jobs in 2027.",
        )

    assert response.status_code == 200
    assert brain.profiles["rahul"]["zodiac"] == "Leo"
    assert brain.memories[("rahul", "career_goal")]["timeframe"] == "2027"


def test_scenario_2_retrieves_relevant_memory(monkeypatch) -> None:
    brain = InMemoryNeo4j()
    brain.upsert_memories(
        "rahul",
        [
            ExtractedFact(
                relation_type="HAS_GOAL",
                category="career",
                key="career_goal",
                value="Switch jobs",
                timeframe="2027",
            ).model_dump(mode="json")
        ],
    )
    with make_test_client(monkeypatch, brain) as client:
        response = post_chat(client, "What should I focus on for my career?")

    assert response.status_code == 200
    assert "career_goal" in response.json()["context_used"]


def test_scenario_3_followup_uses_short_term_memory(monkeypatch) -> None:
    brain = InMemoryNeo4j()
    with make_test_client(monkeypatch, brain) as client:
        first = post_chat(client, "What should I focus on for my career?")
        followup = post_chat(client, "Why do you say that?")
        followup_input = client.deterministic_model.generation_inputs[-1]

    assert first.status_code == followup.status_code == 200
    assert "updating your resume" in followup.json()["response"].casefold()
    assert "conversation_history" in followup.json()["context_used"]
    assert any(
        isinstance(message, AIMessage)
        and "updating your resume" in message.content.casefold()
        for message in followup_input
    )


def test_scenario_4_long_term_memory_survives_new_session(monkeypatch) -> None:
    brain = InMemoryNeo4j()
    with make_test_client(monkeypatch, brain) as client:
        post_chat(
            client,
            "My name is Rahul, born 15 Aug 1995 in Delhi. "
            "I want to switch jobs in 2027.",
        )
        response = post_chat(
            client,
            "What do you remember about my career goals?",
            session_id="session-999",
        )
        new_session_input = client.deterministic_model.generation_inputs[-1]

    assert response.status_code == 200
    assert "career_goal" in response.json()["context_used"]
    assert "switch jobs in 2027" in response.json()["response"].casefold()
    assert "Switch jobs" in new_session_input[0].content
    assert "2027" in new_session_input[0].content


def test_scenario_5_irrelevant_memory_is_filtered(monkeypatch) -> None:
    brain = InMemoryNeo4j()
    brain.upsert_memories(
        "rahul",
        [
            {
                "relation_type": "HAS_GOAL",
                "category": "career",
                "key": "career_goal",
                "value": "Switch jobs",
                "timeframe": "2027",
            },
            {
                "relation_type": "HAS_GOAL",
                "category": "health",
                "key": "health_goal",
                "value": "Run a marathon",
                "timeframe": None,
            },
        ],
    )
    with make_test_client(monkeypatch, brain) as client:
        response = post_chat(client, "What should I focus on for my career?")

    assert response.status_code == 200
    assert "career_goal" in response.json()["context_used"]
    assert "health_goal" not in response.json()["context_used"]


def test_scenario_6_correction_updates_existing_memory(monkeypatch) -> None:
    brain = InMemoryNeo4j()
    with make_test_client(monkeypatch, brain) as client:
        post_chat(
            client,
            "My name is Rahul, born 15 Aug 1995 in Delhi. "
            "I want to switch jobs in 2027.",
        )
        response = post_chat(
            client,
            "Actually, I decided to switch jobs in 2026 instead of 2027.",
        )

    assert response.status_code == 200
    assert brain.memories[("rahul", "career_goal")]["timeframe"] == "2026"
    assert len([key for key in brain.memories if key == ("rahul", "career_goal")]) == 1


def test_scenario_7_missing_birth_details_are_requested(monkeypatch) -> None:
    brain = InMemoryNeo4j()
    with make_test_client(monkeypatch, brain) as client:
        response = post_chat(
            client,
            "What does my birth chart say about today?",
            user_id="new-user",
        )

    assert response.status_code == 200
    assert "birth date" in response.json()["response"].casefold()


def test_scenario_8_neo4j_disconnect_falls_back_gracefully(monkeypatch) -> None:
    brain = DisconnectedNeo4j()
    with make_test_client(monkeypatch, brain) as client:
        response = post_chat(client, "What should I focus on for my career?")

    assert response.status_code == 200
    assert response.json()["response"]
    assert response.json()["context_used"] == []
