from datetime import date
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph.nodes import (
    analyze_query_node,
    generate_response_node,
    retrieve_brain_node,
    update_memory_node,
)
from app.schemas import ExtractedFact, MemoryExtractionResult, QueryAnalysis


def make_state(message: str = "What should I focus on?") -> dict:
    return {
        "user_id": "user-1",
        "session_id": "session-1",
        "messages": [HumanMessage(content=message)],
        "selected_categories": ["general"],
        "needs_profile": False,
        "retrieved_context": {"date_of_birth": date(1992, 8, 4)},
        "context_used": [],
        "response": "A response",
    }


def test_analyze_query_uses_structured_model_output() -> None:
    llm = Mock()
    analyzer = Mock()
    analyzer.invoke.return_value = QueryAnalysis(
        needs_profile=True,
        relevant_categories=["career", "astrology"],
        is_followup=False,
    )
    llm.with_structured_output.return_value = analyzer

    result = analyze_query_node(make_state("Should I change careers?"), llm)

    assert result == {
        "selected_categories": ["career", "astrology"],
        "needs_profile": True,
    }
    llm.with_structured_output.assert_called_once_with(QueryAnalysis)
    assert analyzer.invoke.call_args.args[0][-1].content == "Should I change careers?"


def test_retrieve_brain_requests_selected_categories_and_adds_context_markers() -> None:
    brain = Mock()
    brain.get_selective_context.return_value = {
        "profile": {"user_id": "user-1", "dob": "1992-08-04"},
        "memories": [
            {"key": "career_goal", "category": "career", "value": "Lead a team"}
        ],
    }
    state = make_state()
    state["selected_categories"] = ["career"]
    state["needs_profile"] = True

    result = retrieve_brain_node(state, brain)

    brain.get_selective_context.assert_called_once_with("user-1", ["career"])
    assert result["retrieved_context"]["sun_sign"] == "Leo"
    assert result["context_used"] == ["user_profile", "career_goal", "sun_sign"]


def test_generate_response_uses_context_and_recent_messages() -> None:
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="Take a measured step.")
    state = make_state()
    state["retrieved_context"] = {
        "profile": {"name": "Ada"},
        "memories": [{"key": "career_goal", "value": "Lead a team"}],
        "sun_sign": "Leo",
    }
    provider = Mock()

    result = generate_response_node(state, provider, llm)

    assert result["response"] == "Take a measured step."
    assert isinstance(llm.invoke.call_args.args[0][0], SystemMessage)
    assert "Leo" in llm.invoke.call_args.args[0][0].content
    provider.generate.assert_not_called()


def test_update_memory_persists_profile_facts_and_exchange() -> None:
    extraction = MemoryExtractionResult(
        should_store=True,
        profile_updates={"name": "Ada", "dob": "1992-08-04"},
        facts=[
            ExtractedFact(
                relation_type="HAS_GOAL",
                category="career",
                key="career_goal",
                value="Lead a team",
                timeframe="2027",
            )
        ],
    )
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = extraction
    brain = Mock()

    update_memory_node(make_state("I want to lead a team by 2027."), brain, llm)

    brain.upsert_profile.assert_called_once_with(
        "user-1", {"name": "Ada", "dob": "1992-08-04", "zodiac": "Leo"}
    )
    brain.upsert_memories.assert_called_once_with(
        "user-1", [extraction.facts[0].model_dump(mode="json")]
    )
    brain.record_exchange.assert_called_once_with(
        "user-1", "I want to lead a team by 2027.", "A response"
    )