import logging
from datetime import date, datetime
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.brain.neo4j_client import Neo4jClient
from app.graph.state import ChatState
from app.llm.factory import LLMProvider
from app.schemas import MemoryExtractionResult, QueryAnalysis
from app.utils.astrology import get_sun_sign


logger = logging.getLogger(__name__)
_VALID_CATEGORIES = {
    "career",
    "finance",
    "relationships",
    "health",
    "general",
    "astrology",
    "preferences",
}


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, (str, dict))
    )


def _latest_human_message(messages: list[BaseMessage]) -> HumanMessage | None:
    return next(
        (message for message in reversed(messages) if isinstance(message, HumanMessage)),
        None,
    )


def _fallback_analysis(state: ChatState) -> dict[str, Any]:
    categories = [
        category
        for category in state.get("selected_categories", [])
        if category in _VALID_CATEGORIES
    ] or ["general"]
    return {
        "selected_categories": categories,
        "needs_profile": state.get("needs_profile", False),
    }


def analyze_query_node(
    state: ChatState, llm: BaseChatModel | None
) -> dict[str, Any]:
    """Choose relevant memory categories and whether profile context is needed."""
    human_message = _latest_human_message(state["messages"])
    if human_message is None or llm is None:
        return _fallback_analysis(state)

    analyzer = llm.with_structured_output(QueryAnalysis)
    try:
        analysis = analyzer.invoke(
            [
                SystemMessage(
                    content=(
                        "Analyze the user's current query. Select only the memory "
                        "categories relevant to answering it."
                    )
                ),
                HumanMessage(content=_message_text(human_message)),
            ]
        )
    except Exception:
        logger.exception("Query analysis failed; using request context")
        return _fallback_analysis(state)

    return {
        "selected_categories": analysis.relevant_categories,
        "needs_profile": analysis.needs_profile,
    }


def retrieve_brain_node(state: ChatState, brain: Neo4jClient) -> dict[str, Any]:
    """Retrieve profile and memories filtered to the analyzed categories."""
    try:
        raw_context = brain.get_selective_context(
            state["user_id"], state["selected_categories"]
        )
    except Exception:
        logger.exception("Neo4j context retrieval failed; continuing without memory")
        raw_context = {"profile": {}, "memories": []}
    profile = raw_context.get("profile", {})
    memories = raw_context.get("memories", [])
    context_used: list[str] = []

    if state["needs_profile"] and profile:
        context_used.append("user_profile")
    for memory in memories:
        key = memory.get("key")
        if key and key not in context_used:
            context_used.append(key)
    if len(state["messages"]) > 1:
        context_used.append("conversation_history")

    retrieved_context = dict(raw_context)
    date_of_birth = profile.get("dob") or state.get("retrieved_context", {}).get(
        "date_of_birth"
    )
    if date_of_birth:
        try:
            if isinstance(date_of_birth, datetime):
                birth_date = date_of_birth.date()
            elif isinstance(date_of_birth, date):
                birth_date = date_of_birth
            else:
                birth_date = date.fromisoformat(str(date_of_birth)[:10])
            retrieved_context["sun_sign"] = get_sun_sign(birth_date)
        except ValueError:
            logger.warning("Could not calculate sun sign from stored date of birth")

    if retrieved_context.get("sun_sign"):
        context_used.append("sun_sign")
    return {"retrieved_context": retrieved_context, "context_used": context_used}


def generate_response_node(
    state: ChatState, provider: LLMProvider, llm: BaseChatModel | None
) -> dict[str, Any]:
    """Generate a response from recent chat and selectively retrieved context."""
    recent_messages = state["messages"][-6:]
    context = state.get("retrieved_context", {})
    profile = context.get("profile", {}) if state.get("needs_profile") else {}
    memories = context.get("memories", [])
    sun_sign = context.get("sun_sign")
    system_prompt = (
        "You are MyNaksh, a personalized astrology AI assistant. "
        "Use the provided profile and long-term memories only when relevant. "
        "If astrology or birth details are needed but missing, politely ask for them.\n\n"
        f"User profile: {profile or 'Not needed for this query'}\n"
        f"Sun sign: {sun_sign or 'Not available'}\n"
        f"Selected categories: {', '.join(state['selected_categories'])}\n"
        f"Relevant long-term memories: {memories}"
    )

    try:
        if llm is not None:
            ai_message = llm.invoke(
                [SystemMessage(content=system_prompt), *recent_messages]
            )
            response = _message_text(ai_message)
        else:
            prompt = system_prompt + "\n\n" + "\n".join(
                f"{message.type}: {_message_text(message)}"
                for message in recent_messages
            )
            response = provider.generate(prompt)
            ai_message = AIMessage(content=response)
    except Exception:
        logger.exception("LLM response generation failed")
        response = (
            "I'm experiencing a brief cosmic disruption connecting to my "
            "reasoning engine. Could you try asking again?"
        )
        ai_message = AIMessage(content=response)

    return {"messages": [ai_message], "response": response}


def _normalize_profile_updates(profile_updates: dict[str, Any]) -> dict[str, Any]:
    updates = {key: value for key, value in profile_updates.items() if value}
    birth_date = updates.get("dob")
    if birth_date:
        try:
            if isinstance(birth_date, datetime):
                parsed_birth_date = birth_date.date()
            elif isinstance(birth_date, date):
                parsed_birth_date = birth_date
            else:
                parsed_birth_date = date.fromisoformat(str(birth_date)[:10])
            updates["zodiac"] = get_sun_sign(parsed_birth_date)
        except ValueError:
            logger.warning("Could not calculate zodiac for extracted date of birth")
    return updates


def update_memory_node(
    state: ChatState, brain: Neo4jClient, llm: BaseChatModel | None
) -> dict[str, Any]:
    """Extract durable profile facts, store them, and persist this exchange."""
    human_message = _latest_human_message(state["messages"])
    if human_message is None:
        return {}

    message_text = _message_text(human_message)
    if llm is not None:
        extractor = llm.with_structured_output(MemoryExtractionResult)
        try:
            extraction = extractor.invoke(
                [
                    SystemMessage(
                        content=(
                            "Extract durable goals, preferences, interests, life events, "
                            "or profile attributes such as name, date/place/time of birth, "
                            "and preferred language. Ignore chitchat and transient states. "
                            "Use consistent snake_case keys so updated facts overwrite old ones."
                        )
                    ),
                    HumanMessage(content=message_text),
                ]
            )
            if extraction.profile_updates:
                profile_updates = _normalize_profile_updates(
                    extraction.profile_updates
                )
                if profile_updates:
                    brain.upsert_profile(state["user_id"], profile_updates)
            if extraction.should_store and extraction.facts:
                brain.upsert_memories(
                    state["user_id"],
                    [fact.model_dump(mode="json") for fact in extraction.facts],
                )
        except Exception:
            logger.exception("Memory extraction failed")

    try:
        brain.record_exchange(state["user_id"], message_text, state["response"])
    except Exception:
        logger.exception("Could not persist chat exchange")
    return {}
