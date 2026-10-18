from functools import partial

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.brain.neo4j_client import Neo4jClient
from app.graph.nodes import (
    analyze_query_node,
    generate_response_node,
    retrieve_brain_node,
    update_memory_node,
)
from app.graph.state import ChatState
from app.llm.factory import LLMProvider, create_llm_provider
from langchain_core.language_models.chat_models import BaseChatModel


def build_chat_graph(
    brain: Neo4jClient,
    provider: LLMProvider | None = None,
    llm: BaseChatModel | None = None,
):
    active_provider = provider or create_llm_provider()
    active_llm = llm if llm is not None else getattr(active_provider, "model", None)

    graph = StateGraph(ChatState)
    graph.add_node(
        "analyze_query", partial(analyze_query_node, llm=active_llm)
    )
    graph.add_node("retrieve_brain", partial(retrieve_brain_node, brain=brain))
    graph.add_node(
        "generate_response",
        partial(
            generate_response_node,
            provider=active_provider,
            llm=active_llm,
        ),
    )
    graph.add_node(
        "update_memory",
        partial(update_memory_node, brain=brain, llm=active_llm),
    )
    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "retrieve_brain")
    graph.add_edge("retrieve_brain", "generate_response")
    graph.add_edge("generate_response", "update_memory")
    graph.add_edge("update_memory", END)
    return graph.compile(checkpointer=MemorySaver())


def create_workflow(
    provider: LLMProvider | None = None,
    neo4j: Neo4jClient | None = None,
    llm: BaseChatModel | None = None,
):
    """Create the chat graph using the application's current workflow API."""
    return build_chat_graph(
        brain=neo4j or Neo4jClient(),
        provider=provider,
        llm=llm,
    )
