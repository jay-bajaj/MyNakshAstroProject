from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from uuid import uuid4

from app.brain.neo4j_client import Neo4jClient
from app.config import get_settings
from app.graph.workflow import build_chat_graph
from langchain_core.messages import HumanMessage


logger = logging.getLogger(__name__)


def run_chat(
    graph,
    user_id: str,
    session_id: str,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    thread_id = f"{user_id}:{session_id}"
    output_fn(f"MyNaksh chat started (session {session_id}). Type 'exit' to quit.")

    while True:
        try:
            message = input_fn("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nChat ended.")
            return

        if message.casefold() in {"exit", "quit"}:
            output_fn("Chat ended.")
            return
        if not message:
            continue

        try:
            result = graph.invoke(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "messages": [HumanMessage(content=message)],
                    "selected_categories": [],
                    "needs_profile": False,
                    "retrieved_context": {},
                    "context_used": [],
                    "response": "",
                },
                config={"configurable": {"thread_id": thread_id}},
            )
            response = result["response"]
        except Exception:
            logger.exception("CLI chat request failed")
            response = "I couldn't complete that request. Please try again."
        output_fn("")
        output_fn(f"MyNaksh: {response}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with MyNaksh in your terminal")
    parser.add_argument("--user-id", help="Stable user ID for long-term memory")
    parser.add_argument("--session-id", help="Set the short-term session ID")
    args = parser.parse_args()

    user_id = args.user_id or input("User ID for long-term memory [local-user]: ").strip()
    user_id = user_id or "local-user"
    session_id = args.session_id or str(uuid4())

    settings = get_settings()
    brain = Neo4jClient(settings)
    try:
        try:
            brain.initialize_schema()
        except Exception:
            logger.exception("Neo4j unavailable; chat will continue without stored memory")
        graph = build_chat_graph(brain)
        run_chat(graph, user_id, session_id)
    finally:
        brain.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    main()
