from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    user_id: str
    session_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    selected_categories: list[str]
    needs_profile: bool
    retrieved_context: dict[str, Any]
    context_used: list[str]
    response: str
    prompt: NotRequired[str]
