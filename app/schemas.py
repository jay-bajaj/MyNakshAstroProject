from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


CategoryType = Literal[
    "career",
    "finance",
    "relationships",
    "health",
    "general",
    "astrology",
    "preferences",
]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    user_id: str = Field(min_length=1, max_length=200)
    session_id: str = Field(min_length=1, max_length=200)
    date_of_birth: date | None = None
    selected_categories: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str
    sun_sign: str | None = None
    context_used: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"


class LLMStructuredOutput(BaseModel):
    content: str
    sun_sign: str | None = None


class QueryAnalysis(BaseModel):
    needs_profile: bool = Field(
        description="True if user asks about astrology, personality, or general life advice."
    )
    relevant_categories: list[CategoryType] = Field(
        description="Select ONLY categories relevant to the current message."
    )
    is_followup: bool = Field(
        description="True if message refers directly to the previous turn."
    )


class ExtractedFact(BaseModel):
    relation_type: Literal[
        "HAS_GOAL", "PREFERS", "INTERESTED_IN", "IMPORTANT_MEMORY"
    ]
    category: CategoryType
    key: str = Field(
        description="Snake_case identifier for deduplication, e.g. 'career_goal'."
    )
    value: str = Field(description="Concise description of the extracted fact.")
    timeframe: str | None = Field(
        default=None,
        description="Optional timeframe, such as '2027' or 'Next Month'.",
    )


class MemoryExtractionResult(BaseModel):
    should_store: bool = Field(
        description="False for greetings, chitchat, or transient questions."
    )
    profile_updates: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Extracted profile updates such as name, date/place/time of birth, "
            "or preferred language."
        ),
    )
    facts: list[ExtractedFact] = Field(default_factory=list)
