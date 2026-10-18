import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage

from app.brain.neo4j_client import Neo4jClient
from app.config import get_settings
from app.graph.workflow import build_chat_graph
from app.schemas import ChatRequest, ChatResponse, HealthResponse


logger = logging.getLogger(__name__)
settings = get_settings()
brain = Neo4jClient(settings)
chat_graph = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global chat_graph
    try:
        brain.initialize_schema()
    except Exception:
        logger.exception("Neo4j schema initialization failed; continuing without it")
    if chat_graph is None:
        chat_graph = build_chat_graph(brain)
    try:
        yield
    finally:
        brain.close()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    thread_id = f"{request.user_id}:{request.session_id}"
    if chat_graph is None:
        raise HTTPException(status_code=503, detail="Chat workflow is not initialized")
    try:
        result = chat_graph.invoke(
            {
                "user_id": request.user_id,
                "session_id": request.session_id,
                "messages": [HumanMessage(content=request.message)],
                "selected_categories": request.selected_categories,
                "needs_profile": request.date_of_birth is None,
                "retrieved_context": {"date_of_birth": request.date_of_birth},
                "context_used": [],
                "response": "",
            },
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as error:
        logger.exception("Chat workflow failed")
        raise HTTPException(status_code=500, detail="Chat workflow failed") from error

    return ChatResponse(
        response=result["response"],
        user_id=request.user_id,
        session_id=request.session_id,
        sun_sign=result.get("retrieved_context", {}).get("sun_sign"),
        context_used=result.get("context_used", []),
    )
