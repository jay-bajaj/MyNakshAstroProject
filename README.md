# MyNaksh

## Terminal Chat

Start an interactive chat from the repository root:

```powershell
.\.venv\Scripts\python.exe -m app.cli
```

The CLI prompts for a stable user ID (default: `local-user`) and generates a session ID for the run. Every message in that run shares the same session, so follow-up questions work automatically. Use `--user-id` to select long-term memory for a user, `--session-id` to set a session ID, and type `exit` or `quit` to leave the chat.

## Memory Design

### Short-Term and Long-Term Memory

Short-term conversation history is managed by LangGraph's `MemorySaver` checkpointer. The API scopes each checkpoint thread to `user_id:session_id`, keeping users and sessions isolated. Response generation uses only the latest six messages, preserving follow-up context such as "Why do you say that?" without sending the full conversation to the LLM.

Long-term profile data and memories are stored in Neo4j by `user_id`, so they can be retrieved across sessions.

### Memory Filtering and Conflict Resolution

The structured memory extractor uses `should_store` to decide whether extracted facts are durable enough to save. Greetings, chitchat, and transient questions should not become long-term facts. Explicit profile updates are upserted separately, and conversation exchanges are persisted as chat history.

Each fact uses a canonical key, such as `career_goal`. Neo4j `MERGE` matches a memory by its user and key, allowing later corrections to update the existing fact instead of creating a conflicting duplicate.

### Selective Context Retrieval

Before retrieval, `analyze_query_node` classifies a message into relevant life domains, such as `career`, `relationships`, or `finance`. Neo4j filters memories with `WHERE m.category IN $categories`, so unrelated facts are not added to the response context. The API's `context_used` field reports which profile or memory information was included.

### Evaluation Strategy

The following evaluation plan can be used as a basis for a labeled test dataset and response-quality review:

1. **Context relevance:** Measure precision and recall against a golden dataset. For example, a career question should retrieve `career_goal` and exclude `health_goal`.
2. **Memory accuracy:** Verify Neo4j state after multi-turn extraction and user corrections, confirming canonical keys update existing memories.
3. **Response personalization:** Compare responses with `retrieve_brain` enabled and disabled. Use an LLM judge to score personalization and consistency from 1 to 5.