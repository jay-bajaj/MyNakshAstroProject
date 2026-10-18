from unittest.mock import Mock

from app.brain.neo4j_client import Neo4jClient
from app.config import Settings


def test_initialize_schema_creates_user_and_memory_constraints() -> None:
    client = Neo4jClient(Settings(neo4j_uri="bolt://localhost:7687"))
    client._driver = Mock()

    client.initialize_schema()

    queries = [call.args[0] for call in client._driver.execute_query.call_args_list]
    assert any(
        "user_id_unique" in query and "user.user_id IS UNIQUE" in query
        for query in queries
    )
    assert any(
        "memory_user_key_unique" in query
        and "(memory.user_id, memory.key) IS UNIQUE" in query
        for query in queries
    )


def test_selective_context_filters_null_memories_and_passes_categories() -> None:
    client = Neo4jClient(Settings(neo4j_uri="bolt://localhost:7687"))
    record = Mock()
    record.data.return_value = {
        "profile": {"user_id": "user-1", "name": "Ada"},
        "memories": [
            {"key": "career_goal", "category": "career", "value": "Switch jobs"},
            {"key": None, "category": None, "value": None},
        ],
    }
    client._driver = Mock()
    client._driver.execute_query.return_value = ([record], None, [])

    context = client.get_selective_context("user-1", ["career"])

    assert context == {
        "profile": {"user_id": "user-1", "name": "Ada"},
        "memories": [
            {"key": "career_goal", "category": "career", "value": "Switch jobs"}
        ],
    }
    query_parameters = client._driver.execute_query.call_args.args[1]
    assert query_parameters["categories"] == ["career"]


def test_selective_context_falls_back_when_neo4j_fails() -> None:
    client = Neo4jClient(Settings(neo4j_uri="bolt://localhost:7687"))
    client._driver = Mock()
    client._driver.execute_query.side_effect = RuntimeError("connection failed")

    assert client.get_selective_context("user-1", ["career"]) == {
        "profile": {},
        "memories": [],
    }


def test_context_and_upserts_are_noops_when_neo4j_is_disabled() -> None:
    client = Neo4jClient(Settings(neo4j_uri=None))

    assert client.get_selective_context("user-1", ["career"]) == {
        "profile": {},
        "memories": [],
    }
    client.upsert_profile("user-1", {"name": "Ada"})
    client.upsert_memories(
        "user-1",
        [
            {
                "key": "career_goal",
                "relation_type": "HAS_GOAL",
                "value": "Switch jobs",
                "category": "career",
                "timeframe": None,
            }
        ],
    )
    assert client._driver is None