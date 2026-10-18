import logging
from typing import Any

from neo4j import GraphDatabase

from app.config import Settings, get_settings


logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._driver: Any | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.neo4j_uri)

    def initialize_schema(self) -> None:
        if not self.enabled:
            return
        self._get_driver().execute_query(
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS "
            "FOR (user:User) REQUIRE user.user_id IS UNIQUE",
            database_=self.settings.neo4j_database,
        )
        self._get_driver().execute_query(
            "CREATE CONSTRAINT memory_user_key_unique IF NOT EXISTS "
            "FOR (memory:MemoryItem) "
            "REQUIRE (memory.user_id, memory.key) IS UNIQUE",
            database_=self.settings.neo4j_database,
        )

    def get_recent_messages(self, user_id: str, limit: int = 5) -> list[str]:
        rows = self.run_query(
            "MATCH (user:User {user_id: $user_id})-[:SENT]->(message:Message) "
            "RETURN message.content AS content "
            "ORDER BY message.created_at DESC LIMIT $limit",
            {"user_id": user_id, "limit": limit},
        )
        return [str(row["content"]) for row in rows]

    def upsert_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        query = (
            "MERGE (user:User {user_id: $user_id}) "
            "SET user += $profile "
            "RETURN user"
        )
        try:
            self.run_query(query, {"user_id": user_id, "profile": profile})
        except Exception:
            logger.exception("Neo4j profile update failed")

    def get_selective_context(
        self, user_id: str, categories: list[str]
    ) -> dict[str, Any]:
        query = (
            "MATCH (user:User {user_id: $user_id}) "
            "OPTIONAL MATCH (user)-[relation:HAS_MEMORY]->(memory:MemoryItem) "
            "WHERE memory.category IN $categories "
            "RETURN properties(user) AS profile, "
            "collect({rel: relation.type, key: memory.key, value: memory.value, "
            "category: memory.category, timeframe: memory.timeframe}) AS memories"
        )
        try:
            rows = self.run_query(
                query, {"user_id": user_id, "categories": categories}
            )
            if not rows:
                return {"profile": {}, "memories": []}
            row = rows[0]
            memories = [
                memory
                for memory in row.get("memories", [])
                if memory.get("key") is not None
            ]
            return {"profile": row.get("profile") or {}, "memories": memories}
        except Exception:
            logger.exception("Neo4j context retrieval failed")
            return {"profile": {}, "memories": []}

    def upsert_memories(
        self, user_id: str, facts: list[dict[str, Any]]
    ) -> None:
        if not facts:
            return
        query = (
            "MERGE (user:User {user_id: $user_id}) "
            "WITH user "
            "UNWIND $facts AS fact "
            "MERGE (user)-[relation:HAS_MEMORY]->(memory:MemoryItem "
            "{user_id: $user_id, key: fact.key}) "
            "SET relation.type = fact.relation_type, "
            "memory.value = fact.value, "
            "memory.category = fact.category, "
            "memory.timeframe = fact.timeframe, "
            "memory.updated_at = timestamp()"
        )
        try:
            self.run_query(query, {"user_id": user_id, "facts": facts})
        except Exception:
            logger.exception("Neo4j memory update failed")

    def record_exchange(self, user_id: str, message: str, reply: str) -> None:
        self.run_query(
            "MERGE (user:User {user_id: $user_id}) "
            "CREATE (user_message:Message {content: $message, created_at: datetime()}) "
            "CREATE (user)-[:SENT]->(user_message) "
            "CREATE (assistant_message:Message {content: $reply, created_at: datetime()}) "
            "CREATE (user)-[:RECEIVED]->(assistant_message)",
            {"user_id": user_id, "message": message, "reply": reply},
        )

    def run_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        records, _, _ = self._get_driver().execute_query(
            query,
            parameters or {},
            database_=self.settings.neo4j_database,
        )
        return [record.data() for record in records]

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _get_driver(self) -> Any:
        if self._driver is None:
            auth = None
            if self.settings.neo4j_username and self.settings.neo4j_password:
                auth = (self.settings.neo4j_username, self.settings.neo4j_password)
            self._driver = GraphDatabase.driver(self.settings.neo4j_uri, auth=auth)
        return self._driver
