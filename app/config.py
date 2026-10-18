from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MyNaksh API"
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"
    llm_provider: str = "stub"
    llm_api_key: str | None = None
    llm_temperature: float = 0.2


@lru_cache
def get_settings() -> Settings:
    return Settings()
