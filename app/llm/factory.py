from collections.abc import Callable
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt."""


class StubLLMProvider:
    def generate(self, prompt: str) -> str:
        return f"Received your message: {prompt}"


class LangChainProvider:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        content = response.content
        if isinstance(content, str):
            return content
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, (str, dict))
        )


def get_llm(
    provider: str = "openai",
    temperature: float = 0.2,
    settings: Settings | None = None,
) -> BaseChatModel:
    active_settings = settings or get_settings()
    provider_name = provider.lower()
    if provider_name == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=active_settings.llm_api_key,
        )
    if provider_name == "google":
        return ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=active_settings.llm_api_key,
        )
    raise ValueError(f"Unsupported provider: {provider}")


ProviderFactory = Callable[[Settings], LLMProvider]
_PROVIDERS: dict[str, ProviderFactory] = {
    "stub": lambda settings: StubLLMProvider(),
    "openai": lambda settings: LangChainProvider(
        get_llm("openai", settings.llm_temperature, settings)
    ),
    "google": lambda settings: LangChainProvider(
        get_llm("google", settings.llm_temperature, settings)
    ),
}


def register_provider(name: str, factory: ProviderFactory) -> None:
    _PROVIDERS[name.lower()] = factory


def create_llm_provider(settings: Settings | None = None) -> LLMProvider:
    active_settings = settings or get_settings()
    provider_name = active_settings.llm_provider.lower()
    try:
        factory = _PROVIDERS[provider_name]
    except KeyError as error:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. Registered providers: {available}"
        ) from error
    return factory(active_settings)
