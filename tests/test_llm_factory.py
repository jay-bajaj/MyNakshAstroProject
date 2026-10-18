from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage

from app.config import Settings
from app.llm.factory import LangChainProvider, create_llm_provider, get_llm


@pytest.mark.parametrize(
    ("provider", "model_class", "expected_model", "expected_temperature"),
    [
        ("openai", "ChatOpenAI", "gpt-4o-mini", 0.4),
        ("google", "ChatGoogleGenerativeAI", "gemini-3.5-flash-lite", None),
    ],
)
def test_get_llm_configures_supported_provider(
    provider: str,
    model_class: str,
    expected_model: str,
    expected_temperature: float | None,
) -> None:
    settings = Settings(llm_api_key="test-key")
    with patch(f"app.llm.factory.{model_class}") as model_factory:
        get_llm(provider, temperature=0.4, settings=settings)

    options = model_factory.call_args.kwargs
    assert options["model"] == expected_model
    assert options.get("temperature") == expected_temperature


def test_get_llm_rejects_unsupported_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider: other"):
        get_llm("other")


def test_langchain_provider_returns_message_text() -> None:
    model = Mock()
    model.invoke.return_value = AIMessage(content="A clear answer")

    provider = LangChainProvider(model)

    assert provider.generate("Question") == "A clear answer"
    model.invoke.assert_called_once_with("Question")


def test_create_llm_provider_keeps_stub_default() -> None:
    provider = create_llm_provider(Settings(llm_provider="stub"))

    assert provider.generate("Hello") == "Received your message: Hello"