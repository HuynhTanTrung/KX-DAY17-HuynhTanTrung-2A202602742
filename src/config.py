from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_COMPACT_THRESHOLD_TOKENS = 4000
DEFAULT_COMPACT_KEEP_MESSAGES = 4

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_GEMINI_JUDGE_MODEL = "gemini-1.5-flash"

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_CUSTOM_BASE_URL = "http://localhost:8000/v1"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_JUDGE_TEMPERATURE = 0.0


@dataclass
class LabConfig:
    """Shared configuration for the memory-systems lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path

    compact_threshold_tokens: int
    compact_keep_messages: int

    model: ProviderConfig
    judge_model: ProviderConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _build_provider_config(
    provider: str,
    model_name: str,
    *,
    role: str,
    temperature: float,
) -> ProviderConfig:
    """Create a ProviderConfig for the given provider name."""

    provider = provider.lower()

    if provider == "gemini":
        api_key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                f"GEMINI_API_KEY (or GOOGLE_API_KEY) is required for the {role} "
                "when LLM_PROVIDER=gemini"
            )
        return ProviderConfig(
            provider="gemini",
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    if provider == "openai":
        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"OPENAI_API_KEY is required for the {role} when LLM_PROVIDER=openai"
            )
        return ProviderConfig(
            provider="openai",
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    if provider == "custom":
        return ProviderConfig(
            provider="custom",
            model_name=model_name,
            temperature=temperature,
            api_key=_env("CUSTOM_API_KEY", "not-needed"),
            base_url=_env("CUSTOM_BASE_URL", DEFAULT_CUSTOM_BASE_URL),
        )

    if provider == "anthropic":
        api_key = _env("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                f"ANTHROPIC_API_KEY is required for the {role} when LLM_PROVIDER=anthropic"
            )
        return ProviderConfig(
            provider="anthropic",
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    if provider == "ollama":
        return ProviderConfig(
            provider="ollama",
            model_name=model_name,
            temperature=temperature,
            api_key=_env("OLLAMA_API_KEY", "not-needed"),
            base_url=_env("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        )

    if provider == "openrouter":
        api_key = _env("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                f"OPENROUTER_API_KEY is required for the {role} when LLM_PROVIDER=openrouter"
            )
        return ProviderConfig(
            provider="openrouter",
            model_name=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

    raise ValueError(
        f"Unsupported provider '{provider}' for the {role}. "
        "Expected one of: gemini, openai, custom, anthropic, ollama, openrouter."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a fully populated LabConfig."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    load_dotenv(dotenv_path=root / ".env", override=False)

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    compact_threshold_tokens = int(
        _env("COMPACT_THRESHOLD_TOKENS", str(DEFAULT_COMPACT_THRESHOLD_TOKENS))
    )
    compact_keep_messages = int(
        _env("COMPACT_KEEP_MESSAGES", str(DEFAULT_COMPACT_KEEP_MESSAGES))
    )

    main_provider = _env("LLM_PROVIDER", "gemini")
    main_model_name = _env("LLM_MODEL", DEFAULT_GEMINI_MODEL)

    judge_provider = _env("JUDGE_PROVIDER", main_provider)
    judge_model_name = _env("JUDGE_MODEL", DEFAULT_GEMINI_JUDGE_MODEL)

    model = _build_provider_config(
        main_provider,
        main_model_name,
        role="main model",
        temperature=DEFAULT_TEMPERATURE,
    )
    judge_model = _build_provider_config(
        judge_provider,
        judge_model_name,
        role="judge model",
        temperature=DEFAULT_JUDGE_TEMPERATURE,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model,
        judge_model=judge_model,
    )