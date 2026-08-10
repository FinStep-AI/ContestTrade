"""
Provider authentication resolution for multi-LLM support.

Inspired by openclaw's layered auth resolution: config -> provider env vars -> generic fallback.
"""
import os
from typing import Optional, Sequence
from urllib.parse import urlparse

# Generic fallback env var (checked after provider-specific vars)
GENERIC_LLM_API_KEY_ENV = "LLM_API_KEY"

# Provider -> ordered env var candidates (first non-empty wins)
PROVIDER_AUTH_ENV_VAR_CANDIDATES: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    "stepfun": ("STEPFUN_API_KEY", "OPENAI_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "ollama": ("OLLAMA_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN"),
    "qwen": ("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
    "kimi": ("KIMI_API_KEY", "MOONSHOT_API_KEY", "OPENAI_API_KEY"),
    "moonshot": ("MOONSHOT_API_KEY", "KIMI_API_KEY", "OPENAI_API_KEY"),
    "zhipu": ("ZHIPU_API_KEY", "OPENAI_API_KEY"),
}

# base_url host patterns -> logical provider id for env var lookup
BASE_URL_PROVIDER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("deepseek.com", "deepseek"),
    ("stepfun", "stepfun"),
    ("dashscope", "qwen"),
    ("api.kimi.com", "kimi"),
    ("moonshot.ai", "moonshot"),
    ("moonshot.cn", "moonshot"),
    ("open.bigmodel.cn", "zhipu"),
    ("generativelanguage.googleapis.com", "gemini"),
    ("anthropic.com", "anthropic"),
)

# OpenAI-compatible provider aliases -> transport provider
OPENAI_COMPATIBLE_ALIASES: frozenset[str] = frozenset({
    "openai",
    "deepseek",
    "stepfun",
    "kimi",
    "moonshot",
    "zhipu",
    "qwen",
    "anthropic-openai",
})

# Provider-specific base URL env vars
PROVIDER_BASE_URL_ENV: dict[str, str] = {
    "openai": "OPENAI_BASE_URL",
    "deepseek": "DEEPSEEK_BASE_URL",
    "stepfun": "STEPFUN_BASE_URL",
    "kimi": "KIMI_BASE_URL",
    "moonshot": "MOONSHOT_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
}


def normalize_secret(value: Optional[str]) -> Optional[str]:
    """Treat null/empty/whitespace-only values as missing."""
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value) if value else None
    stripped = value.strip()
    return stripped if stripped else None


def detect_provider_from_base_url(base_url: Optional[str]) -> Optional[str]:
    """Infer logical provider id from base_url host."""
    if not base_url:
        return None
    host = (urlparse(base_url).netloc or base_url).lower()
    for pattern, provider_id in BASE_URL_PROVIDER_PATTERNS:
        if pattern in host:
            return provider_id
    return None


def resolve_auth_provider_id(provider: str, base_url: Optional[str] = None) -> str:
    """Resolve the provider id used for env var lookup."""
    normalized = (provider or "openai").lower().strip()
    detected = detect_provider_from_base_url(base_url)
    if detected:
        return detected
    if normalized in OPENAI_COMPATIBLE_ALIASES or normalized in PROVIDER_AUTH_ENV_VAR_CANDIDATES:
        return normalized
    return normalized


def is_openai_compatible(provider: str) -> bool:
    """Whether the provider uses OpenAI-compatible API transport."""
    normalized = (provider or "").lower().strip()
    return normalized in OPENAI_COMPATIBLE_ALIASES or normalized == "openai"


def _first_env_value(env_vars: Sequence[str]) -> Optional[str]:
    for env_var in env_vars:
        value = normalize_secret(os.environ.get(env_var))
        if value:
            return value
    return None


def get_env_var_candidates(provider: str, base_url: Optional[str] = None) -> list[str]:
    """Return ordered env var names to try for API key resolution."""
    auth_provider = resolve_auth_provider_id(provider, base_url)
    detected = detect_provider_from_base_url(base_url)
    candidates: list[str] = []

    # When base_url indicates a specific provider, prefer its env vars first
    if detected:
        for env_var in PROVIDER_AUTH_ENV_VAR_CANDIDATES.get(detected, ()):
            if env_var not in candidates:
                candidates.append(env_var)

    for env_var in PROVIDER_AUTH_ENV_VAR_CANDIDATES.get(auth_provider, ()):
        if env_var not in candidates:
            candidates.append(env_var)

    if GENERIC_LLM_API_KEY_ENV not in candidates:
        candidates.append(GENERIC_LLM_API_KEY_ENV)

    if is_openai_compatible(provider) or is_openai_compatible(auth_provider):
        if "OPENAI_API_KEY" not in candidates:
            candidates.append("OPENAI_API_KEY")

    return candidates


def resolve_api_key(
    provider: str,
    config_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve API key with priority: config -> provider env vars -> generic fallback.

    @generated AI Assistant - 2026-08-04 11:35:00
    """
    key = normalize_secret(config_key)
    if key:
        return key
    return _first_env_value(get_env_var_candidates(provider, base_url))


def resolve_base_url(
    provider: str,
    config_base_url: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve base URL from config or provider-specific env var.

    @generated AI Assistant - 2026-08-04 11:35:00
    """
    url = normalize_secret(config_base_url)
    if url:
        return url

    auth_provider = resolve_auth_provider_id(provider, config_base_url)
    env_var = PROVIDER_BASE_URL_ENV.get(auth_provider)
    if env_var:
        return normalize_secret(os.environ.get(env_var))
    return None


def normalize_temperature(base_url: Optional[str], temperature: float) -> float:
    """
    Kimi Code API only accepts temperature=1 for some models.

    @generated AI Assistant - 2026-08-04 13:35:00
    """
    if base_url and "api.kimi.com" in base_url and temperature != 1.0:
        return 1.0
    return temperature


def resolve_chat_completions_url(base_url: Optional[str]) -> str:
    """
    Build OpenAI-compatible chat completions endpoint URL for raw HTTP clients.

    @generated AI Assistant - 2026-08-04 11:35:00
    """
    if not base_url:
        raise ValueError("base_url is required to build chat completions URL")

    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    if "deepseek.com" in url or "api.kimi.com" in url:
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def get_env_config_hints(
    provider: str,
    base_url: Optional[str] = None,
) -> list[str]:
    """
    Detect common .env misconfiguration (wrong var name or provider mismatch).

    @generated AI Assistant - 2026-08-04 11:45:00
    """
    hints: list[str] = []
    needed = get_env_var_candidates(provider, base_url)
    needed_set = set(needed)
    detected = detect_provider_from_base_url(base_url)

    for env_var in needed:
        raw = os.environ.get(env_var)
        if raw is not None and raw.strip() == "":
            hints.append(f".env 中 `{env_var}=` 为空，请填入密钥或删除该行")

    for env_var, value in os.environ.items():
        if env_var in needed_set or not env_var.endswith("_API_KEY"):
            continue
        secret = normalize_secret(value)
        if not secret:
            continue
        if env_var == "GEMINI_API_KEY" and secret.startswith("sk-kimi-"):
            hints.append(
                "检测到 `GEMINI_API_KEY` 中是 Kimi 密钥（sk-kimi-），"
                "当前 config 使用的是 DeepSeek。请改用 `KIMI_API_KEY` 或 `MOONSHOT_API_KEY`，"
                "并将 config.yaml 的 base_url 改为 https://api.kimi.com/coding/v1"
            )
        elif env_var not in needed_set and env_var != GENERIC_LLM_API_KEY_ENV:
            hints.append(
                f"`.env` 中已设置 `{env_var}`，但当前 provider 不会读取它；"
                f"请改用: {', '.join(needed[:3])}"
            )

    if detected == "deepseek" and normalize_secret(os.environ.get("KIMI_API_KEY")):
        hints.append(
            "已设置 `KIMI_API_KEY`，但 config 指向 DeepSeek；"
            "请设置 `DEEPSEEK_API_KEY` 或 `LLM_API_KEY`，或改用 Moonshot 配置"
        )

    return hints


def format_missing_key_error(
    provider: str,
    base_url: Optional[str] = None,
    config_path: str = "llm.api_key",
) -> str:
    """
    Build a provider-aware error message listing env var options.

    @generated AI Assistant - 2026-08-04 11:35:00
    """
    auth_provider = resolve_auth_provider_id(provider, base_url)
    env_vars = get_env_var_candidates(provider, base_url)
    env_hint = " / ".join(f"`{v}`" for v in env_vars[:4])

    detected = detect_provider_from_base_url(base_url)
    provider_label = auth_provider
    if detected and detected != auth_provider:
        provider_label = f"{auth_provider} (detected: {detected})"

    return (
        f"API key required for provider '{provider_label}' but not provided. "
        f"Set `{config_path}` in config.yaml, or export one of: {env_hint}."
    )
