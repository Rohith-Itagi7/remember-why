"""Optional, privacy-conscious Sentry tracing for local agent runs."""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from typing import Any, Iterator

_initialized = False
_sentry_sdk: Any | None = None


def _load_sentry() -> tuple[Any, list[Any]]:
    """Load the SDK and supported LangChain and MCP integrations."""
    import sentry_sdk
    from sentry_sdk.integrations.langchain import LangchainIntegration
    from sentry_sdk.integrations.mcp import MCPIntegration

    return sentry_sdk, [
        LangchainIntegration(include_prompts=False),
        MCPIntegration(include_prompts=False),
    ]


def _scrub_event(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
    """Remove request content and local details from captured error events."""
    for key in ("request", "user", "extra", "breadcrumbs", "message", "logentry"):
        event.pop(key, None)
    contexts = event.get("contexts", {})
    if isinstance(contexts, dict):
        contexts.pop("request", None)
        contexts.pop("user", None)

    exception_info = event.get("exception", {})
    for exception in exception_info.get("values", []) if isinstance(exception_info, dict) else []:
        exception["value"] = "<redacted>"
        stacktrace = exception.get("stacktrace", {})
        for frame in stacktrace.get("frames", []) if isinstance(stacktrace, dict) else []:
            for key in ("vars", "filename", "abs_path", "pre_context", "context_line", "post_context"):
                frame.pop(key, None)
    return event


def _scrub_transaction(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
    """Keep span structure and timing while dropping all span payloads."""
    for key in ("request", "user", "extra", "breadcrumbs", "message", "logentry"):
        event.pop(key, None)
    for span in event.get("spans", []):
        span.pop("data", None)
        span.pop("description", None)
        span.pop("tags", None)
    return event


def initialize_sentry() -> bool:
    """Initialize Sentry only when a DSN is configured; return whether it is active.

    Prompt capture is disabled in both AI integrations, and payload scrubbers drop
    span data, requests, and local error details before events are sent.
    """
    global _initialized, _sentry_sdk

    if _initialized:
        return True
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False

    try:
        sdk, integrations = _load_sentry()
        sample_rate = float(os.environ.get("SENTRY_TRACES_SAMPLE_RATE", "1.0"))
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("SENTRY_TRACES_SAMPLE_RATE must be between 0.0 and 1.0")
        sdk.init(
            dsn=dsn,
            traces_sample_rate=sample_rate,
            trace_lifecycle="static",
            stream_gen_ai_spans=False,
            send_default_pii=False,
            data_collection={
                "user_info": False,
                "http_bodies": [],
                "gen_ai": {"inputs": False, "outputs": False},
                "database_query_data": False,
                "stack_frame_variables": False,
                "frame_context_lines": 0,
            },
            include_local_variables=False,
            include_source_context=False,
            integrations=integrations,
            before_send=_scrub_event,
            before_send_transaction=_scrub_transaction,
        )
    except ImportError:
        warnings.warn(
            "SENTRY_DSN is set but sentry-sdk or an AI integration is unavailable. "
            "Install project requirements to enable tracing. The application will continue without it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    except Exception as error:
        warnings.warn(
            f"Sentry tracing could not be initialized ({type(error).__name__}). The application will continue without it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    _sentry_sdk = sdk
    _initialized = True
    return True


@contextmanager
def agent_run_span() -> Iterator[None]:
    """Create a constant-named parent span without attaching the user question."""
    if _initialized and _sentry_sdk is not None:
        with _sentry_sdk.start_span(op="ai.agent", name="remember_why.ask"):
            yield
        return
    yield
