"""Tests for optional, content-scrubbed Sentry initialization."""

import os
import unittest
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from app.observability import sentry


class SentryInitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        sentry._initialized = False
        sentry._sentry_sdk = None

    def tearDown(self) -> None:
        sentry._initialized = False
        sentry._sentry_sdk = None

    def test_missing_dsn_is_safe_and_skips_sdk_loading(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(sentry, "_load_sentry") as load:
            self.assertFalse(sentry.initialize_sentry())
        load.assert_not_called()

    def test_missing_optional_sdk_does_not_break_application(self) -> None:
        with patch.dict(os.environ, {"SENTRY_DSN": "configured"}), patch.object(
            sentry, "_load_sentry", side_effect=ImportError("not installed")
        ), self.assertWarns(RuntimeWarning):
            self.assertFalse(sentry.initialize_sentry())
        self.assertFalse(sentry._initialized)

    def test_configured_init_uses_privacy_safe_integrations_and_hooks(self) -> None:
        sdk = Mock()
        integrations = [object(), object()]
        with patch.dict(os.environ, {"SENTRY_DSN": "test-dsn", "SENTRY_TRACES_SAMPLE_RATE": "0.25"}), patch.object(
            sentry, "_load_sentry", return_value=(sdk, integrations)
        ):
            self.assertTrue(sentry.initialize_sentry())

        kwargs = sdk.init.call_args.kwargs
        self.assertEqual(kwargs["dsn"], "test-dsn")
        self.assertEqual(kwargs["traces_sample_rate"], 0.25)
        self.assertEqual(kwargs["trace_lifecycle"], "static")
        self.assertFalse(kwargs["stream_gen_ai_spans"])
        self.assertFalse(kwargs["send_default_pii"])
        self.assertEqual(kwargs["data_collection"]["gen_ai"], {"inputs": False, "outputs": False})
        self.assertFalse(kwargs["data_collection"]["user_info"])
        self.assertFalse(kwargs["include_local_variables"])
        self.assertFalse(kwargs["include_source_context"])
        self.assertEqual(kwargs["integrations"], integrations)
        self.assertIs(kwargs["before_send"], sentry._scrub_event)
        self.assertIs(kwargs["before_send_transaction"], sentry._scrub_transaction)

    def test_initialization_is_idempotent(self) -> None:
        sdk = Mock()
        with patch.dict(os.environ, {"SENTRY_DSN": "test-dsn"}), patch.object(
            sentry, "_load_sentry", return_value=(sdk, [])
        ) as load:
            self.assertTrue(sentry.initialize_sentry())
            self.assertTrue(sentry.initialize_sentry())
        load.assert_called_once()
        sdk.init.assert_called_once()

    def test_sdk_integrations_disable_prompt_capture(self) -> None:
        _, integrations = sentry._load_sentry()
        self.assertEqual([integration.__class__.__name__ for integration in integrations], [
            "LangchainIntegration", "MCPIntegration"
        ])
        self.assertTrue(all(integration.include_prompts is False for integration in integrations))

    def test_invalid_sample_rate_fails_closed(self) -> None:
        with patch.dict(os.environ, {"SENTRY_DSN": "test-dsn", "SENTRY_TRACES_SAMPLE_RATE": "2"}), patch.object(
            sentry, "_load_sentry", return_value=(Mock(), [])
        ), self.assertWarns(RuntimeWarning):
            self.assertFalse(sentry.initialize_sentry())
        self.assertFalse(sentry._initialized)


class SentryPrivacyTests(unittest.TestCase):
    def test_transaction_scrubber_keeps_structure_but_removes_payloads(self) -> None:
        event = {
            "transaction": "remember_why.ask",
            "spans": [{"op": "ai.tool", "description": "private OCR text", "data": {"path": "C:/secret"}, "tags": {"q": "private"}}],
        }
        result = sentry._scrub_transaction(event, {})
        self.assertEqual(result["spans"], [{"op": "ai.tool"}])
        self.assertEqual(result["transaction"], "remember_why.ask")

    def test_error_scrubber_removes_user_content_and_local_paths(self) -> None:
        event = {
            "message": "private question",
            "request": {"url": "private"},
            "extra": {"ocr": "private text"},
            "exception": {"values": [{"value": "private path C:/screenshots/a.png", "stacktrace": {"frames": [
                {"filename": "C:/screenshots/a.png", "vars": {"text": "private"}, "function": "ingest"}
            ]}}]},
        }
        result = sentry._scrub_event(event, {})
        self.assertNotIn("message", result)
        self.assertNotIn("request", result)
        self.assertNotIn("extra", result)
        exception = result["exception"]["values"][0]
        self.assertEqual(exception["value"], "<redacted>")
        self.assertEqual(exception["stacktrace"]["frames"], [{"function": "ingest"}])

    def test_agent_span_uses_static_name_and_no_question_data(self) -> None:
        @contextmanager
        def fake_span(**kwargs):
            calls.append(kwargs)
            yield

        calls = []
        sentry._initialized = True
        sentry._sentry_sdk = Mock(start_span=fake_span)
        with sentry.agent_run_span():
            pass
        self.assertEqual(calls, [{"op": "ai.agent", "name": "remember_why.ask"}])

    def test_cli_ask_path_initializes_optional_sentry(self) -> None:
        from main import main

        with patch("app.observability.sentry.initialize_sentry") as initialize, patch(
            "app.agent.agent.ask_memory_agent", return_value="Grounded answer."
        ):
            with redirect_stdout(StringIO()):
                main(["--ask", "What did I save about MCP?"])
        initialize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
