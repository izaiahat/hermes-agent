from unittest.mock import MagicMock, patch


class TestBootMdRuntimeResolution:
    def test_run_boot_agent_passes_resolved_runtime_and_model(self):
        fake_runtime = {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "***",
            "api_mode": "codex_responses",
        }

        with (
            patch(
                "gateway.run._resolve_runtime_agent_kwargs",
                return_value=fake_runtime,
            ),
            patch("gateway.run._resolve_gateway_model", return_value="gpt-5.4"),
            patch("run_agent.AIAgent") as mock_agent,
        ):
            from gateway.builtin_hooks.boot_md import _run_boot_agent

            mock_agent.return_value.run_conversation.return_value = {
                "final_response": "[SILENT]"
            }
            _run_boot_agent("check things")

            kwargs = mock_agent.call_args.kwargs
            assert kwargs["provider"] == "openai-codex"
            assert kwargs["base_url"] == "https://chatgpt.com/backend-api/codex"
            assert kwargs["api_key"] == "***"
            assert kwargs["api_mode"] == "codex_responses"
            assert kwargs["model"] == "gpt-5.4"
            assert kwargs["quiet_mode"] is True
            assert kwargs["skip_context_files"] is True
            assert kwargs["skip_memory"] is True
            assert kwargs["max_iterations"] == 20

    def test_run_boot_agent_fills_default_model_from_provider_when_missing(self):
        fake_runtime = {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "***",
            "api_mode": "codex_responses",
        }

        with (
            patch(
                "gateway.run._resolve_runtime_agent_kwargs",
                return_value=fake_runtime,
            ),
            patch("gateway.run._resolve_gateway_model", return_value=""),
            patch(
                "hermes_cli.models.get_default_model_for_provider",
                return_value="gpt-5.4",
            ),
            patch("run_agent.AIAgent") as mock_agent,
        ):
            from gateway.builtin_hooks.boot_md import _run_boot_agent

            mock_agent.return_value.run_conversation.return_value = {
                "final_response": "[SILENT]"
            }
            _run_boot_agent("check things")

            assert mock_agent.call_args.kwargs["model"] == "gpt-5.4"
