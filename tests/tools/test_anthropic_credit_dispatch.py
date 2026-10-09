from pathlib import Path
from types import SimpleNamespace
import pytest


def test_claude_api_dispatch_refuses_four_dollars(monkeypatch):
    from tools import delegate_tool_config as routes
    from agent import anthropic_credit_guard as guard
    real = guard.enforce_anthropic_credit_floor
    monkeypatch.setattr(guard, 'enforce_anthropic_credit_floor', lambda **kw:
        real(**kw, tracker=lambda: {'remaining_usd': 4.0}, enabled=True))
    parent = SimpleNamespace(provider='custom:anthropic-api-credits', model='claude-opus-5-5',
        base_url='https://api.anthropic.com', api_mode='anthropic_messages', reasoning_config=None)
    with pytest.raises(ValueError, match='below.*5'):
        routes._resolve_child_runtime(parent, {}, 'fixture-key', model=None, override_provider=None,
            override_base_url=None, override_api_key=None, override_api_mode=None,
            override_acp_command=None, override_acp_args=None)


@pytest.mark.parametrize('balance,allowed', [(4.0,False),(5.0,True),(8.74,True),(None,False),(float('nan'),False)])
def test_credit_floor_boundaries(balance, allowed):
    from agent.anthropic_credit_guard import enforce_anthropic_credit_floor as gate
    args = dict(provider='custom', base_url='https://api.anthropic.com/v1', api_key='fixture',
                enabled=True, tracker=lambda: {'remaining_usd': balance})
    if allowed:
        assert gate(**args)['remaining_usd'] == balance
    else:
        with pytest.raises(ValueError):
            gate(**args)


def test_oauth_and_codex_do_not_read_api_balance():
    from agent.anthropic_credit_guard import enforce_anthropic_credit_floor as gate
    def forbidden():
        pytest.fail('included routes must not read the API grant')
    assert gate(provider='anthropic', base_url='https://api.anthropic.com',
                api_key='sk-ant-oat-fixture', enabled=True, tracker=forbidden) is None
    assert gate(provider='openai-codex', base_url='https://chatgpt.com/backend-api/codex',
                enabled=True, tracker=forbidden) is None
