"""Firecrawl's finite TTL must reach the existing session-expiry lifecycle."""
from unittest.mock import Mock

import pytest

from plugins.browser.firecrawl.provider import FirecrawlBrowserProvider
from tools.browser_tool_lifecycle import _session_has_expired


@pytest.mark.parametrize('ttl_env,ttl', [(None, 300), ('60', 60), ('invalid', 300)])
def test_requested_ttl_is_cached_as_expiry(monkeypatch, ttl_env, ttl):
    if ttl_env is None:
        monkeypatch.delenv('FIRECRAWL_BROWSER_TTL', raising=False)
    else:
        monkeypatch.setenv('FIRECRAWL_BROWSER_TTL', ttl_env)
    provider = FirecrawlBrowserProvider()
    monkeypatch.setattr(provider, '_headers', lambda: {})
    monkeypatch.setattr(provider, '_api_url', lambda: 'https://example.invalid')
    response = Mock(status_code=200)
    response.json.return_value = {'id': 'fixture-session', 'cdpUrl': 'wss://example.invalid'}
    post = Mock(return_value=response)
    monkeypatch.setattr(provider, '_post_create', post)
    # The deadline starts before provisioning, not after network latency.
    monkeypatch.setattr('time.time', lambda: 1000.0)
    session = provider.create_session('fixture')
    assert session.get('expires_at') == 1000.0 + ttl
    assert not _session_has_expired(session, now=1000.0 + ttl - 0.001)
    assert _session_has_expired(session, now=1000.0 + ttl)
    assert post.call_args.args[2] == {'ttl': ttl}
