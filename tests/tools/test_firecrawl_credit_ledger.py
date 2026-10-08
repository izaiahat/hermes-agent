import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools import firecrawl_ledger as bridge


def test_bridge_sdk_mcp_receipts_and_ambiguous_failure(monkeypatch):
    records = []
    adapter = SimpleNamespace(record_operation=lambda *a, **kw: records.append(kw))
    monkeypatch.setattr(bridge, '_client', lambda: adapter)
    response = {'id':'job-1','creditsUsed':2,'success':True,'data':{'web':[]}}
    assert bridge.sdk_call('sdk.search','/v2/search',lambda **kw: response,query='fixture')==response
    result=SimpleNamespace(isError=False,content=[SimpleNamespace(text=json.dumps(response))])
    op=bridge.begin('mcp.search','mcp/firecrawl_search',{'query':'fixture'})
    bridge.finish(op,'mcp.search','mcp/firecrawl_search',{'query':'fixture'},result)
    assert [r['outcome'] for r in records]==['started','success','started','success']
    assert records[-1]['payload']['creditsUsed']==2
    def failed(**kw):
        raise TimeoutError('fixture ambiguity')
    with pytest.raises(TimeoutError):
        bridge.sdk_call('sdk.search','/v2/search',failed,query='fixture')
    assert records[-1]['outcome']=='ambiguous_transport_error'


def test_sdk_and_mcp_share_ledger_without_secret_retention(tmp_path,monkeypatch):
    client=Path('/home/ubuntu/business/.worktrees/firecrawl-integration-20261007/scripts/firecrawl_client.py')
    if not client.exists():
        pytest.skip('workspace bridge integration runs on the GLP host')
    ledger=tmp_path/'ops.jsonl'
    monkeypatch.setenv('GLP_FIRECRAWL_CLIENT',str(client))
    monkeypatch.setenv('FIRECRAWL_OPERATION_LEDGER',str(ledger))
    monkeypatch.setattr(bridge,'_CLIENT',None)
    payload={'success':True,'data':{'metadata':{'creditsUsed':1,'scrapeId':'scrape-fixture'}}}
    assert bridge.sdk_call('hermes.web_search','/v2/search',lambda **kw:payload,query='PRIVATE QUERY')==payload
    result=SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))],isError=False)
    op=bridge.begin('hermes.mcp.firecrawl_scrape','mcp/firecrawl_scrape',{'url':'https://example.invalid/private'})
    bridge.finish(op,'hermes.mcp.firecrawl_scrape','mcp/firecrawl_scrape',{},result)
    rows=[json.loads(s) for s in ledger.read_text().splitlines()]
    assert len(rows)==4
    assert all(r['creditsUsed']==1 for r in rows if r['outcome']=='success')
    assert 'PRIVATE QUERY' not in ledger.read_text()
    assert '/private' not in ledger.read_text()


def test_sdk4493_search_and_redirect_metadata(monkeypatch):
    import asyncio
    from firecrawl.v2.types import Document, DocumentMetadata, SearchData, SearchResultWeb
    from plugins.web.firecrawl import provider
    search = SearchData(web=[SearchResultWeb(url='https://www.fda.gov/', title='FDA')])
    assert provider._extract_web_search_results(search)[0]['url'] == 'https://www.fda.gov/'
    document = Document(markdown='fixture', metadata=DocumentMetadata(
        source_url='https://redirect.example.invalid/', title='Redirect', credits_used=1))
    monkeypatch.setattr(provider, '_get_firecrawl_client', lambda: SimpleNamespace(scrape=lambda **kw: document))
    monkeypatch.setattr(provider, 'check_website_access', lambda url: None)
    monkeypatch.setattr(provider, 'is_safe_url', lambda url: url != 'https://redirect.example.invalid/')
    monkeypatch.setattr(bridge, '_client', lambda: None)
    result = asyncio.run(provider._scrape_one('https://www.fda.gov/', ['markdown'], 'markdown'))
    assert result['url'] == 'https://redirect.example.invalid/'
    assert result['error'] == provider._UNSAFE_REDIRECT_MSG
    assert result['content'] == ''


@pytest.mark.parametrize('denial', ['network', 'policy'])
def test_all_typed_redirect_candidates_are_checked(monkeypatch, denial):
    import asyncio
    from firecrawl.v2.types import Document, DocumentMetadata
    from plugins.web.firecrawl import provider
    safe, denied = 'https://example.invalid/', 'https://denied.example.invalid/'
    doc = Document(markdown='guarded fixture', metadata=DocumentMetadata(url=safe, source_url=denied))
    monkeypatch.setattr(provider, '_get_firecrawl_client', lambda: SimpleNamespace(scrape=lambda **kw: doc))
    monkeypatch.setattr(provider, 'is_safe_url', lambda u: not (denial == 'network' and u == denied))
    blocked = {'host': 'denied.example.invalid', 'rule': 'fixture', 'message': 'Policy denied', 'source': 'fixture'}
    monkeypatch.setattr(provider, 'check_website_access', lambda u: blocked if denial == 'policy' and u == denied else None)
    monkeypatch.setattr(bridge, '_client', lambda: None)
    result = asyncio.run(provider._scrape_one(safe, ['markdown'], 'markdown'))
    assert result['content'] == ''
    assert result['error'] == (provider._UNSAFE_REDIRECT_MSG if denial == 'network' else 'Policy denied')


@pytest.mark.parametrize('path', ['sdk', 'mcp'])
@pytest.mark.parametrize('effect', ['cancel', 'timeout', 'completed'])
def test_original_cancel_survives_terminal_accounting_failure(monkeypatch, path, effect):
    import asyncio
    from tools import mcp_tool_handlers as handlers
    outcomes = []
    def record(*args, **kw):
        outcomes.append(kw['outcome'])
        if kw['outcome'] != 'started':
            raise OSError('offline terminal persistence failure')
    monkeypatch.setattr(bridge, '_client', lambda: SimpleNamespace(record_operation=record))
    original = asyncio.CancelledError('offline caller cancel') if effect == 'cancel' else TimeoutError('offline uncertainty')
    calls = []
    def sdk(**kw):
        calls.append('dispatched')
        if effect != 'completed':
            raise original
        return {'success': True, 'data': {'web': []}}
    async def rpc(*args, **kw):
        return sdk()
    server = SimpleNamespace(_rpc_lock=asyncio.Lock(), session=SimpleNamespace(call_tool=rpc),
        _pending_call_context=None, _inflight_tasks=set(), _reconnecting=False)
    monkeypatch.setattr(handlers, '_trust_gate_check', lambda *a: None)
    monkeypatch.setattr(handlers, '_check_circuit_breaker', lambda *a: None)
    monkeypatch.setattr(handlers, '_acquire_call_server', lambda *a: (server, None))
    monkeypatch.setattr(handlers, '_tool_is_read_only', lambda *a: False)
    monkeypatch.setattr(handlers, '_dispatch', lambda name, srv, op, call, *a, **kw: asyncio.run(call()))
    expected = RuntimeError if effect == 'completed' else type(original)
    with pytest.raises(expected) as caught:
        if path == 'sdk':
            bridge.sdk_call('fixture', '/v2/scrape', sdk, url='https://example.invalid/')
        else:
            handlers._make_tool_handler('firecrawl', 'firecrawl_scrape', 30)({'url': 'https://example.invalid/'})
    if effect != 'completed':
        assert caught.value is original
    else:
        assert 'do not replay' in str(caught.value)
    assert calls == ['dispatched']
    assert outcomes == ['started', 'success' if effect == 'completed' else 'ambiguous_transport_error']
    assert server._pending_call_context is None
    assert not server._inflight_tasks


@pytest.fixture
def native_dispatch(monkeypatch):
    import asyncio
    from tools import mcp_tool_handlers as handlers
    effects, outcomes, reconnects = [], [], []

    async def rpc(*args, **kwargs):
        effects.append('completed')
        return SimpleNamespace(content=[], isError=False)

    server = SimpleNamespace(
        _rpc_lock=asyncio.Lock(), session=SimpleNamespace(call_tool=rpc),
        _pending_call_context=None, _inflight_tasks=set(), _reconnecting=False)
    monkeypatch.setattr(handlers, '_trust_gate_check', lambda *a: None)
    monkeypatch.setattr(handlers, '_check_circuit_breaker', lambda *a: None)
    monkeypatch.setattr(handlers, '_acquire_call_server', lambda *a: (server, None))
    monkeypatch.setattr(handlers, '_tool_is_read_only', lambda *a: True)
    monkeypatch.setattr(handlers._loop, '_run_on_mcp_loop', lambda call, **kw: asyncio.run(call()))
    monkeypatch.setattr(handlers, '_core', SimpleNamespace(
        _bump_server_error=lambda *a, **kw: None, _reset_server_error=lambda *a: None,
        _STDIO_RESPAWN_WAIT_SEC=15))
    monkeypatch.setattr(handlers, '_lookup_reconnectable_server', lambda *a, **kw: server)
    monkeypatch.setattr(handlers, '_mcp_loop_running', lambda: True)

    def reconnect(*a, **kw):
        reconnects.append('requested')
        server._stdio_children_dead = lambda: False
        return True

    monkeypatch.setattr(handlers._loop, '_signal_reconnect', reconnect)
    monkeypatch.setattr(handlers._loop, '_signal_reconnect_and_wait', reconnect)
    return server, effects, outcomes, reconnects


@pytest.mark.parametrize('fault_kind', ['broken-pipe', 'session-expired', '401-message', '401-typed'])
def test_completed_terminal_accounting_never_recovers(monkeypatch, native_dispatch, fault_kind):
    import httpx
    from tools import mcp_tool_handlers as handlers
    fault = {
        'broken-pipe': BrokenPipeError('broken pipe'),
        'session-expired': RuntimeError('session expired'),
        '401-message': OSError('HTTP 401 Unauthorized'),
        '401-typed': httpx.HTTPStatusError('HTTP 401 Unauthorized',
            request=httpx.Request('POST', 'https://example.invalid/'), response=httpx.Response(401)),
    }[fault_kind]
    server, effects, outcomes, reconnects = native_dispatch

    def record(*a, **kw):
        outcomes.append(kw['outcome'])
        if kw['outcome'] != 'started':
            raise fault

    monkeypatch.setattr(bridge, '_client', lambda: SimpleNamespace(record_operation=record))
    result = json.loads(handlers._make_tool_handler('firecrawl', 'firecrawl_scrape', 30)({}))
    assert 'outcome unknown; do not replay' in result['error']
    assert (effects, outcomes, reconnects) == (['completed'], ['started', 'success'], [])
    assert result['outcome_uncertain'] is True and result['retry'] is False
    assert server._pending_call_context is None and not server._inflight_tasks


def test_genuine_pre_effect_stdio_death_still_recovers(native_dispatch):
    from tools import mcp_tool_handlers as handlers
    server, effects, outcomes, reconnects = native_dispatch
    server._stdio_children_dead = lambda: True
    result = json.loads(handlers._make_tool_handler('offline-control', 'fixture', 30)({}))
    assert 'error' not in result
    assert effects == ['completed'] and reconnects == ['requested'] and not outcomes
    assert server._pending_call_context is None and not server._inflight_tasks


@pytest.mark.parametrize('wrapper', ['context', 'cause', 'group'])
def test_accounting_boundary_overrides_wrapped_transport(monkeypatch, native_dispatch, wrapper):
    from tools import mcp_tool_errors as errors
    from tools import mcp_tool_handlers as handlers
    _, effects, _, reconnects = native_dispatch
    fault = BrokenPipeError('broken pipe')
    boundary = None
    try:
        raise fault
    except Exception:
        try:
            # Use the maintained producer to obtain a real retained-context accounting error.
            monkeypatch.setattr(bridge, '_client', lambda: SimpleNamespace(
                record_operation=lambda *a, **kw: (_ for _ in ()).throw(fault)))
            bridge.finish('started-id', 'fixture', 'mcp/fixture', {})
        except Exception as exc:
            boundary = exc
    assert boundary is not None
    assert boundary.__context__ is fault
    if wrapper == 'group':
        outer = ExceptionGroup('session expired', [RuntimeError('broken pipe'), boundary])
    else:
        outer = RuntimeError('session expired')
        setattr(outer, '__' + wrapper + '__', boundary)
    assert errors._is_session_expired_error(outer) is False
    assert errors._is_auth_error(outer) is False

    async def raise_outer():
        raise outer

    recovered = []
    def must_not_recover(*a):
        recovered.append(True)
        return '{}'
    result = json.loads(handlers._dispatch('firecrawl', native_dispatch[0], 'fixture', raise_outer, 30,
                                         (must_not_recover,), lambda *a: None))
    assert 'outcome unknown; do not replay' in result['error']
    assert result['retry'] is False
    assert effects == [] and reconnects == [] and recovered == []
