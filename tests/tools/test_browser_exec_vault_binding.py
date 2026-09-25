"""Registered browser_exec must bind the vault to its actual email-first tab."""
import asyncio
import contextlib
import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from tools import browser_supervisor as supervision
from tools import browser_use_cli as browser
from tools import browser_vault_tool as vault
from tools.registry import registry


@pytest.mark.parametrize('failure', [RuntimeError, ImportError])
def test_bound_exec_evaluation_exception_never_uses_legacy_daemon(monkeypatch, failure):
    def raise_on_evaluate(expression):
        raise failure('fixture-secret-must-not-escape')

    supervisor = SimpleNamespace(browser_exec_target_id='named-tab', evaluate_runtime=raise_on_evaluate)
    monkeypatch.setattr(supervision.SUPERVISOR_REGISTRY, 'get', lambda task: supervisor)
    legacy_calls = []
    monkeypatch.setattr('tools.browser_tool._last_session_key', lambda task: legacy_calls.append('session') or task)
    monkeypatch.setattr('tools.browser_tool_session._run_browser_command',
                        lambda *args: legacy_calls.append('eval') or {'success': True, 'data': {'result': 'wrong tab'}})

    result = vault._eval_js('task', 'window.location.href')
    assert result == {'success': False, 'error': 'Bound browser evaluation failed.'}
    assert not legacy_calls

    supervisor.browser_exec_target_id = None
    assert vault._eval_js('task', 'window.location.href') == {'success': True, 'result': 'wrong tab'}
    assert legacy_calls == ['session', 'eval']


def test_named_exec_email_first_tab_is_the_vault_target(monkeypatch):
    task = 'worker-task-not-session-name'
    supervisor = supervision.CDPSupervisor(task_id=task, cdp_url='ws://fixture/devtools/browser/1')
    supervisor._loop = SimpleNamespace(is_running=lambda: True)
    supervisor._active = True
    supervisor._page_session_id = 'blank'
    pages = {'blank': 'about:blank', 'other-login': 'https://example.com/other',
             'email-first': 'https://example.com/login'}
    attached = []

    async def cdp(method, params=None, **kwargs):
        params = params or {}
        if method == 'Target.getTargets':
            return {'result': {'targetInfos': [dict(targetId=k, url=v, type='page') for k, v in pages.items()]}}
        if method == 'Target.attachToTarget':
            attached.append(params['targetId'])
            return {'result': {'sessionId': params['targetId']}}
        if method == 'Runtime.evaluate':
            target = kwargs['session_id']
            expr = params['expression']
            value = pages[target] if expr == 'window.location.href' else (target == 'other-login')
            if 'querySelectorAll' in expr:
                value = []  # email-first page has no fillable password
            return {'result': {'result': {'value': value}}}
        return {'result': {}}

    async def noop(*a, **kw):
        pass

    monkeypatch.setattr(supervision, '_schedule', lambda coro, loop, **kw: asyncio.run(coro))
    monkeypatch.setattr(supervisor, '_cdp', cdp)
    monkeypatch.setattr(supervisor, '_enable_page_domains', noop)
    monkeypatch.setattr(supervisor, '_install_dialog_bridge', noop)
    reg = supervision._SupervisorRegistry()
    reg._by_task[task] = supervisor
    def start(**kw):
        assert kw['task_id'] == task
        return supervisor
    monkeypatch.setattr(reg, 'get_or_start', start)
    monkeypatch.setattr(supervision, 'SUPERVISOR_REGISTRY', reg)
    monkeypatch.setattr(browser, '_find_cli', lambda: ['fixture-cli'])
    monkeypatch.setattr(browser, '_base_subprocess_env', lambda: {})
    def route(env, session, task_id, local):
        assert session == 'named-insurance' and task_id == task
        env.update(BU_CDP_WS=supervisor.cdp_url, _HERMES_BU_PRIVATE_BROWSER='1')
    monkeypatch.setattr(browser, '_route_backend', route)
    def run(cmd, code, env, timeout):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            # Target.getTargetInfo without a target ID returns the BROWSER,
            # not the current tab. Use the harness's explicit current_tab API.
            exec(code, {'current_tab': lambda: {'targetId': 'email-first'},
                        'cdp': lambda method: {'targetInfo': {'targetId': 'browser-not-page'}}})
        return subprocess.CompletedProcess(cmd, 0, out.getvalue(), '')
    monkeypatch.setattr(browser, '_run_cli_killing_process_group', run)
    meta = SimpleNamespace(kind='login', origin='https://example.com', allowed_origins=['https://example.com'])
    backend = SimpleNamespace(needs_unlock=False, get_meta=lambda handle: meta)
    monkeypatch.setattr('agent.vault_backends.backend_for_handle', lambda handle: backend)

    result = json.loads(registry.dispatch('browser_exec', {'session': 'named-insurance', 'code': 'print("public")'}, task_id=task))
    assert result['output'] == 'public\n'
    assert vault._current_page_origin(task) == 'https://example.com'
    result = json.loads(registry.dispatch('browser_vault_fill', {'handle': 'fixture-only'}, task_id=task))
    assert result == {'success': False, 'error': 'No login form fields were found on the current page.'}
    assert attached and set(attached) == {'email-first'}
    supervisor.bind_exec_target('')
    assert vault._current_page_origin(task) is None  # never fall back to another daemon


def test_registered_passwordless_request_arms_before_click(monkeypatch, tmp_path):
    from agent import vault_email_otp as otp

    cfg = {'origin': 'https://example.com', 'identifier': 'fixture@example.com',
           'handle': 'fixture-only', 'state_path': str(tmp_path / 'otp.json')}
    meta = SimpleNamespace(kind='login', origin=cfg['origin'], identifier=cfg['identifier'])
    monkeypatch.setattr('agent.vault_backends.backend_for_handle', lambda handle: SimpleNamespace(get_meta=lambda h: meta))
    monkeypatch.setattr(vault, '_current_page_origin', lambda task: cfg['origin'])
    monkeypatch.setattr(vault, '_ensure_supervisor', lambda task: SimpleNamespace(browser_exec_target_id='email-first'))
    monkeypatch.setattr(otp, 'configured', lambda h, o, i: dict(cfg))
    monkeypatch.setattr(otp, '_gmail', lambda cfg: None)
    monkeypatch.setattr(otp, '_ids', lambda *a: ['baseline-message'])
    dom = {'idStamp': None, 'buttonStamp': None, 'marker': None, 'clicks': 0}

    def evaluate(task, expression):
        if 'button.click()' in expression:
            attempt = json.loads((tmp_path / 'otp.json').read_text())['attempts'][cfg['origin']]
            assert attempt['task_id'] == task and attempt['baseline'] == ['baseline-message']
        script = """
const state = %s;
const location = {origin:'https://example.com'};
const id = {value:'fixture@example.com', type:'email', dataset:{hermesRequest:state.idStamp}, getClientRects:()=>[1]};
const button = {tagName:'BUTTON', dataset:{hermesRequest:state.buttonStamp}, getClientRects:()=>[1], click:()=>state.clicks++};
id.form = button.form = {querySelectorAll:()=>[]};
const document = {querySelectorAll:s=>s==='#email'?[id]:s==='#continue'?[button]:[]};
const sessionStorage = {setItem:(k,v)=>state.marker=v};
const result = %s;
state.idStamp=id.dataset.hermesRequest; state.buttonStamp=button.dataset.hermesRequest;
console.log(JSON.stringify({result,state}));
""" % (json.dumps(dom), expression)
        response = json.loads(subprocess.check_output(['node', '-e', script], text=True))
        dom.update(response['state'])
        return {'success': True, 'result': response['result']}

    monkeypatch.setattr(vault, '_eval_js', evaluate)
    args = {'handle': 'fixture-only', 'action': 'request', 'identifier_selector': '#email', 'request_selector': '#continue'}
    result = json.loads(registry.dispatch('browser_vault_enter_code', args, task_id='request-task'))
    assert result['success'] and result['source'] == 'gmail'
    attempt = json.loads((tmp_path / 'otp.json').read_text())['attempts'][cfg['origin']]
    assert dom['marker'] == attempt['target_nonce'] and dom['clicks'] == 1
    assert not attempt['consumed']
    result = json.loads(registry.dispatch('browser_vault_enter_code', args, task_id='request-task'))
    assert result['error_type'] == 'email_otp_attempt_pending' and dom['clicks'] == 1
