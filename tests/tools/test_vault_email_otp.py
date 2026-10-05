"""Synthetic fixtures only; never read a real mailbox or real vault in tests."""
import base64
import json
from email.message import EmailMessage
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from agent import vault_email_otp as otp


@pytest.fixture
def cfg(tmp_path):
    return {'origin':'https://carrier.example', 'identifier':'agent-one', 'handle':'vault_test',
            'sender':'noreply@carrier.example', 'subject':'Agent Portal Login Code',
            'auth_domain':'carrier.example', 'mailbox':'owner@example.com',
            'state_path':str(tmp_path/'otp.json'), 'token_path':str(tmp_path/'token.json'),
            'body_pattern':r'^Your Verification Code is:\s*(?P<code>[0-9]{6})\s*$',
            'body_recipient':True, 'target_nonce':'synthetic-tab-nonce'}


def message(cfg, **changes):
    m=EmailMessage()
    m['From']=changes.get('sender',cfg['sender'])
    m['To']=changes.get('recipient',cfg['mailbox'])
    m['Subject']=changes.get('subject',cfg['subject'])
    m['Received']='from mail.carrier.example by mx.google.com with ESMTPS id synthetic'
    m['Authentication-Results']=changes.get('auth','mx.google.com; dkim=pass header.i=@carrier.example; dmarc=pass header.from=carrier.example')
    m.set_content(changes.get('body','Your Verification Code is: 123456\nRecipient: owner@example.com\n'))
    return {'id':changes.get('id','new-message'),'internalDate':str(changes.get('received',1001)*1000),
            'raw':base64.urlsafe_b64encode(m.as_bytes()).decode()}


def test_sender_recipient_subject_auth_and_freshness(cfg):
    assert otp.extract_code(message(cfg),cfg,1000,1002)=='123456'
    bad=[{'sender':'spoof@example.com'},{'recipient':'other@example.com'}, {'subject':'Password reset'},
         {'auth':'mx.google.com; dkim=fail header.i=@carrier.example; dmarc=fail header.from=carrier.example'},
         {'auth':'other.example; dkim=pass header.i=@carrier.example; dmarc=pass header.from=carrier.example'},
         {'received':999},{'received':1050},
         {'body':'Your Verification Code is: 123456\nRecipient: other@example.com\n'},
         {'body':'Your Verification Code is: 123456\nYour Verification Code is: 654321\nRecipient: owner@example.com\n'}]
    for change in bad:
        assert otp.extract_code(message(cfg,**change),cfg,1000,1002) is None


def test_explicit_html_template_only(cfg):
    from email import message_from_bytes
    data=message(cfg)
    mail=message_from_bytes(base64.urlsafe_b64decode(data['raw']),policy=__import__('email.policy',fromlist=['default']).default)
    mail.set_content('<p>Security Validation Code:</p><b>123456</b><script>654321</script>',subtype='html')
    data['raw']=base64.urlsafe_b64encode(mail.as_bytes()).decode()
    bound={**cfg,'body_recipient':False,'body_pattern':r'Security Validation Code:\s*(?P<code>[0-9]{6})'}
    assert otp.extract_code(data,bound,1000,1002) is None
    assert otp.extract_code(data,{**bound,'body_mime':'text/html'},1000,1002)=='123456'


def test_enrollment_inspection_returns_predicates_not_body_or_code(cfg):
    data = message(cfg, subject='Fixture code 123456', body='BLC Applications - Prod\n123456\n')
    result = otp.inspect_enrollment_template(data, [r'(?<![0-9])(?P<code>[0-9]{6})(?![0-9])'])
    assert result == {'subject_without_digits': None, 'templates': [
        {'mime': 'text/plain', 'baltimore_named': False, 'matches': [0]}]}
    assert '123456' not in json.dumps(result)
    assert 'BLC Applications' not in json.dumps(result)


def test_binding_requires_exact_handle_origin_and_identifier(cfg):
    section={'vault':{'email_otp':{**cfg,'bindings':{'vault_test':cfg}}}}
    with patch('hermes_cli.config.load_config_readonly',return_value=section):
        assert otp.configured('vault_test',cfg['origin'],cfg['identifier'])
        assert otp.configured('unconfigured',cfg['origin'],cfg['identifier']) is None
        with pytest.raises(otp.EmailOTPError,match='binding_mismatch'):
            otp.configured('vault_test','https://evil.example',cfg['identifier'])
        with pytest.raises(otp.EmailOTPError,match='binding_mismatch'):
            otp.configured('vault_test',cfg['origin'],'other-account')


def arm(cfg):
    with patch.object(otp,'_gmail'),patch.object(otp,'_ids',return_value=['old-message']),patch.object(otp.time,'time',return_value=1000):
        otp.arm(cfg,'session')


def test_single_attempt_consumption_is_persistent_and_secret_free(cfg):
    arm(cfg)
    api=MagicMock()
    api.users.return_value.messages.return_value.get.return_value.execute.return_value=message(cfg)
    with patch.object(otp,'_gmail',return_value=api),patch.object(otp,'_ids',return_value=['old-message','new-message']),patch.object(otp.time,'time',return_value=1002):
        assert otp.retrieve(cfg,'session')=='123456'
        with pytest.raises(otp.EmailOTPError,match='expired_or_used'):
            otp.retrieve(cfg,'session')
    data=open(cfg['state_path']).read()
    assert '123456' not in data and 'Recipient:' not in data and 'new-message' in data


def test_concurrent_accounts_and_ambiguous_emails_are_refused(cfg):
    arm(cfg)
    with patch.object(otp,'_gmail'),patch.object(otp,'_ids',return_value=[]),patch.object(otp.time,'time',return_value=1002):
        with pytest.raises(otp.EmailOTPError,match='attempt_pending'):
            otp.arm({**cfg,'handle':'another'},'other-session')
    api=MagicMock()
    api.users.return_value.messages.return_value.get.return_value.execute.side_effect=[message(cfg),message(cfg,id='second')]
    with patch.object(otp,'_gmail',return_value=api),patch.object(otp,'_ids',return_value=['new-message','second']),patch.object(otp.time,'time',return_value=1002):
        with pytest.raises(otp.EmailOTPError,match='ambiguous'):
            otp.retrieve(cfg,'session')
        with pytest.raises(otp.EmailOTPError,match='no_matching_attempt'):
            otp.retrieve(cfg,'other-session')


def test_tool_routes_email_only_to_secret_fill(cfg):
    from tools import browser_vault_tool as tool
    backend=MagicMock()
    backend.get_meta.return_value=SimpleNamespace(kind='login',origin=cfg['origin'],identifier=cfg['identifier'])
    controls=[{'index':0,'form_index':0,'type':'text','name':'code','id':'otp','autocomplete':'one-time-code','token':'test','visible':True,'disabled':False,'read_only':False}]
    with patch.object(tool,'_focus_bound_origin'),patch.object(tool,'_current_page_origin',return_value=cfg['origin']),patch.object(tool,'_eval_js',return_value={'success':True,'result':controls}),patch('agent.vault_backends.backend_for_handle',return_value=backend),patch.object(otp,'configured',return_value=cfg),patch.object(otp,'retrieve',return_value='123456'),patch.object(tool,'_eval_js_secret',return_value={'success':True,'result':{'filled':1}}) as secret:
        result=tool.browser_vault_enter_code('vault_test')
    assert json.loads(result)['source']=='gmail'
    assert '123456' not in result
    assert secret.call_count==1


@pytest.mark.parametrize('takeover', [False, True], ids=['agent-control', 'takeover-during-arm'])
def test_registered_request_rechecks_lease_after_mailbox_baseline(cfg, monkeypatch, tmp_path, takeover):
    from pathlib import Path
    import socket
    import subprocess
    from tools import browser_tool, browser_supervisor, browser_vault_tool as tool
    from tools.bot_desktop import lease, runtime
    from tools.registry import registry

    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes'))
    def denied(*args, **kwargs):
        raise AssertionError('This regression forbids network and browser subprocesses')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(subprocess, 'Popen', denied)
    monkeypatch.setattr(runtime, 'published_env', lambda: {'DISPLAY': ':fixture'})
    monkeypatch.setattr(browser_tool, '_last_session_key', lambda task: task)
    monkeypatch.setattr(browser_tool, '_active_sessions', {'request-task': {'features': {'local': True}}})
    meta = SimpleNamespace(kind='login', origin=cfg['origin'], identifier=cfg['identifier'])
    monkeypatch.setattr('agent.vault_backends.backend_for_handle', lambda handle: SimpleNamespace(get_meta=lambda h: meta))
    monkeypatch.setattr(tool, '_current_page_origin', lambda task: cfg['origin'])
    monkeypatch.setattr(otp, 'configured', lambda *args: dict(cfg))
    monkeypatch.setattr(otp, '_gmail', lambda config: None)
    baselines = []
    def baseline(*args):
        baselines.append(True)
        if takeover:
            lease.acquire('fixture-human')
        return ['baseline-message']
    monkeypatch.setattr(otp, '_ids', baseline)
    actions = []
    class Supervisor:
        browser_exec_target_id = 'fixture-target'
        def evaluate_runtime(self, expression):
            click = 'button.click()' in expression
            if click:
                assert Path(cfg['state_path']).exists()  # Real arm committed before dispatch.
            actions.append({'click': click, 'human_holds': lease.human_holds()})
            return {'ok': True, 'result': True}
    supervisor = Supervisor()
    monkeypatch.setattr(browser_supervisor.SUPERVISOR_REGISTRY, 'get', lambda task: supervisor)
    args = {'handle': cfg['handle'], 'action': 'request',
            'request_selector': '#request', 'identifier_selector': '#identifier'}
    result = json.loads(registry.dispatch('browser_vault_enter_code', args, task_id='request-task'))
    assert baselines == [True]
    state_path = Path(cfg['state_path'])
    armed = state_path.read_bytes()
    attempt = json.loads(armed)['attempts'][cfg['origin']]
    assert attempt['handle'] == cfg['handle'] and attempt['task_id'] == 'request-task'
    assert attempt['baseline'] == ['baseline-message'] and attempt['target_nonce']
    assert attempt['consumed'] is False
    assert actions == ([{'click': False, 'human_holds': False}] if takeover else
                       [{'click': False, 'human_holds': False}, {'click': True, 'human_holds': False}])
    if takeover:
        assert result['code'] == 'human_has_control' and lease.human_holds()
    else:
        assert result['success'] is True and result['source'] == 'gmail'
    # A refused write must not erase/re-arm the pending attempt or allow replay.
    with pytest.raises(otp.EmailOTPError, match='email_otp_attempt_pending'):
        otp.arm({**cfg, 'target_nonce': 'replacement-nonce'}, 'request-task')
    assert state_path.read_bytes() == armed
