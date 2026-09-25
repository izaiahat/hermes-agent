"""Model-blind Gmail OTP source for explicitly configured vault logins.

Called only inside browser_vault_fill/enter_code. No message body or code is
persisted or returned by the tool. Gmail access is read-only; sender, mailbox,
subject, origin, handle and a fresh password-fill attempt must all agree.
"""
from __future__ import annotations

import base64
import fcntl
import json
import os
import re
import tempfile
import time
from contextlib import contextmanager
from email import message_from_bytes
from email.policy import default
from email.utils import getaddresses, parseaddr
from pathlib import Path
from html.parser import HTMLParser


class _TemplateText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.ignored += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.ignored = max(0, self.ignored - 1)

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


class EmailOTPError(Exception):
    """Only fixed, non-secret reason codes may cross the tool boundary."""


def configured(handle, origin, identifier):
    from hermes_cli.config import load_config_readonly
    section = (load_config_readonly().get('vault') or {}).get('email_otp') or {}
    binding = (section.get('bindings') or {}).get(handle)
    if not binding:
        return None
    if binding.get('origin') != origin or binding.get('identifier') != identifier:
        raise EmailOTPError('email_otp_binding_mismatch')
    required = ('mailbox', 'token_path', 'state_path')
    if any(not section.get(k) for k in required):
        raise EmailOTPError('email_otp_config_invalid')
    if any(not binding.get(k) for k in ('sender', 'subject', 'body_pattern', 'auth_domain')):
        raise EmailOTPError('email_otp_config_invalid')
    return {**section, **binding, 'handle': handle, 'origin': origin}


def _gmail(cfg):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from google_auth_httplib2 import AuthorizedHttp
    import httplib2
    token = Path(cfg['token_path']).expanduser()
    if token.stat().st_mode & 0o077:
        raise EmailOTPError('email_otp_token_permissions')
    creds = Credentials.from_authorized_user_file(str(token))
    api = build('gmail', 'v1', http=AuthorizedHttp(creds, http=httplib2.Http(timeout=15)), cache_discovery=False)
    owner = api.users().getProfile(userId='me').execute()['emailAddress']
    if owner.lower() != cfg['mailbox'].lower():
        raise EmailOTPError('email_otp_mailbox_mismatch')
    return api


@contextmanager
def _state(cfg):
    path = Path(cfg['state_path']).expanduser()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(str(path) + '.lock', os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {'attempts': {}, 'used': []}
        yield data
        outfd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.otp-state-')
        try:
            with os.fdopen(outfd, 'w') as out:
                json.dump(data, out)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


def _ids(api, cfg, after):
    # Include trash because the operator's genuine MFA messages are currently
    # there. Do not untrash, mark read, alter filters or fetch unrelated mail.
    sender = cfg['sender']
    if not re.fullmatch(r'[A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+', sender):
        raise EmailOTPError('email_otp_config_invalid')
    q = f'in:anywhere from:{sender} after:{int(after)}'
    res = api.users().messages().list(userId='me', q=q, includeSpamTrash=True, maxResults=25).execute()
    if res.get('nextPageToken'):
        raise EmailOTPError('email_otp_too_many_candidates')
    return [m['id'] for m in res.get('messages', [])]


def arm(cfg, task_id):
    """Snapshot before password submission; only subsequently received mail qualifies."""
    api = _gmail(cfg)
    now = time.time()
    previous_ids = _ids(api, cfg, now - 600)
    with _state(cfg) as state:
        # One pending challenge per origin: same-origin carrier brands/accounts
        # cannot race and take each other's shared-sender code.
        prior = state['attempts'].get(cfg['origin'])
        if prior and now - prior['started'] < 600 and not prior.get('consumed'):
            raise EmailOTPError('email_otp_attempt_pending')
        state['attempts'][cfg['origin']] = {
            'handle': cfg['handle'], 'task_id': task_id, 'started': now,
            'target_nonce': cfg['target_nonce'],
            'baseline': previous_ids, 'consumed': False,
        }


def extract_code(message, cfg, started, now):
    """Validate a Gmail raw record and parse only the configured MFA template."""
    received = int(message.get('internalDate', '0')) / 1000
    if received < started or received > now + 15 or now - received > 600:
        return None
    raw = base64.urlsafe_b64decode(message['raw'])
    if len(raw) > 128_000:
        return None
    mail = message_from_bytes(raw, policy=default)
    if len(mail.get_all('From', [])) != 1 or parseaddr(str(mail['From']))[1].lower() != cfg['sender'].lower():
        return None
    if len(mail.get_all('Subject', [])) != 1 or str(mail['Subject']).strip() != cfg['subject']:
        return None
    recipients = [a.lower() for _, a in getaddresses([str(v) for v in mail.get_all('To', [])])]
    if recipients != [cfg['mailbox'].lower()]:
        return None
    # Trust the first Google-added Authentication-Results header, never a
    # matching injected/forwarded header deeper in the message.
    headers = list(mail.raw_items())
    authentication = [(i, str(v)) for i, (k, v) in enumerate(headers)
                      if k.lower() == 'authentication-results']
    if len(authentication) != 1:
        return None
    index, auth = authentication[0]
    if not auth.lstrip().startswith('mx.google.com;'):
        return None
    received_headers = [str(v) for k, v in headers[:index] if k.lower() == 'received']
    if not received_headers or not re.search(r'\bby\s+mx\.google\.com\s+with\s+E?SMTPS?\b', received_headers[-1], re.I):
        return None
    domain = re.escape(cfg['auth_domain'])
    if not re.search(r'dmarc=pass\b[^;]*\bheader\.from=' + domain + r'(?=[;\s]|$)', auth):
        return None
    if not (re.search(r'dkim=pass\b[^;]*\bheader\.i=@' + domain + r'(?=[;\s]|$)', auth)
            or re.search(r'spf=pass\b[^;]*\bsmtp\.mailfrom=[^;\s@]*@' + domain + r'(?=[;\s]|$)', auth)):
        return None
    mime = cfg.get('body_mime', 'text/plain')
    if mime not in ('text/plain', 'text/html'):
        return None
    parts = [p.get_content() for p in mail.walk()
             if p.get_content_type() == mime and p.get_content_disposition() != 'attachment']
    if len(parts) != 1:
        return None
    text = parts[0]
    if mime == 'text/html':
        parser = _TemplateText()
        parser.feed(text)
        text = '\n'.join(parser.parts)
    matches = list(re.finditer(cfg['body_pattern'], text, re.IGNORECASE | re.MULTILINE))
    if len(matches) != 1 or 'code' not in matches[0].groupdict():
        return None
    code = matches[0].group('code')
    if not re.fullmatch(r'[0-9]{6}', code):
        return None
    # Some providers put the recipient in the body as well. If required, it
    # must be exact; never infer customer-account identity from sender alone.
    if cfg.get('body_recipient'):
        found = re.findall(r'^Recipient:\s*([^\s]+)\s*$', text, re.IGNORECASE | re.MULTILINE)
        if [a.lower() for a in found] != [cfg['mailbox'].lower()]:
            return None
    return code


def inspect_enrollment_template(message, patterns):
    """Inspect a candidate MIME template without returning body or code values."""
    raw = base64.urlsafe_b64decode(message['raw'])
    if len(raw) > 128_000:
        raise EmailOTPError('email_otp_message_too_large')
    mail = message_from_bytes(raw, policy=default)
    subject = str(mail.get('Subject', '')).strip()
    result = {'subject_without_digits': subject if not re.search(r'\d', subject) else None,
              'templates': []}
    for mime in ('text/plain', 'text/html'):
        parts = [p.get_content() for p in mail.walk()
                 if p.get_content_type() == mime and p.get_content_disposition() != 'attachment']
        if len(parts) != 1:
            continue
        text = parts[0]
        if mime == 'text/html':
            parser = _TemplateText()
            parser.feed(text)
            text = '\n'.join(parser.parts)
        result['templates'].append({
            'mime': mime,
            'baltimore_named': bool(re.search(r'Baltimore|baltlife', text, re.I)),
            'matches': [i for i, pattern in enumerate(patterns)
                        if len(list(re.finditer(pattern, text, re.I | re.M))) == 1],
        })
    return result


def retrieve(cfg, task_id):
    """Return a code to the caller inside the vault boundary, or a safe reason."""
    api = None
    deadline = time.monotonic() + 45
    while True:
        with _state(cfg) as state:
            attempt = state['attempts'].get(cfg['origin'])
            if not attempt or attempt['handle'] != cfg['handle'] or attempt['task_id'] != task_id:
                raise EmailOTPError('email_otp_no_matching_attempt')
            if not cfg.get('target_nonce') or attempt.get('target_nonce') != cfg['target_nonce']:
                raise EmailOTPError('email_otp_target_mismatch')
            if attempt.get('consumed') or time.time() - attempt['started'] > 600:
                raise EmailOTPError('email_otp_attempt_expired_or_used')
            if api is None:
                api = _gmail(cfg)
            candidates = []
            for mid in _ids(api, cfg, attempt['started']):
                if mid in attempt['baseline'] or mid in state['used']:
                    continue
                message = api.users().messages().get(userId='me', id=mid, format='raw').execute()
                code = extract_code(message, cfg, attempt['started'], time.time())
                if code:
                    candidates.append((mid, code))
            if len(candidates) > 1:
                raise EmailOTPError('email_otp_ambiguous')
            if candidates:
                mid, code = candidates[0]
                # Reserve before browser fill: an ambiguous transport failure
                # must never replay the same credential automatically.
                state['used'].append(mid)
                state['attempts'][cfg['origin']]['consumed'] = True
                return code
        if time.monotonic() >= deadline:
            raise EmailOTPError('email_otp_not_received')
        time.sleep(3)
