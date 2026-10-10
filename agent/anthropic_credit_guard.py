"""Default-profile Claude API reserve guard, separate from included OAuth quota.

A live tracker read is required for new paid-API children. Unknown fails closed.
Other profiles remain untouched; no purchases, fallback or credential changes.
"""
import json
import math
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse
from hermes_constants import get_hermes_home

FLOOR_USD = 5.0


def credit_snapshot():
    tracker = get_hermes_home() / 'scripts/anthropic_api_credit_tracker.py'
    try:
        proc = subprocess.run([sys.executable, str(tracker)], capture_output=True, text=True, timeout=45, stdin=subprocess.DEVNULL)
        if proc.returncode:
            raise ValueError('tracker_failed')
        raw = json.loads(proc.stdout)
        remaining = float(raw['remaining_usd'])
        if not math.isfinite(remaining):
            raise ValueError('tracker_nonfinite')
        return {'remaining_usd': remaining, 'spent_usd': raw.get('spent_usd'),
                'grant_usd': raw.get('grant_usd'), 'expires': raw.get('expires'),
                'source': raw.get('source'), 'balance_kind': 'estimated grant remaining',
                'dispatch_allowed': remaining >= FLOOR_USD, 'floor_usd': FLOOR_USD}
    except Exception:
        return {'remaining_usd': None, 'dispatch_allowed': False, 'floor_usd': FLOOR_USD,
                'source': 'tracker_unavailable'}


def enforce_anthropic_credit_floor(*, provider=None, base_url=None, api_key=None, tracker=None, enabled=None):
    if enabled is None:
        enabled = get_hermes_home().resolve() == (Path.home() / '.hermes').resolve()
    if not enabled:
        return
    # Do not confuse the Claude Max OAuth pool with this API grant.
    if str(api_key or '').startswith('sk-ant-oat'):
        return
    route = str(provider or '').lower()
    if route != 'custom:anthropic-api-credits' and urlparse(base_url or '').hostname != 'api.anthropic.com':
        return
    data = (tracker or credit_snapshot)()
    value = data.get('remaining_usd')
    try:
        remaining = float(value)
    except (ValueError, TypeError):
        raise ValueError('Claude API dispatch held: balance UNKNOWN; no child admitted.') from None
    if not math.isfinite(remaining) or remaining < FLOOR_USD:
        raise ValueError('Claude API dispatch held: balance below $5 reserve; no child admitted.')
    return data
