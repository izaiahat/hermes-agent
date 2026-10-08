"""Optional deployment bridge to the workspace's shared Firecrawl ledger.

Other installations without the configured client remain unchanged. The GLP
host installs this path from reviewed source; no credentials are loaded here.
"""
import importlib.util
import json
import os
from pathlib import Path
import uuid

_CLIENT = None


class CompletedEffectAccountingError(RuntimeError):
    """Provider returned; terminal persistence failed. Never reconnect or replay.

    Cause/context may contain transport-looking bookkeeping errors, but they
    describe accounting, not the already completed provider invocation.
    """

    def __init__(self):
        super().__init__('Firecrawl terminal accounting failed; outcome unknown; do not replay')



def _client():
    global _CLIENT
    if _CLIENT is None:
        path = Path(os.environ.get('GLP_FIRECRAWL_CLIENT', '/home/ubuntu/business/scripts/firecrawl_client.py'))
        if not path.is_file():
            return None
        spec = importlib.util.spec_from_file_location('_glp_firecrawl_ledger', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLIENT = module
    return _CLIENT


def _plain(value):
    if hasattr(value, 'model_dump') and not hasattr(value, 'content'):
        return value.model_dump(by_alias=True)
    if isinstance(value, dict):
        return value
    structured = getattr(value, 'structuredContent', None) or getattr(value, 'structured_content', None)
    if isinstance(structured, dict):
        return structured
    for part in getattr(value, 'content', []) or []:
        try:
            payload = json.loads(getattr(part, 'text', ''))
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def begin(caller, endpoint, options):
    client = _client()
    if client is None:
        return None
    operation = str(uuid.uuid4())
    client.record_operation(caller, endpoint, options, outcome='started', operation_id=operation,
                            run_id=os.environ.get('HERMES_SESSION_ID'))
    return operation


def finish(operation, caller, endpoint, options, result=None, failed=False):
    if operation is None:
        return
    payload = _plain(result)
    # SDK scrape returns the document, rather than the REST envelope.
    if 'metadata' in payload and 'data' not in payload:
        payload = {'success': True, 'data': payload}
    outcome = 'ambiguous_transport_error' if failed else 'provider_error' if (payload.get('success') is False or getattr(result, 'isError', False)) else 'success'
    try:
        _client().record_operation(caller, endpoint, options, payload=payload, outcome=outcome,
                                   operation_id=operation, run_id=os.environ.get('HERMES_SESSION_ID'))
    except Exception:
        # Never expose bookkeeping as a reconnectable/auth transport error.
        # The existing started receipt remains uncertain; no operation replay.
        raise CompletedEffectAccountingError() from None


def finish_failed(operation, caller, endpoint, options):
    """Best-effort terminal accounting while an original exception unwinds.

    The durable started receipt already records uncertainty. A bookkeeping
    failure cannot replace cancellation or reclassify the original transport.
    """
    try:
        finish(operation, caller, endpoint, options, failed=True)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            'Firecrawl terminal accounting failed (%s); original failure preserved',
            type(exc).__name__)


def sdk_call(caller, endpoint, method, **options):
    operation = begin(caller, endpoint, options)
    try:
        result = method(**options)
    except BaseException:
        finish_failed(operation, caller, endpoint, options)
        raise
    finish(operation, caller, endpoint, options, result)
    return result
