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
