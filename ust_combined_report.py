"""Build one portable, interactive Treasury report from the existing monitors.

Run in mercury_env: python ust_combined_report.py
Use --from-existing to package existing HTML without refreshing Bloomberg.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DEFAULT_BACKTEST_ROOT = ROOT.parent / 'BackTestingTool'
DEFAULT_OUTPUT = ROOT / 'ust_combined_output' / 'UST_Combined_Report.html'
PLOTLY_SCRIPT = re.compile(r'<script\b[^>]*>(.*?)</script\s*>', re.I | re.S)
PLOTLY_SLOT = '<!-- UST_COMBINED_PLOTLY_LIBRARY -->'


def latest_report(folder: Path, pattern: str) -> Path:
    """Date-stamped names sort by report date rather than incidental file copies."""
    paths = sorted(folder.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No {pattern} in {folder}. Run without --from-existing first.')
    return paths[-1]


def refresh_reports(backtest_root: Path) -> None:
    jobs = [
        (ROOT, ['ust_fair_value_report.py', '--format', 'interactive']),
        (ROOT, ['bund_fair_value_report.py']),
        (ROOT, ['ust_technical_monitor.py']),
        (ROOT, ['swaption_skew_report.py', '--format', 'interactive']),
        (backtest_root, ['strategy_example/ust10y_cta_comparison.py', '--update']),
    ]
    for cwd, args in jobs:
        print(f'Refreshing {args[0]}...', flush=True)
        subprocess.run([sys.executable, '-u', *args], cwd=cwd, check=True)


def compress_text(value: str) -> str:
    return base64.b64encode(gzip.compress(value.encode('utf-8'), mtime=0)).decode('ascii')


def prepare_documents(sources: list[tuple[str, str, Path]], inline_plotly_js: str | None = None) -> dict:
    libraries: dict[str, str] = {}
    documents = []
    for key, title, path in sources:
        source = path.read_text(encoding='utf-8')
        original_digest = hashlib.sha256(source.encode('utf-8')).hexdigest()
        external = re.compile(r'<script\b[^>]*\bsrc=[\"\x27]https://cdn\.plot\.ly/plotly[^\"\x27]*[\"\x27][^>]*>\s*</script\s*>', re.I)
        inlined = False
        if inline_plotly_js is not None and external.search(source):
            source = external.sub(lambda _: '<script>'+inline_plotly_js+'</script>', source)
            inlined = True
        if not re.search(r'<html\b', source, re.I) or not re.search(r'</html\s*>', source, re.I):
            raise ValueError(f'Incomplete HTML report: {path}')
        library_ids = []

        def extract_library(match):
            script = match.group(1)
            if 'plotly.js v' not in script[:500]:
                return match.group(0)
            digest = hashlib.sha256(match.group(0).encode('utf-8')).hexdigest()
            libraries.setdefault(digest, compress_text(match.group(0)))
            library_ids.append(digest)
            return PLOTLY_SLOT

        document = PLOTLY_SCRIPT.sub(extract_library, source)
        if len(library_ids) != 1:
            raise ValueError(f'Expected one embedded Plotly library in {path}, found {len(library_ids)}')
        dates = re.findall(r'20\d{6}', path.stem)
        source_label = f'Report dated {dates[-1][:4]}-{dates[-1][4:6]}-{dates[-1][6:]}' if dates else title
        # Relative links in the original document retain their original meaning.
        if '<base ' not in document.lower():
            document = re.sub(r'(<head\b[^>]*>)', lambda m: m.group(0) +
                '<base href="' + html.escape(path.resolve().as_uri(), quote=True) + '">', document, count=1, flags=re.I)
        documents.append(dict(id=key, title=title, source=path.name, source_label=source_label,
            library=library_ids[0], data=compress_text(document),
            source_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(),
            original_source_sha256=original_digest, plotly_inlined=inlined))
    return dict(libraries=libraries, documents=documents)


def build_report(sources: list[tuple[str, str, Path]], output: Path = DEFAULT_OUTPUT,
                 title: str = 'Treasury Research Monitor', inline_plotly_js: str | None = None) -> Path:
    payload = prepare_documents(sources, inline_plotly_js=inline_plotly_js)
    payload['title'] = title
    generated = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M China time')
    buttons = ''.join(f'<a class="nav-link" role="tab" id="tab-{d["id"]}" '
        f'aria-controls="panel-{d["id"]}" aria-selected="false" href="#{d["id"]}">'
        f'{html.escape(d["title"])}</a>' for d in payload['documents'])
    panels = ''.join(f'<section class="report-panel" role="tabpanel" id="panel-{d["id"]}" '
        f'aria-labelledby="tab-{d["id"]}" hidden><iframe title="{html.escape(d["title"])}" '
        f'id="frame-{d["id"]}"></iframe></section>' for d in payload['documents'])
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>REPORT_TITLE</title><style>
*{box-sizing:border-box}html,body{height:100%;margin:0}body{font:14px Arial,sans-serif;color:#183247;background:#eef3f7;display:flex;flex-direction:column}
header{flex:none;background:#12344d;color:white;padding:15px 22px 0;box-shadow:0 2px 8px #0002}h1{font-size:22px;margin:0 0 5px}.meta{color:#c7d7e3;font-size:12px;margin:0 0 12px}
nav{display:flex;gap:8px;flex-wrap:wrap}.nav-link{padding:12px 18px;text-decoration:none;color:#e3edf5;border-radius:7px 7px 0 0;font-weight:bold;background:#ffffff13}.nav-link:hover,.nav-link:focus-visible{background:#ffffff30}.nav-link[aria-selected=true]{color:#12344d;background:#fff}
.status-bar{flex:none;display:flex;align-items:center;justify-content:space-between;gap:15px;padding:9px 22px;background:white;border-bottom:1px solid #d6e1e8;color:#526574;font-size:12px}#status{margin:0}.help{white-space:nowrap}main{flex:1;min-height:0}.report-panel{height:100%}.report-panel[hidden]{display:none}iframe{border:0;width:100%;height:100%;display:block;background:white}
@media(max-width:650px){header{padding:12px 10px 0}h1{font-size:18px}.nav-link{padding:10px;font-size:12px}.status-bar{padding:8px 10px}.help{display:none}}
</style></head><body><header><h1>REPORT_TITLE</h1>
<p class="meta">One interactive report · Packaged GENERATED · Source dates shown below; packaging does not refresh market data.</p>
<nav role="tablist" aria-label="Report sections">BUTTONS</nav></header>
<div class="status-bar"><p id="status" role="status" aria-live="polite">Loading report…</p><span class="help">Links can be bookmarked · Chart controls stay within each section</span></div>
<main>PANELS</main><noscript>Enable JavaScript to view the interactive analyses.</noscript>
<script type="application/json" id="report-data">PAYLOAD</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('report-data').textContent);
const LIBRARIES=new Map();
const LOADING=new Map();
let activeId=null;
async function unzip(encoded){
  if(typeof DecompressionStream==='undefined') throw new Error('Please open this report in a current version of Chrome, Edge, Firefox or Safari.');
  const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
  return await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
}
async function loadDocument(doc){
  if(!LIBRARIES.has(doc.library)) LIBRARIES.set(doc.library,unzip(DATA.libraries[doc.library]));
  const [source,library]=await Promise.all([unzip(doc.data),LIBRARIES.get(doc.library)]);
  const content=source.replace('<!-- UST_COMBINED_PLOTLY_LIBRARY -->',()=>library);
  const frame=document.getElementById('frame-'+doc.id);
  await new Promise((resolve,reject)=>{
    frame.onload=resolve;
    frame.onerror=()=>reject(new Error('Could not load '+doc.title));
    frame.srcdoc=content;
  });
}
function showStatus(doc){
  document.getElementById('status').textContent=doc.title+' · '+doc.source_label+' · '+doc.source;
}
async function selectSection(){
  const requested=location.hash.slice(1);
  const doc=DATA.documents.find(d=>d.id===requested)||DATA.documents[0];
  activeId=doc.id;
  document.querySelectorAll('.nav-link').forEach(link=>{
    const selected=link.id==='tab-'+doc.id;
    link.setAttribute('aria-selected',String(selected));link.tabIndex=selected?0:-1;
  });
  DATA.documents.forEach(d=>{document.getElementById('panel-'+d.id).hidden=d.id!==doc.id;});
  document.title=doc.title+' | '+DATA.title;
  document.getElementById('status').textContent='Loading '+doc.title+'…';
  if(!LOADING.has(doc.id)) LOADING.set(doc.id,loadDocument(doc));
  try{
    await LOADING.get(doc.id);
    if(activeId===doc.id){
      showStatus(doc);
      // Resize charts after re-showing a previously hidden section.
      const win=document.getElementById('frame-'+doc.id).contentWindow;
      win.dispatchEvent(new win.Event('resize'));
    }
  }catch(error){
    LOADING.delete(doc.id);
    if(activeId===doc.id)document.getElementById('status').textContent='Unable to load '+doc.title+': '+error.message;
  }
}
document.querySelector('nav').addEventListener('keydown',event=>{
  if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
  event.preventDefault();
  const i=DATA.documents.findIndex(d=>d.id===activeId);
  const n=DATA.documents.length;
  const next=event.key==='Home'?0:event.key==='End'?n-1:(i+(event.key==='ArrowRight'?1:-1)+n)%n;
  const doc=DATA.documents[next];location.hash=doc.id;document.getElementById('tab-'+doc.id).focus();
});
window.addEventListener('hashchange',selectSection);
selectSection();
</script></body></html>'''
    page = page.replace('REPORT_TITLE', html.escape(title)).replace('GENERATED', html.escape(generated)).replace('BUTTONS', buttons).replace('PANELS', panels)
    page = page.replace('PAYLOAD', json.dumps(payload, separators=(',', ':')).replace('</', '<\\/'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
    print(f'Combined report: {output} ({output.stat().st_size / 1024**2:.1f} MB)')
    for _, title, path in sources:
        print(f'  {title}: {path}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-existing', action='store_true', help='Package existing reports without querying Bloomberg')
    parser.add_argument('--backtest-root', type=Path, default=DEFAULT_BACKTEST_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.from_existing:
        refresh_reports(args.backtest_root)
    sources = [
        ('fair-value', 'UST fair value', latest_report(ROOT/'ust_fair_value_output', 'UST_FairValue_Report_Interactive_????????.html')),
        ('bund-fair-value', 'Bund fair value', latest_report(ROOT/'bund_fair_value_output', 'Bund_FairValue_Report_????????.html')),
        ('technical', 'Technical & DM gov bonds', latest_report(ROOT/'ust_technical_output', 'UST_Technical_Monitor_????????.html')),
        ('momentum', 'Momentum & walk-forward', args.backtest_root/'strategy_example'/'ust10y_momentum_output'/'ust10y_cta_strategy_comparison.html'),
        ('swaptions', 'Swaption vol & skew', latest_report(ROOT/'swaption_skew_output', 'Swaption_Skew_Report_Interactive_????????.html')),
    ]
    build_report(sources, args.output)


if __name__ == '__main__':
    main()
