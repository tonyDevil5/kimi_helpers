// Offline application-logic check. No browser, network, or UI automation.
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const html = fs.readFileSync(process.argv[2]||'adhoc_output/adhoc_cmo_caps.html','utf8');
const data = html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1];
const comparison = html.match(/<script type="application\/json" id="comparison">([\s\S]*?)<\/script>/)[1];
const script = html.slice(html.lastIndexOf('<script>')+8, html.lastIndexOf('</script>'));
const nodes = {};
function node(id){if(!nodes[id])nodes[id]={value:({cap:'all',group:'all','shock-group':'all','watch-model':'full',model:'full',view:'price','scenario-model':'full','shock-cap':'6','shock-rate':'20','shock-cc':'10','perf-cap':'6','perf-group':'all','perf-view':'bar','perf-unit':'dollars','perf-start':'2026-08-01','perf-end':'2026-10-02'})[id]||'',textContent:id==='data'?data:id==='comparison'?comparison:'',addEventListener(){},set innerHTML(text){this.html=text;if(id==='ticker'){this.value=text.match(/value="([^"]+)"/)?.[1]||''}},get innerHTML(){return this.html}};return nodes[id]}
let plots=0,lastPlot;
let perfPlot;
const context = vm.createContext({document:{getElementById:node,querySelectorAll:()=>[]},Plotly:{react(id,traces,layout){if(id==='forward-chart'){assert(traces[0].x.length>0);assert.equal(traces[0].x.length,traces[0].y.length);return;}if(id==='perf-chart'){perfPlot={traces,layout};return;}assert.equal(id,'chart');assert(layout.height>=470);assert(Array.isArray(traces));lastPlot={traces,layout};plots++}},console});
vm.runInContext(script,context);
const perf=JSON.parse(comparison);
assert.deepEqual(perf.bonds.map(b=>b.ticker).sort(),JSON.parse(data).map(b=>b.ticker).sort());
for(const [start,end] of [['2026-08-01','2026-10-02'],['2026-09-01','2026-10-02'],['2026-08-15','2026-09-15']])for(const cap of ['6','6.5','7','7.5','all'])for(const view of ['bar','line'])for(const unit of ['dollars']){
 node('perf-start').value=start;node('perf-end').value=end;
 node('perf-cap').value=cap;node('perf-view').value=view;node('perf-unit').value=unit;vm.runInContext('performance()',context);
 const window=vm.runInContext('comparisonWindow($("perf-start").value,$("perf-end").value)',context);
 assert(!window.error);assert(window.start>=start&&window.end<=end);
 const expected=window.bonds.filter(b=>(cap==='all'||b.cap===Number(cap))&&b.change!==null);
 const scale=1;
 assert(!node('perf-tables').innerHTML.includes('NaN'));
 if(view==='bar'){assert.equal(perfPlot.traces[0].x.length,expected.length+4);assert(Math.abs(perfPlot.traces[0].x[0]-window.benchmarks[0].change*scale)<1e-10);}
 else{assert.equal(perfPlot.traces.length,expected.length+4);for(const trace of perfPlot.traces)assert.equal(trace.y[0],0);}
}
const bonds = JSON.parse(data);
assert(vm.runInContext('comparisonWindow("2026-10-02","2026-08-01").error',context));
assert(vm.runInContext('comparisonWindow("2030-01-01","2030-02-01").error',context));
const aug=vm.runInContext('comparisonWindow("2026-08-01","2026-10-02")',context);assert.equal(aug.start,'2026-08-03');
for(const b of aug.bonds){const i=perf.dates.indexOf(aug.start),j=perf.dates.indexOf(aug.end),raw=perf.bonds.find(k=>k.ticker===b.ticker);if(b.change!==null)assert(Math.abs(b.change-(raw.prices[j]-raw.prices[i]))<1e-10);}
for(const b of bonds)for(const model of ['full','rolling'])for(const view of ['price','residual','factors','scatter','validation']){
  node('ticker').value=b.ticker;node('model').value=model;node('view').value=view;
  vm.runInContext('render()',context);
  assert(!node('cards').innerHTML.includes('NaN'));
  if(view==='scatter'&&b.models[model]){
    const latest=lastPlot.traces.find(t=>t.name==='Latest observation');
    assert(latest);
    assert.equal(latest.x[0],b.models[model].fit.at(-1));
    assert.equal(latest.y[0],b.series.px.at(-1));
    assert(latest.text[0].includes(b.dates.at(-1)));
  }else if(view!=='scatter'){
    for(const marker of lastPlot.traces.filter(t=>t.name.startsWith('Latest '))){
      const original=lastPlot.traces.find(t=>t.name===marker.name.slice(7));
      let i=original.y.length-1;while(i>=0&&original.y[i]===null)i--;
      assert.equal(marker.y[0],original.y[i]);assert.equal(marker.x[0],original.x[i]);
    }
  }
}
for(const cap of ['6','6.5','7']){const prior=node('ticker').value;node('cap').value=cap;vm.runInContext('table()',context);assert.equal(node('ticker').value,prior);const shown=[...node('watchlist').innerHTML.matchAll(/data-ticker="([^"]+)"/g)].map(m=>m[1]);assert.equal(shown.length,bonds.filter(b=>b.cap===Number(cap)).length);assert(shown.every(t=>bonds.find(b=>b.ticker===t).cap===Number(cap)));}
assert.equal(plots,bonds.length*10+1);
assert(html.includes('<th>FLT_SPREAD</th>'));
assert(html.includes('<th>MTG_COLLAT_TYP</th>'));
assert(html.includes('<th>Underlying cpn (%)</th>'));
assert(!html.includes('id="shock-scenario"'));
for(const model of ['full','rolling'])for(const cap of ['6','6.5','7','7.5'])for(const group of ['all','portfolio','analysis'])for(const [rate,cc] of [[20,10],[-20,-10],[-20,10],[20,-10],[0,0],[0,-10],[-7.5,2.5],[7.5,-2.5]]){
  node('scenario-model').value=model;node('shock-cap').value=cap;node('shock-group').value=group;
  node('shock-rate').value=String(rate);node('shock-cc').value=String(cc);
  vm.runInContext('shocks()',context);
  const panel=node('scenario-tables').innerHTML;
  const shown=[...panel.matchAll(/data-shock-bond="([^"]+)"/g)].map(m=>m[1]);
  const expected=bonds.filter(b=>b.cap===Number(cap)&&(group==='all'||b.group===group));
  assert.equal(shown.length,expected.length);
  assert(shown.every(t=>expected.some(b=>b.ticker===t)));
  assert.equal((panel.match(/<h4>Cap /g)||[]).length,1);
  for(const b of expected){
    node('ticker').value=b.ticker;
    const impact=vm.runInContext('scenarioImpact(selected(),selectedScenario())',context);
    if(!b.models[model]){assert.equal(impact,null);continue;}
    assert(Math.abs(impact.parts.CT10-b.models[model].b.CT10*rate/100)<1e-10);
    assert(Math.abs(impact.parts.cc-b.models[model].b.cc*cc)<1e-10);
    assert.equal(impact.parts['2s10s'],0);assert.equal(impact.parts.vol,0);
    assert(Math.abs((impact.shockedPrice-impact.shockedFair)-(impact.price-impact.fair))<1e-10);
  }
  node('cap').value='6';vm.runInContext('table();render()',context);
  assert.equal(node('scenario-tables').innerHTML,panel);
}
for(const invalid of ['', 'bad', 'Infinity']){
  node('shock-rate').value=invalid;vm.runInContext('shocks()',context);
  assert(node('scenario-tables').innerHTML.includes('valid numeric'));
}
console.log('Passed: signed and mixed shocks, bp conversion, cap/group filters, fractional/zero inputs and independent plot controls.');
