const fs=require('fs'),vm=require('vm'),assert=require('assert');
const h=fs.readFileSync(process.argv[2]||'fund_sensitivity_output/fund_sensitivity_monitor.html','utf8');
const payload=h.match(/<script type="application\/json" id="payload">([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(payload),funds=Object.keys(data);
const tableHtml=h.match(/<div id="fund-table">([\s\S]*?)<\/div>/)[1];
const clean=s=>s.replace(/<[^>]*>/g,'').trim();
const heads=[...tableHtml.matchAll(/<th>([\s\S]*?)<\/th>/g)].map(m=>({textContent:clean(m[1]),attrs:{},setAttribute(k,v){this.attrs[k]=v},getAttribute(k){return this.attrs[k]}}));
const rows=[...tableHtml.match(/<tbody>([\s\S]*?)<\/tbody>/)[1].matchAll(/<tr>([\s\S]*?)<\/tr>/g)].map(m=>{
 const cells=[...m[1].matchAll(/<td>([\s\S]*?)<\/td>/g)].map(v=>({textContent:clean(v[1])}));
 return {cells,textContent:cells.map(c=>c.textContent).join(' '),style:{}};
});
const tbody={rows:rows.slice(),appendChild(r){this.rows=this.rows.filter(x=>x!==r);this.rows.push(r)}};
const table={tBodies:[tbody],querySelectorAll:()=>heads};
const ytd=heads.findIndex(x=>x.textContent==='YTD USD return (%)');assert(ytd>=0);
const nodes={payload:{textContent:payload},fund:{value:funds[0]},factor:{value:'CT10'},search:{value:''},
 'fund-table':{querySelector:()=>table},'sort-column':{value:String(ytd)},'sort-order':{value:'desc'},
 'fund-group':{value:'All'},'reset-table':{},'row-count':{}};
let plots=0;
const context=vm.createContext({document:{getElementById:id=>nodes[id],querySelectorAll:q=>q==='table'?[table]:[]},
 Plotly:{react(id,traces,layout){assert(layout.height>=400);for(const t of traces){assert.equal(t.x.length,t.y.length);assert(t.y.every(Number.isFinite));}plots++;}}});
vm.runInContext(h.slice(h.lastIndexOf('<script>')+8,h.lastIndexOf('</script>')),context);
for(const fund of funds)for(const factor of ['CT10','FN basis','IG OAS','2s10s','1y10y vol']){
 nodes.fund.value=fund;nodes.factor.value=factor;vm.runInContext('render()',context);
}
assert.equal(plots,3+funds.length*5*3);
function checkSorted(asc){
 let previous=asc?-Infinity:Infinity,missing=false;
 for(const r of tbody.rows){const text=r.cells[ytd].textContent;if(text==='—'){missing=true;continue;}
  assert(!missing);const n=Number(text.replace(/,/g,''));assert(Number.isFinite(n));assert(asc?n>=previous:n<=previous);previous=n;
 }
}
checkSorted(false);nodes['sort-order'].value='asc';nodes['sort-order'].onchange();checkSorted(true);
heads[ytd].onclick();checkSorted(false);assert.equal(heads[ytd].getAttribute('aria-sort'),'descending');
nodes['fund-group'].value='Active';nodes['fund-group'].onchange();
assert.equal(tbody.rows.filter(r=>r.style.display!=='none').length,30);
nodes.search.value='PONAX';nodes.search.oninput();assert.equal(tbody.rows.filter(r=>r.style.display!=='none').length,1);
nodes['reset-table'].onclick();assert.equal(tbody.rows.filter(r=>r.style.display!=='none').length,60);
assert(tbody.rows.every((r,i)=>r===rows[i]));
console.log(`PASS: ${funds.length} fund selectors × five factors; fitted, scatter and rolling plots have valid data`);
console.log('PASS: numeric YTD sorting both directions, missing values last, header toggle, group/search filters and reset');
