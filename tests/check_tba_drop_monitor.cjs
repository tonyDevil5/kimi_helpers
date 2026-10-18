const fs=require('fs'),vm=require('vm'),assert=require('assert');
const h=fs.readFileSync('tba_drop_realization_output/tba_drop_realization_monitor.html','utf8');
const payload=h.match(/<script type="application\/json" id="payload">([\s\S]*?)<\/script>/)[1],data=JSON.parse(payload),coupons=Object.keys(data);
const nodes={payload:{textContent:payload},coupon:{value:coupons[0]}};let charts=0;
const context=vm.createContext({document:{getElementById:id=>nodes[id],querySelectorAll:()=>[]},Plotly:{react(id,traces,layout){assert(layout.height>=350);for(const t of traces){assert.equal(t.x.length,t.y.length);assert(t.y.every(v=>v===null||Number.isFinite(v)));}charts++;}}});
vm.runInContext(h.slice(h.lastIndexOf('<script>')+8,h.lastIndexOf('</script>')),context);
assert.equal(coupons.length,6);
for(const c of coupons){nodes.coupon.value=c;vm.runInContext('render()',context);const d=data[c];assert.equal(d.roll_dates.length,22);assert(d.roll_dates.every(date=>date<=d.dates.at(-1)));for(let i=0;i<d.dates.length;i++)if(d.total[i]!==null)assert(Math.abs(d.closed[i]+d.open[i]-d.total[i])<1e-5);for(const date of d.roll_dates){const i=d.dates.indexOf(date);assert(Math.abs(d.open[i])<1e-5);}}
assert.equal(charts,4+6*4);assert.equal((h.match(/class="detail"/g)||[]).length,6);
console.log('PASS: six coupons, four charts each; closed plus open equals total; every new roll starts with zero open P&L');
