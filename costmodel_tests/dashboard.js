'use strict';
const data=JSON.parse(document.getElementById('dashboard-data').textContent), reports=data.reports, labels=data.labels;
const $=id=>document.getElementById(id), node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
let selected=0,sortDirection=null,jsonCaseName='',activeMode='auto',trendCaseName='';
const modeLabels=data.mode_labels,modeNames=['auto','simd','simt_only','mixed'];
const modeColors={auto:'#244ee5',simd:'#19856a',simt_only:'#8a54bb',mixed:'#c17a15'};
const num=(v,d=3)=>v===null||v===undefined?'—':Number(v).toFixed(d);
function badge(status){return node('span',labels[status]||status,'badge '+status)}
function artifactLink(run,path,text){const a=node('a',text);a.href=run.base+path;return a}
function modeRow(row,mode){
 const matches=(row.mode_results||[]).filter(r=>r.mode===mode).slice().sort((a,b)=>Number(['NOT_RUN','NOT_APPLICABLE'].includes(a.status))-Number(['NOT_RUN','NOT_APPLICABLE'].includes(b.status))||Number(a.factor!==1)-Number(b.factor!==1)||a.factor-b.factor);
 return matches.length?{...row,...matches[0]}:{...row,status:'NOT_RUN',precision:null,predicted_us:null,actual_us:null,ape_percent:null,actual_route:null,actual_factor:null};
}
function toggleSort(){
 sortDirection=sortDirection==='desc'?'asc':'desc';
 $('sort').textContent=sortDirection==='desc'?'按 APE 升序':'按 APE 降序';
 $('apeSort').textContent=sortDirection==='desc'?'APE ↓':'APE ↑';
 $('apeHeader').setAttribute('aria-sort',sortDirection==='desc'?'descending':'ascending');updateTable();
}
function showModes(){
 const run=reports[selected];$('modeOverview').replaceChildren();
 for(const mode of modeNames){
  const summary=(run.mode_summary||{})[mode]||{},button=node('button',undefined,'mode-card'+(mode===activeMode?' active':''));
  button.id='mode_'+mode;button.type='button';button.setAttribute('aria-pressed',String(mode===activeMode));
  button.append(node('strong',modeLabels[mode]),node('div',summary.mape_percent==null?'—':num(summary.mape_percent,2)+'%','mode-value'),node('small','MAPE · '+(summary.valid_count||0)+' / '+run.total+' 个有效实测'));
  button.addEventListener('click',()=>{activeMode=mode;showModes();updateTable()});$('modeOverview').append(button);
 }
 const c=(run.mode_summary?.[activeMode]?.counts)||{};
 $('passed').textContent=(c.PASS||0)+' / '+run.total;$('precision').textContent=c.PRECISION_FAIL||0;$('timing').textContent=c.TIME_FAIL||0;
 $('abnormal').textContent=(c.EXECUTION_FAIL||0)+(c.BLOCKED||0)+(c.UNVERIFIED||0);$('pending').textContent=(c.NOT_RUN||0)+(c.NOT_APPLICABLE||0);
}
function drawTrend(){
 const run=reports[selected],days=(data.trends||{})[run.id]||[],which=$('trendCase').value;
 const series=Object.fromEntries(modeNames.map(mode=>[mode,days.map(day=>which?(day.cases?.[which]?.[mode]??null):(day.modes?.[mode]?.mape_percent??null))]));
 const values=Object.values(series).flat().filter(v=>typeof v==='number'&&Number.isFinite(v));
 $('trendEmpty').classList.toggle('hidden',values.length>0);$('trendChart').classList.toggle('hidden',!values.length);
 $('trendNote').textContent='截至所选批次日期 · 每天取同用例清单、同并发数的最新已完成批次 · '+(which?'展示单用例 APE':'MAPE 为有效用例 APE 的算术平均')+' · 仅统计精度通过的 Profiler 实测；悬停数据点查看覆盖数，缺失日期留空。';
 const svg=$('trendChart');svg.replaceChildren();$('trendLegend').replaceChildren();
 for(const mode of modeNames){const label=node('span');const dot=node('i');dot.style.background=modeColors[mode];label.append(dot,node('span',modeLabels[mode]));$('trendLegend').append(label)}
 if(!values.length)return;
 const ns='http://www.w3.org/2000/svg',make=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs||{}))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n};
 const left=60,right=970,top=20,bottom=220,max=Math.max(15,...values)*1.12,x=i=>left+(right-left)*i/9,y=v=>bottom-(bottom-top)*v/max;
 for(let i=0;i<=4;i++){const v=max*i/4;svg.append(make('line',{x1:left,x2:right,y1:y(v),y2:y(v),stroke:'#e7edf2'}),make('text',{x:48,y:y(v)+4,'text-anchor':'end',fill:'#68798a','font-size':11},num(v,1)+'%'))}
 svg.append(make('line',{x1:left,x2:right,y1:y(10),y2:y(10),stroke:'#b83748','stroke-dasharray':'5 5'}),make('text',{x:right,y:y(10)-7,'text-anchor':'end',fill:'#b83748','font-size':11},'10%'));
 days.forEach((day,i)=>svg.append(make('text',{x:x(i),y:246,'text-anchor':'middle',fill:'#68798a','font-size':11},day.date.slice(5))));
 for(const mode of modeNames){
  let path='',previous=false;
  series[mode].forEach((v,i)=>{if(v===null){previous=false;return}path+=(previous?' L ':' M ')+x(i)+' '+y(v);previous=true});
  svg.append(make('path',{d:path,fill:'none',stroke:modeColors[mode],'stroke-width':2.4}));
  series[mode].forEach((v,i)=>{if(v===null)return;const point=make('circle',{cx:x(i),cy:y(v),r:4.5,fill:modeColors[mode],stroke:'#fff','stroke-width':1.5,tabindex:0,class:'trend-point'}),coverage=which?'单用例':(days[i].modes?.[mode]?.valid_count||0)+' / '+run.total+' 个有效实测';point.append(make('title',{},days[i].date+' · '+modeLabels[mode]+' · '+num(v,2)+'% · '+coverage+' · '+days[i].run_id));svg.append(point)});
 }
}
function setupTrend(){
 const run=reports[selected];$('trendCase').replaceChildren();const all=node('option','全部用例（MAPE）');all.value='';$('trendCase').append(all);
 for(const row of run.rows){const o=node('option',row.name);o.value=row.name;$('trendCase').append(o)}
 if(!run.rows.some(r=>r.name===trendCaseName))trendCaseName='';$('trendCase').value=trendCaseName;drawTrend();
}
function updateTable(){
 const run=reports[selected],query=$('search').value.trim().toLowerCase(),filter=$('filter').value;
 let rows=run.rows.map(r=>modeRow(r,activeMode)).filter(r=>(r.name+' '+r.family).toLowerCase().includes(query)&&(!filter||(filter==='failures'?!['PASS','NOT_RUN','NOT_APPLICABLE'].includes(r.status):r.status===filter)));
 if(sortDirection)rows=rows.slice().sort((a,b)=>{const av=a.ape_percent,bv=b.ape_percent;if(av==null)return bv==null?0:1;if(bv==null)return -1;return sortDirection==='desc'?bv-av:av-bv});
 $('rows').replaceChildren();
 for(const r of rows){
  const tr=node('tr'),name=node('td');name.append(node('div',r.name,'case'),node('div',r.family,'sub'));tr.append(name);
  tr.append(node('td',r.precision===true?'通过':r.precision===false?'失败':'未验证'));
  const route=node('td');route.append(node('div',r.actual_route||'未记录'));if(r.actual_factor!=null)route.append(node('div','F'+r.actual_factor,'sub'));if(r.factor!==1)route.append(node('div','测量 F'+r.factor,'sub'));route.title=r.actual_route_kind||'未取得 TA 最终生效路线';tr.append(route);
  tr.append(node('td',num(r.predicted_us),'numeric'),node('td',num(r.actual_us),'numeric'),node('td',r.ape_percent===null?'—':num(r.ape_percent,2)+'%','numeric'+(r.status==='TIME_FAIL'?' red':'')));
  const status=node('td');status.append(badge(r.status));tr.append(status,node('td',String(r.attempts??0),'numeric'));
  const records=node('td');if(r.log)records.append(artifactLink(run,r.log,'日志'));if(r.json_view){if(records.childNodes.length)records.append(node('span',' · '));const b=node('button','JSON ('+r.json_files.length+')','jsonbutton');b.type='button';b.setAttribute('aria-label','查看 '+r.name+' 的 JSON');b.addEventListener('click',()=>openCaseJson(r.name,true));records.append(b)}else if(r.evidence){if(records.childNodes.length)records.append(node('span',' · '));records.append(artifactLink(run,r.evidence,'数据'))}
  if(r.detail){const details=node('details');details.append(node('summary','原因'),node('pre',r.detail,'detail'));records.append(details)}
  tr.append(records);$('rows').append(tr);
 }
 if(!rows.length){const tr=node('tr'),td=node('td','没有匹配的用例');td.colSpan=9;tr.append(td);$('rows').append(tr)}
 $('tableFoot').textContent=modeLabels[activeMode]+' · 显示 '+rows.length+' / '+run.total+' 个用例 · 最终一次尝试 · SIMD/SIMT-only 以 F1 为主，Mix 按清单登记的 factor 展示';
}
function openCaseJson(name,scroll=false){
 const run=reports[selected],row=run.rows.find(r=>r.name===name);if(!row)return;
 jsonCaseName=name;$('jsonCase').value=name;const ready=!!row.json_view;
 $('jsonFrame').classList.toggle('hidden',!ready);$('jsonEmpty').classList.toggle('hidden',ready);
 $('jsonFrame').src=ready?run.base+row.json_view:'about:blank';
 if(scroll)$('jsonSection').scrollIntoView({behavior:'smooth',block:'start'});
}
function setupJson(){
 const run=reports[selected];$('jsonCase').replaceChildren();
 for(const row of run.rows){const option=node('option',row.name+' · '+(row.json_files||[]).length+' 个文件');option.value=row.name;$('jsonCase').append(option)}
 $('jsonCase').disabled=!run.rows.length;$('jsonSummary').replaceChildren();
 if(run.links.summary){const a=artifactLink(run,run.links.summary,'下载全部用例指标汇总 JSON');a.download='dashboard.json';$('jsonSummary').append(node('span',' · '),a)}
 const first=run.rows.find(r=>r.name===jsonCaseName)||run.rows.find(r=>r.json_view)||run.rows[0];
 if(first)openCaseJson(first.name);else{$('jsonFrame').src='about:blank';$('jsonFrame').classList.add('hidden');$('jsonEmpty').classList.remove('hidden')}
}
function show(index){
 selected=index;const run=reports[index],c=run.counts;$('runSelect').value=String(index);$('runStatus').replaceWith(Object.assign(badge(run.status),{id:'runStatus'}));
 $('runDate').textContent=run.started_at.replace('T',' ').replace('+08:00','');$('runNote').textContent=run.label||('设备 '+(run.device??'—'));
 const messages=[run.message,...(run.warnings||[])].filter(Boolean);const message=messages.join('\n\n')||(run.status==='RUNNING'?'本批次尚无结束记录；请刷新查看进度，或检查运行日志确认进程状态。':'');$('banner').textContent=message;$('banner').classList.toggle('hidden',!message);
 $('passed').textContent=(c.PASS||0)+' / '+run.total;$('precision').textContent=c.PRECISION_FAIL||0;$('timing').textContent=c.TIME_FAIL||0;
 $('abnormal').textContent=(c.EXECUTION_FAIL||0)+(c.BLOCKED||0)+(c.UNVERIFIED||0);$('pending').textContent=c.NOT_RUN||0;
 $('metadata').replaceChildren();
 const protocol=(run.requested_modes||['auto']).map(mode=>modeLabels[mode]||mode).join(' / ')+' · Profiler · 3 个测量进程 · APE ≤ 10%';
 const occupancy=run.occupancy||{},occupancyText=occupancy.status==='BUSY'?'检测到 '+(occupancy.processes||[]).length+' 个占用进程':occupancy.status==='IDLE'?'未检测到占用进程':occupancy.status==='UNKNOWN'?'未能确认':'未记录';
 const items=[['TA 版本',run.ta_version||'尚未完成环境预检'],['测试仓提交',(run.git.sha||'—')+(run.git.dirty?' · 有本地修改':'')],['Profile',run.profile_version||'—'],['Profile SHA256',run.profile_sha256||'—'],['验收协议',protocol],['用例并发',(run.jobs??1)+' 个 case · 共用 NPU '+(run.device??'—')],['设备状态',(run.health.status||'未记录')+' · NPU '+(run.device??'—')],['启动前占用快照',occupancyText]];
 for(const [label,value] of items){const div=node('div');div.append(node('dt',label),node('dd',value));$('metadata').append(div)}
 $('artifacts').replaceChildren();for(const [key,label] of [['log','完整运行日志'],['metadata','运行信息'],['profile','Profile 快照信息'],['occupancy','占用快照']]){if(run.links[key]){if($('artifacts').childNodes.length)$('artifacts').append(node('span',' · '));$('artifacts').append(artifactLink(run,run.links[key],label))}}
 document.querySelectorAll('#history button').forEach((b,i)=>b.classList.toggle('active',i===index));showModes();updateTable();setupTrend();setupJson();
}
if(!reports.length){$('empty').classList.remove('hidden');$('content').classList.add('hidden');$('runSelect').disabled=true}else{
 reports.forEach((r,i)=>{const option=node('option',r.started_at.replace('T',' ').replace('+08:00','')+' · '+(labels[r.status]||r.status));option.value=String(i);$('runSelect').append(option)});
 reports.slice(0,14).forEach((r,i)=>{const b=node('button');b.type='button';b.append(node('div',r.started_at.slice(0,16).replace('T',' '),'day'),node('div',(r.counts.PASS||0)+' / '+r.total,'number'));const track=node('div',undefined,'track'),fill=node('div',undefined,'fill');fill.style.width=(r.total?100*(r.counts.PASS||0)/r.total:0)+'%';track.append(fill);b.append(track,badge(r.status));b.addEventListener('click',()=>show(i));$('history').append(b)});
 $('runSelect').addEventListener('change',()=>show(Number($('runSelect').value)));$('jsonCase').addEventListener('change',()=>openCaseJson($('jsonCase').value));$('search').addEventListener('input',updateTable);$('filter').addEventListener('change',updateTable);$('sort').addEventListener('click',toggleSort);$('apeSort').addEventListener('click',toggleSort);$('trendCase').addEventListener('change',()=>{trendCaseName=$('trendCase').value;drawTrend()});show(0);
}
