const {spawn} = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'shopping-browser-'));
const browser = ['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe'].find(p => fs.existsSync(p));
assert(browser, 'Install Chrome or Edge to run browser checks');
const chrome = spawn(browser, [
  '--headless=new', '--no-first-run', '--no-default-browser-check', '--disable-gpu',
  '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank',
], {windowsHide: true, stdio: 'ignore'});
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const report = {checks: [], errors: [], posts: []};
let ws;
(async () => {
  let active;
  for (let n=0;n<100;n++) {try{active=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8');break;}catch{} await sleep(100);}
  assert(active, 'Chrome debugging endpoint');
  const port = active.split('\n')[0];
  const target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method:'PUT'})).json();
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject) => {ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
  let seq=0;
  const pending=new Map();
  ws.addEventListener('message', (event) => {
    const m=JSON.parse(event.data);
    if (m.id) {const handler=pending.get(m.id); if(handler){pending.delete(m.id); m.error?handler.reject(new Error(JSON.stringify(m.error))):handler.resolve(m.result);}}
    if (m.method==='Runtime.exceptionThrown') report.errors.push(m.params.exceptionDetails.text);
    if (m.method==='Network.requestWillBeSent' && m.params.request.url.includes('/api/') && m.params.request.method==='POST') report.posts.push(JSON.parse(m.params.request.postData));
  });
  function call(method,params={}) {return new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}));});}
  async function evaluate(expression) {
    const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
    if(r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  }
  async function until(expression,label) {
    for(let n=0;n<300;n++){if(await evaluate(expression)){report.checks.push(label);return;}await sleep(100);}
    throw new Error('Timed out: '+label+' | '+await evaluate('document.body.innerText'));
  }
  await call('Page.enable'); await call('Runtime.enable'); await call('Network.enable');
  await call('Emulation.setDeviceMetricsOverride',{width:1360,height:980,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url:'http://127.0.0.1:8000/'});
  await until("document.querySelector('#connection')?.textContent==='会话保存在本机'",'initial load');
  assert(await evaluate("!document.querySelector('#welcome').hidden"));
  assert.equal(await evaluate("document.querySelector('#online').checked"),false);
  await evaluate("document.querySelectorAll('[data-example]')[1].click(); document.querySelector('#chat-form').requestSubmit()");
  await until("document.querySelector('#connection').textContent==='等待你补充条件'",'clarification pause');
  assert.equal(await evaluate("document.querySelectorAll('.product-card').length"),0);
  await evaluate("document.querySelector('#query').value='买跑步鞋。'; document.querySelector('#chat-form').requestSubmit()");
  await until("document.querySelectorAll('.product-card').length>=1 && document.querySelector('#connection').textContent==='会话保存在本机'",'resume and product cards');
  assert(await evaluate("document.querySelector('.price').textContent.includes('¥')"));
  await evaluate("document.querySelector('.citations summary')?.click()");
  assert(await evaluate("document.querySelector('.citations')?.open"));
  report.session=await evaluate("document.querySelector('#session-id').value");
  const count=await evaluate("document.querySelectorAll('.message.user').length");
  await call('Page.reload');
  await until("document.querySelectorAll('.message.user').length===2 && document.querySelector('#connection').textContent==='会话保存在本机'",'reload restores both turns');
  assert.equal(count,2);
  assert(await evaluate("document.querySelector('.composer-wrap').getBoundingClientRect().top >= document.querySelector('.workspace').getBoundingClientRect().bottom-1"), 'composer does not cover chat');
  report.checks.push('composer stays outside chat scroll area');
  await evaluate("const pane=document.querySelector('.workspace'); pane.scrollTop=document.querySelector('.message.assistant:last-child').offsetTop-pane.offsetTop");
  await sleep(250);
  fs.writeFileSync(path.join(root,'runs','web_desktop.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})).data,'base64'));
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await sleep(250);
  assert(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'), 'mobile has no horizontal overflow');
  assert(await evaluate("document.querySelector('.composer-wrap').getBoundingClientRect().top >= document.querySelector('.workspace').getBoundingClientRect().bottom-1"));
  report.checks.push('mobile layout fits viewport');
  fs.writeFileSync(path.join(root,'runs','web_mobile.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})).data,'base64'));
  await evaluate("document.querySelector('#new-session').click()");
  assert(await evaluate("!document.querySelector('#welcome').hidden"));
  assert.notEqual(await evaluate("document.querySelector('#session-id').value"),report.session);
  await evaluate(`document.querySelector('#session-id').value=${JSON.stringify(report.session)}; document.querySelector('#restore-form').requestSubmit()`);
  await until("document.querySelectorAll('.message.user').length===2 && !document.querySelector('#send').disabled",'manual session restore');
  // Failed parsing has no existing cache; it must stop offline, without sending online=true.
  await evaluate("document.querySelector('#new-session').click(); document.querySelector('#query').value='web-offline-cache-check-unique-20261025'; document.querySelector('#chat-form').requestSubmit()");
  await until("!document.querySelector('#retry').hidden && !document.querySelector('#retry').disabled",'offline cache miss exposes retry');
  assert(await evaluate("document.querySelector('#send').disabled"));
  await evaluate("document.querySelector('#retry').click()");
  await until("!document.querySelector('#retry').disabled && document.querySelector('#connection').textContent==='进度已保存，可以重试'",'explicit offline retry retains failure');
  assert(report.posts.every(body => body.online===false));
  assert.equal(report.errors.length,0);
  await evaluate("window.__shoppingXss=0; assistantMessage({status:'completed',candidates:[{product_id:'x',title:'<img src=x onerror=window.__shoppingXss=1>',sku:{price:100,properties:{}}}],answer:{claims:[{text:'<script>window.__shoppingXss=1</script>',citations:[{quote:'<img onerror=alert(1)>'}]}]}})");
  assert.equal(await evaluate('window.__shoppingXss'),0);
  assert.equal(await evaluate("document.querySelectorAll('#messages img, #messages script').length"),0);
  report.checks.push('model and product HTML stays inert text');
  report.checks.push('all requests explicitly offline','no uncaught browser errors');
  fs.writeFileSync(path.join(root,'runs','web_browser_check.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report));
})().catch(error=>{console.error(error.stack);process.exitCode=1;}).finally(async()=>{if(ws)ws.close();chrome.kill();});
