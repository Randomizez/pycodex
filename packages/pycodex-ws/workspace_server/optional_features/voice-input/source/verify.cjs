// Run manually with Node 18+ and Playwright; see ../README.md.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

async function main() {
  const frontend = __dirname;
  const feature = path.resolve(frontend, '..');
  const workspace = path.resolve(feature, '../..');
  const source = fs.readFileSync(path.join(workspace, 'workspace.html'), 'utf8');
  const html = source.replace(/__([A-Z_]+)__/g, (_, name) => ({
    WORKSPACE_TITLE: 'Voice input check', ENABLE_BOARD: 'false', ACCESS_ROLE: 'admin',
    BOARD_URL: 'about:blank', CWD_LABEL: '/test',
  }[name] || '')).replace('</head>', '<link rel="stylesheet" href="frontend/custom.css"></head>')
    .replace('</body>', '<script type="module" src="frontend/custom.js" data-pycodex-custom-script></script></body>')
    .replace('    startContentViews();', `
      window.voiceFixture = {composer, selectSession, setSessionsOpen, sessionState,
        active: () => activeSessionId};
      startContentViews();`)
    .replace('await extension.mount(Object.freeze({', 'window.voiceFixture.dispose = await extension.mount(Object.freeze({');
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    args: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
  });
  try {
    const context = await browser.newContext({viewport: {width: 1100, height: 850}, hasTouch:true});
    await context.grantPermissions(['microphone'], {origin: 'https://voice.test'});
    const page = await context.newPage();
    const errors = [], requests = [], polishRequests = [], workspacePosts = [];
    let held = null, heldPolish = null, nextMode = 'ok', nextPolishMode = 'ok';
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(() => {
      window.voiceTracks = [];
      window.voicePromptFocuses = 0;
      document.addEventListener('focusin', event => {
        if (event.target.id === 'prompt') voicePromptFocuses++;
      });
      const acquire = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
      navigator.mediaDevices.getUserMedia = async options => {
        const stream = await acquire(options);
        voiceTracks.push(...stream.getTracks());
        if (window.holdPermission) await new Promise(resolve => { window.releasePermission = resolve; });
        return stream;
      };
    });
    const sessions = ['a', 'b'].map(id => ({id, title: 'Session ' + id, cwd: '/test', running: false}));
    const events = ': keepalive\r\n\r\ndata: ' + JSON.stringify({type:'transcript.text.delta', delta:'识别旧文'})
      + '\r\n\r\ndata: ' + JSON.stringify({type:'transcript.text.done', text:'识别成功。'}) + '\r\n\r\n';
    await page.route('**/*', async route => {
      const request = route.request(), url = new URL(request.url());
      if (url.hostname === 'api.stepfun.com') {
        assert.equal(request.method(), 'POST');
        assert.equal(request.headers().authorization, 'Bearer synthetic-voice-key');
        const body = request.postDataJSON();
        if (url.pathname === '/step_plan/v1/chat/completions') {
          assert.equal(body.model, 'step-3.7-flash');
          assert.equal(body.reasoning_effort, 'low');
          assert.equal(body.stream, false);
          assert.equal(body.messages.length, 2);
          assert.equal(body.messages[0].role, 'system');
          assert.deepEqual(body.messages[1], {role:'user', content:'识别成功。'});
          polishRequests.push(body);
          const mode = nextPolishMode;
          nextPolishMode = 'ok';
          if (mode === 'hold') await new Promise(resolve => {heldPolish = resolve;});
          return route.fulfill({status:mode === 'failed' ? 500 : 200,
            headers:{'access-control-allow-origin':'*'}, json:{choices:[{
              finish_reason:mode === 'incomplete' ? 'length' : 'stop',
              message:{role:'assistant', content:'整理后的文字。'},
            }]}});
        }
        assert.equal(url.pathname, '/step_plan/v1/audio/asr/sse');
        assert.deepEqual(body.audio.input.format, {type:'pcm', codec:'pcm_s16le', rate:16000, bits:16, channel:1});
        assert.equal(body.audio.input.transcription.model, 'stepaudio-2.5-asr');
        const pcm = Buffer.from(body.audio.data, 'base64');
        assert(pcm.length > 3000 && pcm.length < 16000 * 2 * 10);
        assert(pcm.some(value => value !== 0), 'fake microphone should produce real encoded samples');
        requests.push({bytes:pcm.length});
        const mode = nextMode;
        nextMode = 'ok';
        if (mode === 'hold') await new Promise(resolve => {held = resolve;});
        return route.fulfill({status:mode === 'denied' ? 401 : 200,
          headers:{'access-control-allow-origin':'*'},
          contentType: 'text/event-stream', body:mode === 'denied' ? '{}' : events});
      }
      if (url.hostname !== 'voice.test') {
        return route.fulfill({contentType:'text/javascript', body:url.pathname.includes('mermaid')
          ? 'export default {initialize(){},render(){return {svg:""}}};' : ''});
      }
      if (url.pathname === '/') return route.fulfill({contentType:'text/html', body:html});
      if (url.pathname.startsWith('/frontend/')) {
        const name = url.pathname.slice('/frontend/'.length);
        return route.fulfill({contentType:name.endsWith('.css') ? 'text/css' : 'text/javascript',
          body:fs.readFileSync(path.join(frontend, name))});
      }
      if (url.pathname === '/voice-parser.js') return route.fulfill({contentType:'text/javascript',
        body:fs.readFileSync(path.join(frontend, 'voice-input.js'),'utf8') + '\nexport {transcript};'});
      if (url.pathname.startsWith('/assets/')) return route.fulfill({
        contentType:url.pathname.endsWith('.css') ? 'text/css' : 'text/javascript',
        body:fs.readFileSync(path.join(workspace, url.pathname.slice('/assets/'.length)))});
      if (request.method() === 'POST') workspacePosts.push(url.pathname);
      if (url.pathname === '/api/sessions') return route.fulfill({json:{sessions, profiles:[], default_profile:null}});
      if (url.pathname === '/api/views') return route.fulfill({json:{views:[]}});
      if (url.pathname === '/api/session') {
        const sessionId = url.searchParams.get('session_id') || 'a';
        return route.fulfill({json:{session_id:sessionId, sessions,
          snapshot:{cwd:'/test', turns:[], spinner:'', queued_inputs:[], running:false}}});
      }
      if (url.pathname.endsWith('/history')) return route.fulfill({json:{items:[], next_cursor:null}});
      throw new Error('Unexpected workspace request: ' + url.pathname);
    });
    await page.goto('https://voice.test/');
    await page.waitForFunction(() => window.voiceFixture?.active() === 'a');
    await page.locator('.voice-mic').waitFor();
    assert.equal(await page.locator('.voice-controls').count(), 1);
    assert.equal(await page.locator('.voice-controls button').count(), 1);
    assert.equal(await page.locator('.voice-panel, .voice-config').count(), 0);
    await page.locator('.voice-mic').click();
    assert.equal(await page.locator('.voice-settings').isVisible(), true);
    await page.locator('#voice-api-key').fill('synthetic-voice-key');
    await page.locator('.voice-settings-save').click();
    assert.equal(await page.evaluate(() => localStorage.getItem('pycodex.voice.stepApiKey')), 'synthetic-voice-key');
    assert.equal(await page.locator('#voice-api-key').inputValue(), '');
    await page.waitForFunction(() => document.querySelector('.voice-mic').dataset.state === 'recording');
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => voiceTracks.every(track => track.readyState === 'ended'));
    await page.reload();
    await page.locator('.voice-mic').waitFor();
    const original = [
      {type:'text', text:'before '},
      {type:'context', id:'context-1', label:'file.py:1', text:'reference'},
      {type:'text', text:' after'},
    ];
    await page.evaluate(parts => {
      voiceFixture.composer.set({parts, selection:{start:8,end:8}});
      voiceFixture.composer.focus();
    }, original);
    const originalFocusCount = await page.evaluate(() => voicePromptFocuses);
    async function start() {
      await page.locator('.voice-mic').click();
      await page.waitForFunction(() => document.querySelector('.voice-mic').getAttribute('aria-pressed') === 'true');
      await page.waitForTimeout(500);
    }
    async function stop() { await page.locator('.voice-mic').click(); }
    nextPolishMode = 'hold';
    await start();
    await stop();
    await page.waitForFunction(() => document.querySelector('.voice-mic').dataset.state === 'polishing');
    while (!heldPolish) await new Promise(resolve => setTimeout(resolve, 20));
    assert.deepEqual(await page.evaluate(() => voiceFixture.composer.read().parts), original,
      'raw transcript must not enter the draft while polishing');
    heldPolish(); heldPolish = null;
    await page.waitForFunction(() => document.querySelector('.voice-mic').dataset.state === 'idle');
    let parts = await page.evaluate(() => voiceFixture.composer.read().parts);
    assert.equal(parts.map(item => item.text).join(''), 'before reference整理后的文字。 after');
    assert.deepEqual(parts.find(item => item.type === 'context'), original[1]);
    assert.equal(await page.evaluate(() => voicePromptFocuses), originalFocusCount);
    assert.notEqual(await page.evaluate(() => document.activeElement.id), 'prompt');
    await page.locator('#prompt').focus();
    await page.keyboard.press('Control+z');
    assert.deepEqual(await page.evaluate(() => voiceFixture.composer.read().parts), original);
    assert.equal(await page.evaluate(() => voiceTracks.every(track => track.readyState === 'ended')), true);

    const beforeCancel = requests.length;
    await start();
    await page.keyboard.press('Escape');
    await page.waitForTimeout(150);
    assert.equal(requests.length, beforeCancel);
    assert.equal(await page.evaluate(() => voiceTracks.every(track => track.readyState === 'ended')), true);

    nextMode = 'hold';
    await start();
    await stop();
    await page.waitForFunction(() => document.querySelector('.voice-mic').dataset.state === 'transcribing');
    while (!held) await new Promise(resolve => setTimeout(resolve, 20));
    await page.evaluate(() => voiceFixture.selectSession('b'));
    held(); held = null;
    await page.waitForTimeout(150);
    assert.equal(await page.evaluate(() => voiceFixture.composer.read().parts.length), 0);
    await page.evaluate(() => voiceFixture.selectSession('a'));
    assert.deepEqual(await page.evaluate(() => voiceFixture.composer.read().parts), original);

    await page.evaluate(() => {window.holdPermission = true;});
    await page.locator('.voice-mic').click();
    await page.waitForFunction(() => typeof window.releasePermission === 'function');
    await page.locator('.voice-mic').click();
    await page.evaluate(() => {window.holdPermission = false; releasePermission();});
    await page.waitForFunction(() => voiceTracks.every(track => track.readyState === 'ended'));

    nextPolishMode = 'hold';
    await start(); await stop();
    await page.waitForFunction(() => document.querySelector('.voice-mic').dataset.state === 'polishing');
    while (!heldPolish) await new Promise(resolve => setTimeout(resolve, 20));
    await page.locator('.voice-mic').click();
    heldPolish(); heldPolish = null;
    await page.waitForTimeout(150);
    assert.deepEqual(await page.evaluate(() => voiceFixture.composer.read().parts), original);

    nextPolishMode = 'hold';
    await start(); await stop();
    await page.waitForFunction(() => document.querySelector('.voice-mic').dataset.state === 'polishing');
    while (!heldPolish) await new Promise(resolve => setTimeout(resolve, 20));
    await page.evaluate(() => voiceFixture.selectSession('b'));
    heldPolish(); heldPolish = null;
    await page.waitForTimeout(150);
    assert.equal(await page.evaluate(() => voiceFixture.composer.read().parts.length), 0);
    await page.evaluate(() => voiceFixture.selectSession('a'));
    assert.deepEqual(await page.evaluate(() => voiceFixture.composer.read().parts), original);

    for (const mode of ['failed', 'incomplete']) {
      nextPolishMode = mode;
      await start(); await stop();
      await page.waitForFunction(() => document.querySelector('.voice-notice').classList.contains('visible'));
      await page.waitForFunction(() => document.querySelector('.voice-mic').dataset.state === 'idle');
      assert.deepEqual(await page.evaluate(() => voiceFixture.composer.read().parts), original,
        'failed or truncated polish must leave the draft intact');
    }
    nextMode = 'denied';
    await start(); await stop();
    await page.waitForFunction(() => document.querySelector('.voice-settings').open);
    assert.match(await page.locator('.voice-settings-error').textContent(), /401/);
    assert.deepEqual(await page.evaluate(() => voiceFixture.composer.read().parts), original);
    assert.equal((await page.locator('.voice-settings-error').textContent()).includes('synthetic-voice-key'), false);
    await page.locator('.voice-settings-cancel').click();

    const parserChecks = await page.evaluate(async () => {
      const {transcript} = await import('/voice-parser.js');
      const encode = item => 'data: ' + JSON.stringify(item) + '\r\n\r\n';
      const bytes = new TextEncoder().encode(': ping\r\n\r\n'
        + encode({type:'transcript.text.delta', delta:'中'})
        + encode({type:'transcript.text.delta', delta:'文'})
        + encode({type:'transcript.text.done', text:'中文。'}));
      const stream = new ReadableStream({start(controller) {
        for (const byte of bytes) controller.enqueue(new Uint8Array([byte]));
        controller.close();
      }});
      const final = await transcript(new Response(stream));
      let incomplete = '', failed = '';
      try { await transcript(new Response(encode({type:'transcript.text.delta', delta:'partial'}))); }
      catch (error) { incomplete = error.message; }
      try { await transcript(new Response(encode({type:'error', message:'synthetic failure'}))); }
      catch (error) { failed = error.message; }
      return {final, incomplete, failed};
    });
    assert.equal(parserChecks.final, '中文。');
    assert.match(parserChecks.incomplete, /提前结束/);
    assert.equal(parserChecks.failed, 'synthetic failure');
    assert.deepEqual(workspacePosts, [], 'speech must not automatically send any message or key to the workspace');
    assert.deepEqual(errors, []);
    const heights = [];
    async function checkHeight() {
      const sizes = await page.evaluate(() => {
        const controls = document.querySelector('.voice-controls');
        const surface = document.querySelector('.composer-surface');
        const withButton = surface.getBoundingClientRect().height;
        controls.style.display = 'none';
        const withoutButton = surface.getBoundingClientRect().height;
        controls.style.display = '';
        return {withButton, withoutButton};
      });
      assert.equal(sizes.withButton, sizes.withoutButton, 'voice button must not increase composer height');
      heights.push(sizes);
    }
    await page.evaluate(() => {
      document.activeElement.blur();
      voiceFixture.composer.set({parts:[]});
    });
    await checkHeight();
    await page.screenshot({path:path.join(feature, 'preview.png')});
    await page.setViewportSize({width:390,height:844});
    await page.evaluate(() => {
      document.querySelector('.workspace').dataset.mobilePane = 'chat';
      voiceFixture.setSessionsOpen(false, false, false);
    });
    await page.waitForTimeout(400);
    await checkHeight();
    const mobileFocusCount = await page.evaluate(() => voicePromptFocuses);
    for (let i = 0; i < 2; i++) {
      await page.locator('.voice-mic').tap();
      await page.waitForFunction(() => document.querySelector('.voice-mic').dataset.state === 'recording');
      await page.waitForTimeout(500);
      await page.locator('.voice-mic').tap();
      await page.waitForFunction(() => document.querySelector('.voice-mic').dataset.state === 'idle');
      assert.notEqual(await page.evaluate(() => document.activeElement.id), 'prompt');
    }
    assert.equal(await page.evaluate(() => voicePromptFocuses), mobileFocusCount,
      'touch recording and final text must never focus the editor');
    assert.equal(await page.evaluate(() => voiceFixture.composer.read().parts.map(part => part.text).join('')),
      '整理后的文字。整理后的文字。');
    assert.equal(await page.locator('.composer-surface').evaluate(node => node.getBoundingClientRect().height),
      heights[1].withoutButton, 'dictation into an empty draft must not create a leading blank line');
    await page.screenshot({path:path.join(feature, 'preview-mobile.png')});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);

    await start();
    await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pagehide', {persisted:true})));
    await page.waitForFunction(() => voiceTracks.every(track => track.readyState === 'ended'));
    assert.equal(await page.locator('.voice-mic').getAttribute('data-state'), 'idle');
    await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pageshow', {persisted:true})));
    await start();
    await page.evaluate(() => voiceFixture.dispose());
    await page.waitForFunction(() => voiceTracks.every(track => track.readyState === 'ended'));
    assert.equal(await page.locator('.voice-controls, .voice-notice, .voice-settings').count(), 0);
    await page.evaluate(() => voiceFixture.selectSession('b'));
    assert.equal(await page.locator('.voice-controls').count(), 0, 'disposed component must not reappear');
    await page.reload();
    await page.locator('.voice-mic').waitFor();
    await start();
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => voiceTracks.every(track => track.readyState === 'ended'));
    assert.deepEqual(workspacePosts, []);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed:true, real_microphone_recordings:requests.length,
      polish_requests:polishRequests.length,
      composer_heights:heights,
      checks:['single button without information bar','key entry auto-start and persistence','real MediaRecorder and PCM resampling',
        'Step 3.7 polish before draft insertion','final text and context preservation',
        'single undo','cancel recording','cancel pending permission','cancel polish','session isolation and late response in both stages',
        'failed or truncated polish leaves draft intact','invalid key asks again',
        'HTTP error','UTF-8 and CRLF split SSE','missing final and SSE error','no automatic submission',
        'mobile touch never focuses editor','repeated dictation insertion','unchanged empty composer height','mobile layout',
        'page hide and page cache restoration','unmount during recording','reload after unmount']}));
  } finally { await browser.close(); }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
