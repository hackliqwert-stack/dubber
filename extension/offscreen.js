// Live mode: captures the audio of a tab, streams it to the local server over WebSocket,
// plays the original quietly and the Russian dub on top as the utterances come back.
const WS_URL = "ws://127.0.0.1:8765/live";
const MAX_BACKLOG = 25; // s of queued dub; anything beyond this is skipped instead of lagging forever

let S = null;

function report(patch) {
  chrome.runtime.sendMessage({ type: "live:status", ...patch }).catch(() => {});
}

function stop(reason) {
  if (!S) return;
  const s = S;
  S = null;
  try { s.ws.close(); } catch (_) {}
  try { s.stream.getTracks().forEach((t) => t.stop()); } catch (_) {}
  try { s.ctx.close(); } catch (_) {}
  report({ status: reason ? "error" : "idle", message: reason || "", active: false });
}

async function start({ streamId, opts }) {
  stop();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId } },
    video: false,
  });
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule("live-worklet.js");

  const src = ctx.createMediaStreamSource(stream);
  const origGain = ctx.createGain();
  origGain.gain.value = opts.origVol;
  src.connect(origGain).connect(ctx.destination); // tab capture mutes the tab: pass the original through ourselves

  const dubGain = ctx.createGain();
  dubGain.gain.value = opts.dubVol;
  dubGain.connect(ctx.destination);

  const node = new AudioWorkletNode(ctx, "downsampler");
  const sink = ctx.createGain();
  sink.gain.value = 0;
  src.connect(node);
  node.connect(sink).connect(ctx.destination);

  const ws = new WebSocket(WS_URL);
  ws.binaryType = "arraybuffer";
  S = { ctx, stream, ws, origGain, dubGain, playEnd: 0, t0: ctx.currentTime, pending: null, lang: null, count: 0 };

  ws.onopen = () => ws.send(JSON.stringify({ voice: opts.voice }));
  ws.onerror = () => { if (S && S.ws === ws) stop("Нет связи с программой Dubber — запущена ли она?"); };
  ws.onclose = () => { if (S && S.ws === ws) stop("Связь с программой потеряна"); };
  ws.onmessage = async (ev) => {
    if (!S || S.ws !== ws) return;
    if (typeof ev.data === "string") {
      const m = JSON.parse(ev.data);
      if (m.type === "ready") report({ status: "listening", message: "Слушаю…", active: true });
      else if (m.type === "dub") S.pending = m;
      else if (m.type === "same_language") stop("Видео уже на русском");
      else if (m.type === "error") report({ status: "error", message: m.error, active: true });
      return;
    }
    const meta = S.pending;
    S.pending = null;
    if (!meta) return;
    try {
      const buf = await S.ctx.decodeAudioData(ev.data);
      const now = S.ctx.currentTime;
      const backlog = Math.max(0, S.playEnd - now);
      if (backlog > MAX_BACKLOG) return; // too far behind: skip this phrase
      const rate = Math.min(1.5, 1 + backlog / 15);
      const when = Math.max(now + 0.05, S.playEnd);
      const b = S.ctx.createBufferSource();
      b.buffer = buf;
      b.playbackRate.value = rate;
      b.connect(S.dubGain);
      b.start(when);
      S.playEnd = when + buf.duration / rate;
      S.lang = meta.lang;
      S.count++;
      const lag = Math.max(0, when - S.t0 - meta.t0);
      report({ status: "playing", message: "", active: true, lang: meta.lang, lag: Math.round(lag), phrases: S.count, last: meta.ru });
    } catch (_) { /* undecodable chunk: ignore */ }
  };

  node.port.onmessage = (e) => {
    if (!S || ws.readyState !== WebSocket.OPEN) return;
    const f = new Float32Array(e.data);
    const i16 = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) i16[i] = Math.max(-1, Math.min(1, f[i])) * 32767;
    ws.send(i16.buffer);
  };

  stream.getAudioTracks()[0].onended = () => stop();
  report({ status: "starting", message: "Подключаюсь…", active: true });
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.target !== "offscreen") return false;
  if (msg.type === "live:start") {
    start(msg).then(() => sendResponse({ ok: true })).catch((e) => {
      stop(String((e && e.message) || e));
      sendResponse({ ok: false });
    });
    return true;
  }
  if (msg.type === "live:stop") { stop(); sendResponse({ ok: true }); return false; }
  if (msg.type === "live:opts" && S) {
    S.origGain.gain.value = msg.opts.origVol;
    S.dubGain.gain.value = msg.opts.dubVol;
    sendResponse({ ok: true });
    return false;
  }
  return false;
});
