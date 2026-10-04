// Dubber content script (injected on demand from the popup via activeTab).
// Plays the Russian dub chunks produced by the local server in sync with the page's <video>.
(() => {
  if (window.__dubber) return; // already injected - the popup talks to it through messages

  const SCHEDULE_AHEAD = 4;   // s: schedule a chunk this long before it starts
  const PREFETCH = 70;        // s of dub to keep downloaded ahead of the playhead
  const MIN_AHEAD = 1.2;      // s: autopause below this
  const RESUME_AHEAD = 12;    // s: autopause releases when this much dub is ready
  const DRIFT_MAX = 0.3;      // s

  const S = {
    enabled: false,
    state: "idle",            // idle starting preparing waiting playing done offline error same
    message: "",
    opts: { voice: "auto", origVol: 0.12, dubVol: 1.0, autopause: true },
    url: "", urlKey: "",
    job: null, jobId: null, recreated: false,
    video: null,
    ctx: null, dubGain: null, origGain: null, duckMode: null, volBase: null, volOurs: null,
    chunks: new Map(),        // k -> {state, buf}
    scheduled: new Map(),     // k -> {src, wall, offset, rate}
    pausedByUs: false, ownPlay: false, userOverride: false,
    ducked: false,
    timers: [], badge: null, listeners: [],
  };
  const mediaNodes = new WeakMap(); // video -> GainNode  (createMediaElementSource can only be called once)

  // ------------------------------------------------------------------ helpers
  function send(msg) {
    return new Promise((resolve) => {
      try {
        chrome.runtime.sendMessage(msg, (res) => {
          if (chrome.runtime.lastError || !res) {
            resolve({ ok: false, error: chrome.runtime.lastError ? chrome.runtime.lastError.message : "no response" });
          } else resolve(res);
        });
      } catch (e) {
        // extension was reloaded/updated: this script is orphaned
        resolve({ ok: false, error: "context_invalidated" });
        cleanup();
      }
    });
  }

  function urlKey(u) {
    try {
      const x = new URL(u);
      if (/(^|\.)youtube\.com$/.test(x.hostname) && x.pathname === "/watch") return "yt:" + x.searchParams.get("v");
      if (x.hostname === "youtu.be") return "yt:" + x.pathname.slice(1, 12);
      return x.origin + x.pathname + x.search;
    } catch (_) { return u; }
  }

  function findVideo() {
    let best = null, bestScore = 0;
    for (const v of document.querySelectorAll("video")) {
      const r = v.getBoundingClientRect();
      let score = r.width * r.height;
      if (!v.paused) score *= 2;
      if (v.readyState === 0 && !(v.duration > 0)) score *= 0.1;
      if (score > bestScore) { best = v; bestScore = score; }
    }
    return best;
  }

  function setState(state, message) {
    S.state = state;
    S.message = message || "";
    renderBadge();
  }

  function on(target, ev, fn, opts) {
    target.addEventListener(ev, fn, opts);
    S.listeners.push(() => target.removeEventListener(ev, fn, opts));
  }

  // ------------------------------------------------------------------ audio graph
  function ensureAudio() {
    if (!S.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      S.ctx = new AC({ latencyHint: "interactive" });
      S.dubGain = S.ctx.createGain();
      S.dubGain.connect(S.ctx.destination);
    }
    S.dubGain.gain.value = S.opts.dubVol;
    if (S.ctx.state !== "running") S.ctx.resume().catch(() => {});
  }

  function setupDuck() {
    if (S.duckMode || !S.video || S.ctx.state !== "running") return;
    const v = S.video;
    if (mediaNodes.has(v)) { S.origGain = mediaNodes.get(v); S.duckMode = "webaudio"; return; }
    const src = v.currentSrc || "";
    let sameOrigin = false;
    try { sameOrigin = new URL(src, location.href).origin === location.origin; } catch (_) {}
    if (src.startsWith("blob:") || !src || sameOrigin) {
      try {
        const node = S.ctx.createMediaElementSource(v);
        const g = S.ctx.createGain();
        node.connect(g); g.connect(S.ctx.destination);
        mediaNodes.set(v, g);
        S.origGain = g; S.duckMode = "webaudio";
        return;
      } catch (_) { /* fall through */ }
    }
    S.duckMode = "volume"; // cross-origin file: routing through WebAudio would silence it
  }

  function applyDuck(active) {
    if (S.ducked === active && S.duckMode) return;
    setupDuck();
    S.ducked = active;
    const level = active ? S.opts.origVol : 1;
    if (S.duckMode === "webaudio") {
      S.origGain.gain.setTargetAtTime(level, S.ctx.currentTime, 0.2);
    } else if (S.duckMode === "volume" && S.video) {
      if (S.volBase === null) S.volBase = S.video.volume;
      S.volOurs = S.volBase * level;
      S.video.volume = S.volOurs;
    }
  }

  function restoreOriginal() {
    if (S.duckMode === "webaudio" && S.origGain) S.origGain.gain.setTargetAtTime(1, S.ctx.currentTime, 0.05);
    if (S.duckMode === "volume" && S.video && S.volBase !== null) S.video.volume = S.volBase;
    S.volBase = null; S.volOurs = null; S.ducked = false;
  }

  // ------------------------------------------------------------------ chunks
  function chunkSec() { return (S.job && S.job.chunk_sec) || 20; }

  function isReady(k) {
    return !!S.job && S.job.ready.some(([a, b]) => k >= a && k < b);
  }

  function aheadSec(t) {
    if (!S.job) return 0;
    const CH = chunkSec(), k = Math.floor(t / CH);
    for (const [a, b] of S.job.ready) if (k >= a && k < b) return Math.min(b * CH, S.job.duration) - t;
    return 0;
  }

  async function fetchChunk(k) {
    const jobId = S.jobId;
    const rec = { state: "loading", buf: null };
    S.chunks.set(k, rec);
    const r = await send({ type: "chunk", id: jobId, k });
    if (jobId !== S.jobId || !S.enabled) return;
    if (!r.ok) { S.chunks.delete(k); return; }
    try {
      const bin = atob(r.data.b64);
      const u8 = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      rec.buf = await S.ctx.decodeAudioData(u8.buffer);
      rec.state = "ready";
    } catch (e) {
      S.chunks.delete(k);
    }
  }

  function scheduleChunk(k, rec) {
    const v = S.video, CH = chunkSec();
    const rate = v.playbackRate || 1;
    const dt = k * CH - v.currentTime;           // video seconds until the chunk starts (<0: already started)
    const offset = dt < 0 ? -dt : 0;
    if (offset >= rec.buf.duration - 0.05) return;
    const src = S.ctx.createBufferSource();
    src.buffer = rec.buf;
    src.playbackRate.value = rate;
    src.connect(S.dubGain);
    const wall = S.ctx.currentTime + Math.max(0, dt) / rate;
    src.start(wall, offset);
    S.scheduled.set(k, { src, wall, offset, rate });
    src.onended = () => { const cur = S.scheduled.get(k); if (cur && cur.src === src) S.scheduled.delete(k); };
  }

  function stopAll() {
    for (const rec of S.scheduled.values()) { try { rec.src.onended = null; rec.src.stop(); } catch (_) {} }
    S.scheduled.clear();
  }

  function driftCheck(t) {
    const CH = chunkSec();
    const k = Math.floor(t / CH), rec = S.scheduled.get(k);
    if (!rec || S.ctx.currentTime < rec.wall) return;
    const expected = t - k * CH;
    const actual = (S.ctx.currentTime - rec.wall) * rec.rate + rec.offset;
    if (Math.abs(actual - expected) > DRIFT_MAX) {
      try { rec.src.onended = null; rec.src.stop(); } catch (_) {}
      S.scheduled.delete(k); // rescheduled on the next tick at the right offset
    }
  }

  // ------------------------------------------------------------------ main loops
  function ownPlay() {
    S.ownPlay = true;
    const p = S.video.play();
    if (p && p.catch) p.catch(() => { S.pausedByUs = false; setState("waiting", "Озвучка готова — нажмите ▶"); });
    setTimeout(() => { S.ownPlay = false; }, 300);
  }

  function tick() {
    const v = S.video;
    if (!S.enabled || !v || !S.job) return;
    if (!v.isConnected) { const nv = findVideo(); if (nv) rebindVideo(nv); return; }
    const t = v.currentTime, CH = chunkSec();
    const k0 = Math.floor(t / CH);

    for (let k = k0; k <= Math.floor((t + PREFETCH) / CH); k++) {
      if (isReady(k) && !S.chunks.has(k)) fetchChunk(k);
    }
    for (const k of [...S.chunks.keys()]) {
      if (k < k0 - 1 || k > k0 + 8) { const s = S.scheduled.get(k); if (s) { try { s.src.stop(); } catch (_) {} S.scheduled.delete(k); } S.chunks.delete(k); }
    }

    const terminal = S.job.status === "done";
    const ahead = aheadSec(t);
    const nearEnd = S.job.duration && t >= S.job.duration - 1;

    // ---- autopause: never let the original run on without a dub
    if (S.opts.autopause && !nearEnd && S.job.status !== "same_language") {
      if (ahead < MIN_AHEAD && !v.paused && !S.userOverride && !terminal) {
        S.pausedByUs = true; S.ownPlay = true; v.pause(); setTimeout(() => { S.ownPlay = false; }, 300);
        setState("waiting", "Дожидаюсь озвучки…");
      } else if (S.pausedByUs && v.paused && (ahead >= RESUME_AHEAD || (terminal && ahead >= MIN_AHEAD))) {
        S.pausedByUs = false; ownPlay();
      }
      if (S.userOverride && ahead >= RESUME_AHEAD) S.userOverride = false;
    }

    const playing = !v.paused && !v.seeking && v.readyState >= 3;
    if (playing && S.ctx.state === "running") {
      for (let k = k0; k <= k0 + 1; k++) {
        const rec = S.chunks.get(k);
        if (!rec || rec.state !== "ready" || S.scheduled.has(k)) continue;
        if (k * CH - t > SCHEDULE_AHEAD) continue;
        scheduleChunk(k, rec);
      }
      driftCheck(t);
    } else {
      stopAll();
    }

    const dubbing = ahead > 0.3 && S.ctx.state === "running";
    applyDuck(dubbing);
    if (dubbing && S.state !== "playing" && !S.pausedByUs) setState("playing", "");
    else if (!dubbing && S.state === "playing") setState("waiting", S.userOverride ? "Озвучка ещё готовится…" : "Дожидаюсь озвучки…");
    else if (S.state === "playing") renderBadge();
  }

  async function poll() {
    if (!S.enabled) return;
    if (urlKey(location.href) !== S.urlKey) { restartForNewPage(); return; }
    const v = S.video;
    const r = await send({ type: "jobStatus", id: S.jobId, t: v ? v.currentTime : 0 });
    if (!S.enabled) return;
    if (!r.ok) {
      if (r.offline) { setState("offline", "Программа Dubber не запущена"); return; }
      if (/unknown job/.test(r.error || "") && !S.recreated) { S.recreated = true; await createJob(); return; }
      setState("error", r.error || "Ошибка");
      return;
    }
    S.recreated = false;
    S.job = r.data;
    const j = r.data;
    if (j.status === "same_language") { setState("same", "Видео уже на русском"); finishSilently(); return; }
    if (j.status === "unsupported" || j.status === "error" || j.status === "cancelled") {
      setState("error", j.error || "Не удалось озвучить видео");
      finishSilently();
      return;
    }
    if (S.state === "offline" || S.state === "starting") setState("preparing", "");
    renderBadge();
  }

  function finishSilently() {
    // keep the badge with the message, but release the video
    stopAll();
    restoreOriginal();
    if (S.pausedByUs && S.video) { S.pausedByUs = false; ownPlay(); }
    clearTimers();
    S.enabled = false;
  }

  // ------------------------------------------------------------------ lifecycle
  function attachVideoListeners() {
    const v = S.video;
    on(v, "pause", () => { stopAll(); if (!S.ownPlay) { S.pausedByUs = false; } });
    on(v, "play", () => { if (!S.ownPlay && S.pausedByUs) { S.pausedByUs = false; S.userOverride = true; } });
    on(v, "waiting", stopAll);
    on(v, "seeking", stopAll);
    on(v, "ratechange", stopAll);
    on(v, "ended", stopAll);
    on(v, "volumechange", () => {
      if (S.duckMode === "volume" && S.volOurs !== null && Math.abs(v.volume - S.volOurs) > 0.005) {
        S.volBase = v.volume / (S.ducked ? Math.max(S.opts.origVol, 0.01) : 1); // the viewer moved the slider
        S.volOurs = v.volume;
      }
    });
  }

  function detachListeners() { S.listeners.forEach((f) => f()); S.listeners = []; }
  function clearTimers() { S.timers.forEach(clearInterval); S.timers = []; }

  function rebindVideo(nv) {
    detachListeners();
    stopAll();
    S.duckMode = null; S.origGain = null;
    S.video = nv;
    attachVideoListeners();
  }

  async function createJob() {
    const r = await send({ type: "createJob", url: S.url, voice: S.opts.voice });
    if (!r.ok) {
      if (r.offline) setState("offline", "Программа Dubber не запущена");
      else setState("error", r.error || "Ошибка");
      S.enabled = r.offline ? true : false;
      return false;
    }
    S.job = r.data; S.jobId = r.data.id;
    return true;
  }

  async function start(opts) {
    stop(true);
    Object.assign(S.opts, opts || {});
    S.enabled = true;
    S.video = findVideo();
    if (!S.video) { S.enabled = false; setState("error", "На странице не найдено видео"); return; }
    ensureAudio();
    S.url = location.href; S.urlKey = urlKey(S.url);
    S.recreated = false; S.userOverride = false; S.pausedByUs = false;
    setState("starting", "Запускаю…");
    attachVideoListeners();
    const gesture = () => { if (S.ctx && S.ctx.state !== "running") S.ctx.resume().catch(() => {}); };
    on(document, "pointerdown", gesture, true);
    on(document, "keydown", gesture, true);
    await createJob();
    if (!S.enabled) return;
    S.timers.push(setInterval(poll, 1000), setInterval(tick, 200), setInterval(positionBadge, 400));
    poll();
  }

  function stop(quiet) {
    clearTimers();
    detachListeners();
    stopAll();
    if (S.ctx) restoreOriginal();
    if (S.pausedByUs && S.video) { S.pausedByUs = false; try { S.video.play(); } catch (_) {} }
    S.chunks.clear();
    S.enabled = false; S.job = null; S.jobId = null; S.duckMode = null; S.origGain = null;
    S.userOverride = false; S.pausedByUs = false;
    if (!quiet) { setState("idle", ""); removeBadge(); }
  }

  function cleanup() { stop(false); }

  function restartForNewPage() {
    const opts = { ...S.opts };
    stop(true);
    setState("starting", "Новое видео…");
    setTimeout(() => start(opts), 900); // let the player swap the <video> source first
  }

  // ------------------------------------------------------------------ badge (shadow DOM pill over the video)
  function ensureBadge() {
    if (S.badge) return S.badge;
    const host = document.createElement("div");
    host.style.cssText = "all:initial;position:fixed;z-index:2147483647;left:12px;top:12px;pointer-events:none;";
    const root = host.attachShadow({ mode: "closed" });
    root.innerHTML = `
      <style>
        .pill{display:inline-flex;align-items:center;gap:8px;max-width:360px;padding:6px 8px 6px 12px;border-radius:999px;
          background:rgba(18,18,22,.86);color:#fff;font:500 13px/1.3 system-ui,-apple-system,"Segoe UI",sans-serif;
          box-shadow:0 2px 10px rgba(0,0,0,.35);pointer-events:auto;backdrop-filter:blur(6px);transition:opacity .25s}
        .pill.fade{opacity:.0}
        .dot{width:8px;height:8px;border-radius:50%;background:#9aa0a6;flex:none}
        .dot.ok{background:#34d399}.dot.wait{background:#fbbf24;animation:p 1.2s infinite}.dot.bad{background:#f87171}
        @keyframes p{50%{opacity:.35}}
        button{all:unset;cursor:pointer;width:20px;height:20px;border-radius:50%;text-align:center;line-height:20px;opacity:.7}
        button:hover{opacity:1;background:rgba(255,255,255,.15)}
        .t{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
      </style>
      <div class="pill"><span class="dot"></span><span class="t"></span><button title="Выключить озвучку">✕</button></div>`;
    const pill = root.querySelector(".pill");
    root.querySelector("button").addEventListener("click", () => stop(false));
    pill.addEventListener("click", () => { if (S.ctx && S.ctx.state !== "running") S.ctx.resume().catch(() => {}); });
    S.badge = { host, pill, dot: root.querySelector(".dot"), txt: root.querySelector(".t"), hideAt: 0 };
    (document.fullscreenElement || document.documentElement).appendChild(host);
    on(document, "fullscreenchange", () => {
      (document.fullscreenElement || document.documentElement).appendChild(host);
    });
    return S.badge;
  }

  function badgeText() {
    const j = S.job;
    switch (S.state) {
      case "starting": return ["wait", S.message || "Запускаю…"];
      case "offline": return ["bad", "Программа Dubber не запущена — откройте её и повторите"];
      case "error": return ["bad", S.message];
      case "same": return ["ok", S.message];
      case "idle": return ["", ""];
    }
    if (S.ctx && S.ctx.state !== "running") return ["wait", "Кликните по странице, чтобы включить озвучку"];
    if (S.state === "playing") {
      const lang = j && j.src_lang ? `${j.src_lang.toUpperCase()} → RU` : "RU";
      const left = j && j.status !== "done" ? ` · готово на ${Math.max(0, Math.round(aheadSec(S.video.currentTime) / 60 * 10) / 10)} мин вперёд` : "";
      return ["ok", `Озвучка: ${lang}${left}`];
    }
    // preparing / waiting
    const stage = j && j.stage ? j.stage : "Готовлю озвучку…";
    const pct = j && j.status === "processing" && j.progress ? ` ${Math.round(j.progress * 100)}%` : "";
    const speed = j && j.speed ? ` (×${j.speed})` : "";
    return ["wait", S.state === "waiting" && S.message ? `${S.message} ${stage}${pct}${speed}` : `${stage}${pct}${speed}`];
  }

  function renderBadge() {
    if (S.state === "idle") return removeBadge();
    const b = ensureBadge();
    const [cls, text] = badgeText();
    b.dot.className = "dot " + cls;
    b.txt.textContent = text;
    b.pill.classList.remove("fade");
    positionBadge();
  }

  function positionBadge() {
    if (!S.badge || !S.video) return;
    const r = S.video.getBoundingClientRect();
    const x = Math.min(Math.max(8, r.left + 12), window.innerWidth - 60);
    const y = Math.min(Math.max(8, r.top + 12), window.innerHeight - 40);
    S.badge.host.style.left = x + "px";
    S.badge.host.style.top = y + "px";
  }

  function removeBadge() {
    if (S.badge) { S.badge.host.remove(); S.badge = null; }
  }

  // ------------------------------------------------------------------ popup API
  function snapshot() {
    const j = S.job;
    return {
      enabled: S.enabled, state: S.state, message: S.message, opts: S.opts,
      hasVideo: !!findVideo(),
      job: j ? {
        title: j.title, stage: j.stage, status: j.status, progress: j.progress, speed: j.speed,
        src_lang: j.src_lang, duration: j.duration,
        ahead: S.video ? Math.round(aheadSec(S.video.currentTime)) : 0,
      } : null,
      audioBlocked: !!(S.ctx && S.ctx.state !== "running"),
    };
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (!msg || typeof msg.type !== "string" || !msg.type.startsWith("dubber:")) return false;
    switch (msg.type) {
      case "dubber:state": sendResponse(snapshot()); return false;
      case "dubber:start": start(msg.opts).then(() => sendResponse(snapshot())); return true;
      case "dubber:stop": stop(false); sendResponse(snapshot()); return false;
      case "dubber:debug":
        sendResponse({
          ctx: S.ctx && S.ctx.state, duckMode: S.duckMode, ducked: S.ducked, scheduled: [...S.scheduled.keys()],
          chunks: [...S.chunks.entries()].map(([k, r]) => [k, r.state]), pausedByUs: S.pausedByUs,
          origGain: S.origGain ? +S.origGain.gain.value.toFixed(3) : null, url: S.url,
        });
        return false;
      case "dubber:opts":
        Object.assign(S.opts, msg.opts || {});
        if (S.dubGain) S.dubGain.gain.value = S.opts.dubVol;
        if (S.ducked) { S.ducked = false; applyDuck(true); }
        sendResponse(snapshot());
        return false;
    }
    return false;
  });

  window.__dubber = { snapshot, stop };
})();
