const $ = (id) => document.getElementById(id);

const SETTINGS_VERSION = 2;
const DEFAULTS = { settingsVersion: SETTINGS_VERSION, voice: "baya", origVol: 0.12, dubVol: 1.0, autopause: true };
let settings = { ...DEFAULTS };
let tab = null;
let health = null;
let injectable = true;

function bg(msg) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(msg, (r) => resolve(r || { ok: false, error: "no response" }));
  });
}

async function tabMsg(msg) {
  if (!tab) return null;
  try { return await chrome.tabs.sendMessage(tab.id, msg); } catch (_) { return null; }
}

async function loadSettings() {
  const saved = await chrome.storage.local.get("settings");
  const old = saved.settings || {};
  settings = { ...DEFAULTS, ...old };
  // v1 had "auto" (a different voice per character) as the default.  Existing installs that
  // never changed it should get the faster, more consistent single voice once.
  if (!old.settingsVersion) {
    if (!old.voice || old.voice === "auto") settings.voice = DEFAULTS.voice;
    settings.settingsVersion = SETTINGS_VERSION;
    chrome.storage.local.set({ settings });
  }
}

function saveSettings() {
  chrome.storage.local.set({ settings });
  tabMsg({ type: "dubber:opts", opts: settings });
  if (live.active) bg({ type: "liveOpts", opts: settings });
}

function fillVoices(voices) {
  const sel = $("voice");
  if (sel.options.length) return;
  for (const v of voices || []) sel.add(new Option(v.name, v.id));
  const known = (voices || []).some((v) => v.id === settings.voice);
  sel.value = known ? settings.voice : ((voices && voices[0] && voices[0].id) || DEFAULTS.voice);
  settings.voice = sel.value;
}

function renderSettings() {
  $("voice").value = settings.voice;
  $("orig").value = Math.round(settings.origVol * 100);
  $("origOut").textContent = Math.round(settings.origVol * 100) + "%";
  $("dub").value = Math.round(settings.dubVol * 100);
  $("dubOut").textContent = Math.round(settings.dubVol * 100) + "%";
  $("autopause").checked = settings.autopause;
}

function setDot(el, cls) { el.className = "dot " + (cls || ""); }

function fmtMin(sec) {
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return m ? `${m} мин ${s} с` : `${s} с`;
}

let live = { active: false };

function renderLive() {
  const on = !!live.active;
  $("liveRun").hidden = !on;
  if (!on) return;
  const dot = live.status === "playing" ? "ok" : live.status === "error" ? "bad" : "wait";
  setDot($("liveDot"), dot);
  $("liveText").textContent = live.status === "playing" ? "Живой режим: озвучка идёт"
    : live.status === "error" ? (live.message || "Ошибка") : (live.message || "Слушаю…");
  const bits = [];
  if (live.lang) bits.push(`${live.lang.toUpperCase()} → RU`);
  if (live.lag) bits.push(`отставание ${live.lag} с`);
  if (live.phrases) bits.push(`фраз: ${live.phrases}`);
  if (live.tabId && tab && live.tabId !== tab.id) bits.push("включён в другой вкладке");
  $("liveMeta").textContent = bits.join(" · ");
}

function render(state) {
  const online = !!health;
  $("offline").hidden = online;
  $("control").hidden = !online || !injectable;
  $("blocked").hidden = !online || injectable;

  if (!online) {
    setDot($("srvDot"), "bad");
    $("srvText").textContent = "программа не запущена";
    return;
  }
  setDot($("srvDot"), "ok");
  const gpu = health.gpu && health.gpu.cuda ? health.gpu.name : "без видеокарты (медленно)";
  $("srvText").textContent = gpu;
  fillVoices(health.voices);
  $("firstRun").hidden = Object.values(health.downloaded || {}).every(Boolean);
  $("ver").textContent = "Dubber " + (health.version || "");

  const running = state && state.enabled;
  const finished = state && !state.enabled && (state.state === "same" || state.state === "error");
  $("idleBox").hidden = running || !!live.active;
  $("runBox").hidden = (!running && !finished) || !!live.active;
  renderLive();
  $("startBtn").disabled = !!state && state.hasVideo === false;
  $("hint").textContent = state && state.hasVideo === false ? "На этой странице не видно видео." : "";

  if (running || finished) {
    const j = state.job;
    let text = state.message || "";
    let dot = "wait";
    if (state.state === "playing") { text = "Озвучка идёт"; dot = "ok"; }
    else if (state.state === "same") { text = "Видео уже на русском"; dot = "ok"; }
    else if (state.state === "error") { text = state.message || "Ошибка"; dot = "bad"; }
    else if (state.state === "offline") { text = "Нет связи с программой"; dot = "bad"; }
    else if (j && j.stage) { text = j.stage; }
    else if (!text) { text = "Готовлю…"; }
    if (state.audioBlocked) { text = "Кликните по странице, чтобы включить звук"; dot = "wait"; }
    $("runText").textContent = text;
    setDot($("runDot"), dot);
    $("bar").style.width = j && j.progress ? Math.round(j.progress * 100) + "%" : "0";
    const bits = [];
    if (j && j.title) bits.push(j.title);
    if (j && j.downloading && j.download_progress) bits.push(`скачано ${Math.round(j.download_progress * 100)}%`);
    if (j && j.src_lang) bits.push(`${j.src_lang.toUpperCase()} → RU`);
    if (j && j.speed) bits.push(`обработка ×${j.speed}`);
    if (j && j.ahead) bits.push(`озвучено вперёд: ${fmtMin(j.ahead)}`);
    $("meta").textContent = bits.join(" · ");
    $("stopBtn").textContent = running ? "Выключить озвучку" : "Закрыть";
  }
}

async function refresh() {
  const h = await bg({ type: "health" });
  health = h.ok ? h.data : null;
  const l = await bg({ type: "liveState" });
  live = l.ok ? l.data : { active: false };
  let state = null;
  if (health && injectable) state = await tabMsg({ type: "dubber:state" });
  render(state);
}

async function startLive() {
  $("liveBtn").disabled = true;
  await tabMsg({ type: "dubber:stop" }); // never dub the same tab twice
  chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id }, async (streamId) => {
    if (chrome.runtime.lastError || !streamId) {
      live = { active: true, status: "error", message: (chrome.runtime.lastError && chrome.runtime.lastError.message) || "Не удалось захватить звук вкладки" };
      renderLive();
      $("liveBtn").disabled = false;
      return;
    }
    const r = await bg({ type: "liveStart", tabId: tab.id, streamId, opts: settings });
    if (!r.ok) {
      live = { active: true, status: "error", message: r.error || "Ошибка" };
      renderLive();
    }
    $("liveBtn").disabled = false;
    await refresh();
  });
}

async function stopLive() {
  await bg({ type: "liveStop" });
  await refresh();
}

async function start() {
  $("startBtn").disabled = true;
  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
  } catch (e) {
    injectable = false;
    render(null);
    return;
  }
  await tabMsg({ type: "dubber:start", opts: settings });
  await refresh();
}

async function stop() {
  await tabMsg({ type: "dubber:stop" });
  await refresh();
}

async function init() {
  await loadSettings();
  [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  injectable = !!tab && /^https?:/i.test(tab.url || "");
  renderSettings();

  $("startBtn").addEventListener("click", start);
  $("stopBtn").addEventListener("click", stop);
  $("liveBtn").addEventListener("click", startLive);
  $("liveStopBtn").addEventListener("click", stopLive);
  $("voice").addEventListener("change", () => { settings.voice = $("voice").value; saveSettings(); });
  $("orig").addEventListener("input", () => { settings.origVol = $("orig").value / 100; renderSettings(); saveSettings(); });
  $("dub").addEventListener("input", () => { settings.dubVol = $("dub").value / 100; renderSettings(); saveSettings(); });
  $("autopause").addEventListener("change", () => { settings.autopause = $("autopause").checked; saveSettings(); });

  await refresh();
  renderSettings();
  setInterval(refresh, 1000);
}

init();
