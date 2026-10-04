// Service worker: the only place that talks to the local Dubber server.
// (Content scripts run under the page's origin, so the browser would block their requests to localhost.)

const API = "http://127.0.0.1:8765";

async function request(path, init) {
  const res = await fetch(API + path, init);
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail || ""; } catch (_) {}
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res;
}

function toBase64(buf) {
  const bytes = new Uint8Array(buf);
  let s = "";
  const STEP = 0x8000;
  for (let i = 0; i < bytes.length; i += STEP) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + STEP));
  }
  return btoa(s);
}

const handlers = {
  async health() {
    return await (await request("/health")).json();
  },
  async createJob({ url, voice }) {
    return await (await request("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, voice }),
    })).json();
  },
  async jobStatus({ id, t }) {
    const q = typeof t === "number" ? `?t=${t.toFixed(2)}` : "";
    return await (await request(`/jobs/${id}${q}`)).json();
  },
  async chunk({ id, k }) {
    const res = await request(`/jobs/${id}/chunks/${k}.wav`);
    return { b64: toBase64(await res.arrayBuffer()) };
  },
  async cancel({ id }) {
    return await (await request(`/jobs/${id}/cancel`, { method: "POST" })).json();
  },

  // ---- live mode (tab capture runs in an offscreen document)
  async liveStart({ tabId, streamId, opts }) {
    await ensureOffscreen();
    await setLive({ active: true, tabId, status: "starting", message: "Подключаюсь…", lag: 0, lang: null });
    const r = await chrome.runtime.sendMessage({ target: "offscreen", type: "live:start", streamId, opts });
    if (!r || !r.ok) throw new Error((await getLive()).message || "Не удалось запустить захват звука");
    return await getLive();
  },
  async liveStop() {
    try { await chrome.runtime.sendMessage({ target: "offscreen", type: "live:stop" }); } catch (_) {}
    await closeOffscreen();
    await setLive({ active: false, tabId: null, status: "idle", message: "" });
    return await getLive();
  },
  async liveOpts({ opts }) {
    try { await chrome.runtime.sendMessage({ target: "offscreen", type: "live:opts", opts }); } catch (_) {}
    return {};
  },
  async liveState() {
    return await getLive();
  },
};

const LIVE_IDLE = { active: false, tabId: null, status: "idle", message: "", lag: 0, lang: null };

async function getLive() {
  const s = await chrome.storage.session.get("live");
  return s.live || LIVE_IDLE;
}

async function setLive(patch) {
  const next = { ...(await getLive()), ...patch };
  await chrome.storage.session.set({ live: next });
  chrome.action.setBadgeBackgroundColor({ color: "#4f46e5" });
  chrome.action.setBadgeText({ text: next.active ? "LIVE" : "" });
  return next;
}

async function ensureOffscreen() {
  const ctxs = await chrome.runtime.getContexts({ contextTypes: ["OFFSCREEN_DOCUMENT"] });
  if (ctxs.length) return;
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["USER_MEDIA", "AUDIO_PLAYBACK"],
    justification: "Capture the tab audio and play the Russian dub",
  });
}

async function closeOffscreen() {
  const ctxs = await chrome.runtime.getContexts({ contextTypes: ["OFFSCREEN_DOCUMENT"] });
  if (ctxs.length) await chrome.offscreen.closeDocument();
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg && msg.type === "live:status") {
    const { type, ...patch } = msg;
    setLive(patch);
  }
  return false;
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const fn = msg && handlers[msg.type];
  if (!fn) return false;
  fn(msg)
    .then((data) => sendResponse({ ok: true, data }))
    .catch((e) => {
      const offline = e instanceof TypeError; // fetch() fails with TypeError when nothing listens on the port
      sendResponse({ ok: false, offline, error: offline ? "server_offline" : String(e.message || e) });
    });
  return true; // async response
});
