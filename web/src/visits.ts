// Visitor tracking for the hosted demo only (photos-demo.fly.dev), ported from Transit Planner's page-view tracker.
// startVisits() runs only once /api/stats returns a `demo` object, which the local server never does: the local app,
// iOS, Android and desktop send nothing. One card per page load goes to /api/track with raw human signals; the demo
// server (search/demo/visits.py) scores those, adds IP + rough location and posts to Discord, so webhook URLs never
// ship here.
// Never sent: search text (the app keeps it in ?q=), uploaded photos, photo IDs, agent messages.

// Referral links: ?l, ?r, ... (any value). Edit freely.
const REFERRAL_SOURCES: Record<string, string> = {
  l: "LinkedIn",
  r: "Resume",
  t: "Twitter/X",
  e: "Email",
  g: "GitHub",
  c: "Cover Letter",
};
// ?utm_source=<value> links, keyed by the lowercased value. Unlisted values are shown as they are.
const UTM_SOURCE_LABELS: Record<string, string> = { resume: "Resume" };
// Humans arriving FROM an AI assistant: utm_source value or document.referrer host ("www." stripped).
const AI_ASSISTANT_HOSTS: Record<string, string> = {
  "chatgpt.com": "ChatGPT",
  "chat.openai.com": "ChatGPT",
  "perplexity.ai": "Perplexity",
  "claude.ai": "Claude",
  "gemini.google.com": "Gemini",
  "copilot.microsoft.com": "Copilot",
};
// The app's own URL state (App.tsx): search text and debug flags. Never a referral, never reported.
const APP_PARAMS = ["q", "stripes", "noglass"];

const SKIP_KEY = "skip_tracking", SKIP_PARAM = "m", REFERRAL_KEY = "referral_source", REFERRAL_AT_KEY = "referral_source_at";
const HUMAN_SEEN_KEY = "human_seen";  // sessionStorage: this tab already showed real input
const BOUNCE_MS = 10_000;
const VISIT_DELAY_MS = 5_000;  // give the human score a few seconds of input before sending

export type VisitType = "regular_visit" | "referral_visit" | "ai_referral_visit";
export type Card = { event: string; meta: Record<string, string>; webhookType: VisitType };

// The landing URL as the visitor opened it: App rewrites the address bar on mount, before /api/stats answers.
const LANDING = typeof window === "undefined" ? null
  : { href: location.href, search: location.search, referrer: document.referrer, at: Date.now() };

export function aiAssistantFromHost(hostOrSource: string | null | undefined): string | null {
  if (!hostOrSource) return null;
  return AI_ASSISTANT_HOSTS[hostOrSource.toLowerCase().replace(/^www\./, "")] ?? null;
}

/** Drops ?q=<search text> and /photo/<id> from any URL-ish string. */
export function scrubUrl(s: string): string {
  return s.replace(/([?&])q=[^&#]*(&|(?=#)|$)/g, (_, lead: string, tail: string) => (tail === "&" ? lead : ""))
    .replace(/\/photo\/\d+/g, "/photo/:id");
}

// "?utm_source=richardli.dev&utm_campaign=photos&utm_content=chatgpt.com" -> "richardli.dev (originally chatgpt.com,
// via the photos page)". utm_medium is left out (always "referral").
function describeUtm(p: URLSearchParams): string | null {
  const source = p.get("utm_source");
  if (!source) return null;
  const label = UTM_SOURCE_LABELS[source.toLowerCase()] ?? aiAssistantFromHost(source) ?? source;
  const details = [p.get("utm_content") && `originally ${p.get("utm_content")}`,
    p.get("utm_campaign") && `via the ${p.get("utm_campaign")} page`].filter(Boolean);
  return details.length ? `${label} (${details.join(", ")})` : label;
}

export type Landing = {
  source: string | null;  // referral to report (fresh, or the one stored from an earlier visit)
  fresh: boolean;         // this landing set it: store it
  since: string | null;   // not fresh: when this browser first arrived with it (ISO), if known
  params: URLSearchParams;
  ai: string | null;      // this landing came from an AI assistant (never the stored referral)
  referrerHost: string | null;
  optOut: boolean;        // ?m
};

/** Referral order: short params, then utm_source, then an AI referrer (only with no params), then the stored source. */
export function classify(search: string, referrer: string, stored: string | null, storedAt: string | null = null): Landing {
  const params = new URLSearchParams(search);
  const optOut = params.has(SKIP_PARAM);
  for (const k of [SKIP_PARAM, ...APP_PARAMS]) params.delete(k);
  let source = stored, fresh = false;
  const short = [...params.keys()].find((k) => REFERRAL_SOURCES[k]);
  if (short) { source = REFERRAL_SOURCES[short]; fresh = true; }
  const utm = describeUtm(params);
  if (utm) { source = utm; fresh = true; }
  let referrerHost: string | null = null;
  try { referrerHost = referrer ? new URL(referrer).hostname : null; } catch { /* malformed: none */ }
  const ai = aiAssistantFromHost(params.get("utm_source")) ?? aiAssistantFromHost(referrerHost);
  if (ai && !params.toString()) { source = ai; fresh = true; }
  return { source, fresh, since: fresh || !source ? null : storedAt, params, ai, referrerHost, optOut };
}

function platformOf(ua: string, platform: string) {
  return /iPhone|iPad/.test(ua) ? "iOS" : ua.includes("Android") ? "Android" : platform.includes("Mac") ? "macOS"
    : platform.includes("Win") ? "Windows" : platform.includes("Linux") ? "Linux" : "Unknown";
}

export type VisitInfo = { path: string; url: string; referrer: string; ua: string; platform: string; timeZone?: string };

/** The page-view card, or with bounceMs the one card for a visitor who left within BOUNCE_MS. Params with no known
 *  referral are named after their first value (else key). */
export function visitCard(l: Landing, v: VisitInfo, bounceMs?: number): Card {
  const raw = l.params.toString();
  const bot = /bot|crawler|spider/i.test(v.ua);
  const [k, val] = l.params.entries().next().value ?? [];
  const who = l.source ? `${l.fresh ? "New" : "Returning"} visitor from **${l.source}**`
    : raw ? `New visitor — CUSTOM REFERRAL from **${(val || k || "unknown").toUpperCase()}**`
    : `New visitor on ${v.path}`;
  const event = bot ? `🤖 Bot/crawler on ${v.path}`
    : bounceMs === undefined ? `👀 ${who}` : `👋 ${who} left after ${Math.round(bounceMs / 1000)}s`;
  const since = l.since && !Number.isNaN(Date.parse(l.since))
    ? ` on ${new Date(l.since).toLocaleDateString(undefined, { month: "short", day: "numeric" })}` : "";
  return {
    event,
    webhookType: l.ai ? "ai_referral_visit" : l.source || raw ? "referral_visit" : "regular_visit",
    meta: {
      "🖥️ Device": `${/Mobi|Android|iPhone|iPad/i.test(v.ua) ? "📱 Mobile" : "🖥️ Desktop"} · ${platformOf(v.ua, v.platform)}`,
      "🔗 URL": scrubUrl(v.url),
      // a stored source: say so, or the card reads "from Resume" next to a URL with no ?r
      ...(l.source && !l.fresh ? { "🔁 Referral": `${l.source}, remembered from an earlier visit${since} (this link had none)` } : {}),
      "🕒 Their clock": `${new Date().toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}${v.timeZone ? ` (${v.timeZone})` : ""}`,
      ...(bounceMs !== undefined ? { "⏱️ Time on page": `${Math.round(bounceMs / 1000)}s` } : {}),
      ...(raw ? { "🔗 Params": `?${raw.replace(/=(?=&|$)/g, "")}` } : {}),  // "?r", as in the link, not "?r="
      ...(v.referrer ? { "↩️ Referrer": scrubUrl(v.referrer) } : {}),
      // ChatGPT's app often sends no referrer, its web UI does
      ...(l.ai ? { "🤖 AI assistant": `${l.ai} (${l.referrerHost ? "web, referrer sent" : "no referrer — often the mobile app"})` } : {}),
      ...(bot ? { "🔍 UA": v.ua } : {}),
    },
  };
}

function get(s: Storage, k: string) { try { return s.getItem(k); } catch { return null; } }
function set(s: Storage, k: string, v: string) { try { s.setItem(k, v); } catch { /* private mode */ } }

let started = false;
let slashHeld = false;  // holding "/" while the page loads skips tracking (testing on the live URL)

function skipped() {
  return slashHeld || ["localhost", "127.0.0.1"].includes(location.hostname) || !!get(localStorage, SKIP_KEY);
}

/** Call once /api/stats has a `demo` object. Idempotent. */
export function startVisits() {
  if (started || !LANDING) return;
  started = true;
  const l = classify(LANDING.search, LANDING.referrer, get(localStorage, REFERRAL_KEY), get(localStorage, REFERRAL_AT_KEY));
  if (l.optOut) set(localStorage, SKIP_KEY, "1");
  if (l.fresh && l.source) {
    set(localStorage, REFERRAL_KEY, l.source);
    set(localStorage, REFERRAL_AT_KEY, new Date(LANDING.at).toISOString());
  }

  // Cosmetic and not gated by the skip rules: no tracking param stays in the address bar (App keeps its own).
  const now = new URLSearchParams(location.search);
  if ([...now.keys()].some((k) => !APP_PARAMS.includes(k))) {
    for (const k of [...now.keys()]) if (!APP_PARAMS.includes(k)) now.delete(k);
    history.replaceState(history.state, "", `${location.pathname}${now.toString() ? `?${now}` : ""}${location.hash}`);
  }

  const key = (down: boolean) => (e: KeyboardEvent) => { if (e.key === "/") slashHeld = down; };
  addEventListener("keydown", key(true));
  addEventListener("keyup", key(false));

  const signals = startHumanSignals();
  const info: VisitInfo = { path: location.pathname || "/", url: LANDING.href, referrer: LANDING.referrer,
    ua: navigator.userAgent, platform: navigator.platform, timeZone: timeZone() };
  // sendBeacon survives the page unloading, where a fetch would be cancelled
  const send = (card: Card) => navigator.sendBeacon("/api/track",
    new Blob([JSON.stringify({ ...card, humanSignals: signals() })], { type: "application/json" }));
  // Decided a beat after the listeners attach, so a held "/" (key repeat) is seen, then kept: by the time the card
  // goes out the key is up again, and typing "/" later must not cancel it.
  let skip: boolean | null = null;
  setTimeout(() => { skip = skipped(); }, 300);

  // One card per visit: after VISIT_DELAY_MS, so the human score has some input to judge, or when the page goes away.
  // Gone within BOUNCE_MS: that same card says so ("left after 3s"); after the visit card: a second one, once per tab.
  let visitSent = false;
  const sendVisit = (bounceMs?: number) => {
    if (visitSent || (skip ?? skipped())) return;
    visitSent = true;
    send(visitCard(l, info, bounceMs));
  };
  setTimeout(() => sendVisit(), VISIT_DELAY_MS);
  addEventListener("pagehide", () => {
    const ms = Date.now() - LANDING.at;
    const bounced = ms <= BOUNCE_MS && !/bot|crawler|spider/i.test(navigator.userAgent) && !get(sessionStorage, "bounce_sent");
    if (bounced) set(sessionStorage, "bounce_sent", "1");
    if (!visitSent) sendVisit(bounced ? ms : undefined);
    else if (bounced && !skip) send(visitCard(l, info, ms));
  });
}

function timeZone() {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone || undefined; } catch { return undefined; }
}

// Raw facts for the human-likelihood score; search/demo/visits.py scores them together with the request headers.
// Ported from firefighter-hazard (src/lib/track/human-score.ts).
export type HumanSignals = {
  webdriver: boolean;  // true when Selenium / Puppeteer / Playwright drive the browser
  languagesCount: number; screenWidth: number; screenHeight: number;
  outerWidth: number; outerHeight: number;  // 0 in many headless browsers: no real window around the page
  pluginsCount: number; isMobile: boolean;
  pointerMoves: number; touches: number; scrolls: number; keyPresses: number;
  wasVisible: boolean;         // was the tab ever on screen?
  interactedEarlier: boolean;  // showed real input earlier this tab session
  msObserved: number;
};

/** Starts counting real input; the returned function snapshots the signals so far. */
function startHumanSignals(): () => HumanSignals {
  const startedAt = Date.now();
  const n = { pointerMoves: 0, touches: 0, scrolls: 0, keyPresses: 0 };
  let wasVisible = document.visibilityState === "visible";
  const interactedEarlier = get(sessionStorage, HUMAN_SEEN_KEY) === "1";
  // isTrusted: real input only, not events a script dispatches. "wheel"/"touchmove" rather than "scroll", which also
  // fires for scripted scrolls and scroll restoration. A single stray mouse event isn't enough to remember the tab.
  const count = (type: string, k: keyof typeof n, rememberAt = 1) => addEventListener(type, (e) => {
    if (e.isTrusted && ++n[k] >= rememberAt) set(sessionStorage, HUMAN_SEEN_KEY, "1");
  }, { passive: true });
  count("mousemove", "pointerMoves", 5);
  count("touchstart", "touches");
  count("wheel", "scrolls");
  count("touchmove", "scrolls");
  count("keydown", "keyPresses");
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") wasVisible = true; });
  return () => ({
    webdriver: navigator.webdriver === true,
    languagesCount: navigator.languages?.length ?? 0,
    screenWidth: screen?.width ?? 0, screenHeight: screen?.height ?? 0,
    outerWidth: outerWidth, outerHeight: outerHeight,
    pluginsCount: navigator.plugins?.length ?? 0,
    isMobile: /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent),
    ...n, wasVisible, interactedEarlier, msObserved: Date.now() - startedAt,
  });
}
