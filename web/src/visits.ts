// Visitor tracking for the hosted demo only (photos-demo.fly.dev), ported from Transit Planner's page-view tracker.
// startVisits() runs only once /api/stats returns a `demo` object, which the local server never does: the local app,
// iOS, Android and desktop send nothing. One card per page load (+ one for a quick bounce) goes to /api/track; the demo
// server (search/demo/visits.py) adds IP + approximate location and posts to Discord, so webhook URLs never ship here.
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

const SKIP_KEY = "skip_tracking", SKIP_PARAM = "m", REFERRAL_KEY = "referral_source";
const BOUNCE_MS = 10_000;

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
  params: URLSearchParams;
  ai: string | null;      // this landing came from an AI assistant (never the stored referral)
  referrerHost: string | null;
  optOut: boolean;        // ?m
};

/** Referral order: short params, then utm_source, then an AI referrer (only with no params), then the stored source. */
export function classify(search: string, referrer: string, stored: string | null): Landing {
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
  return { source, fresh, params, ai, referrerHost, optOut };
}

function platformOf(ua: string, platform: string) {
  return /iPhone|iPad/.test(ua) ? "iOS" : ua.includes("Android") ? "Android" : platform.includes("Mac") ? "macOS"
    : platform.includes("Win") ? "Windows" : platform.includes("Linux") ? "Linux" : "Unknown";
}

/** The page-view card. Params with no known referral are named after their first value (else key). */
export function visitCard(l: Landing, v: { path: string; url: string; referrer: string; ua: string; platform: string }): Card {
  const raw = l.params.toString();
  const bot = /bot|crawler|spider/i.test(v.ua);
  const [k, val] = l.params.entries().next().value ?? [];
  const event = bot ? `🤖 Bot/crawler on ${v.path}`
    : l.source ? `👀 New visitor from **${l.source}**`
    : raw ? `👀 New visitor — CUSTOM REFERRAL from **${(val || k || "unknown").toUpperCase()}**`
    : `👀 New visitor on ${v.path}`;
  return {
    event,
    webhookType: l.ai ? "ai_referral_visit" : l.source || raw ? "referral_visit" : "regular_visit",
    meta: {
      "🖥️ Device": `${/Mobi|Android|iPhone|iPad/i.test(v.ua) ? "📱 Mobile" : "🖥️ Desktop"} · ${platformOf(v.ua, v.platform)}`,
      "🛤️ Path": v.path,
      "🔗 URL": scrubUrl(v.url),
      "🕒 Time": new Date().toLocaleString(),
      ...(raw ? { "🔗 Params": `?${raw}` } : {}),
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
  const l = classify(LANDING.search, LANDING.referrer, get(localStorage, REFERRAL_KEY));
  if (l.optOut) set(localStorage, SKIP_KEY, "1");
  if (l.fresh && l.source) set(localStorage, REFERRAL_KEY, l.source);

  // Cosmetic and not gated by the skip rules: no tracking param stays in the address bar (App keeps its own).
  const now = new URLSearchParams(location.search);
  if ([...now.keys()].some((k) => !APP_PARAMS.includes(k))) {
    for (const k of [...now.keys()]) if (!APP_PARAMS.includes(k)) now.delete(k);
    history.replaceState(history.state, "", `${location.pathname}${now.toString() ? `?${now}` : ""}${location.hash}`);
  }

  const key = (down: boolean) => (e: KeyboardEvent) => { if (e.key === "/") slashHeld = down; };
  addEventListener("keydown", key(true));
  addEventListener("keyup", key(false));

  // Quick bounce: left within 10 s. sendBeacon survives the unload; once per tab session.
  addEventListener("pagehide", () => {
    const ms = Date.now() - LANDING.at;
    if (skipped() || /bot|crawler|spider/i.test(navigator.userAgent) || ms > BOUNCE_MS) return;
    if (get(sessionStorage, "bounce_sent")) return;
    set(sessionStorage, "bounce_sent", "1");
    const ref = get(localStorage, REFERRAL_KEY);
    const card: Card = {
      webhookType: l.ai ? "ai_referral_visit" : ref ? "referral_visit" : "regular_visit",
      event: ref ? `👋 Quick bounce from **${ref}** on ${location.pathname}` : `👋 Quick bounce on ${location.pathname}`,
      meta: { "🛤️ Path": location.pathname, "🔗 URL": scrubUrl(location.href), "⏱️ Time on page": `${Math.round(ms / 1000)}s`,
        "🕒 Time": new Date().toLocaleString() },
    };
    navigator.sendBeacon("/api/track", new Blob([JSON.stringify(card)], { type: "application/json" }));
  });

  // A beat after the listeners attach, so a held "/" (key repeat) is seen. Fire and forget.
  setTimeout(() => {
    if (skipped()) return;
    const card = visitCard(l, { path: location.pathname || "/", url: LANDING.href, referrer: LANDING.referrer,
      ua: navigator.userAgent, platform: navigator.platform });
    fetch("/api/track", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(card) })
      .catch(() => {});
  }, 300);
}
