"""Visitor analytics for the hosted demo only (ported from Transit Planner's docs/visitor-tracking.md). Stdlib only.

The local app (search/server.py), iOS, Android and desktop never import this; here every piece is a no-op until its
env var is set. Webhook URLs stay server-side: the browser only ever calls /api/track.
  * Human visits: web/src/visits.ts posts a card (referral source, device, landing URL) plus raw human signals to
    /api/track; report_visit adds a 0-100 human likelihood, the client IP and a rough location, and posts off the
    request path.
  * AI fetchers (ChatGPT-User, GPTBot, ClaudeBot, ...) run no JavaScript, so the middleware spots them by User-Agent:
    report_ai_fetch writes one JSON log line per hit, and posts to Discord only for "user" fetches.
Location: Fly sends no visitor geo-IP headers. It routes each visitor to its nearest server and names that server's
region in fly-region, so the card shows that city as a rough location. No geo-IP service is called.

Env (optional; unset or "" = off):
  DISCORD_WEBHOOK_URL   every card goes to this one webhook; AI traffic is teal, everything else blurple
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

TEAL, BLURPLE = 0x14B8A6, 0x5865F2  # teal = AI traffic (visits and fetches), blurple = everything else
MAX_FIELDS, MAX_NAME, MAX_VALUE, MAX_TITLE = 12, 64, 400, 256  # well inside Discord's embed limits (25 / 256 / 1024 / 6000)

# ---------------------------------------------------------------- AI assistants fetching pages themselves
# kind: "user" = fetched live because a person asked the assistant something just now (rare, worth a Discord card);
# "search" = indexing to cite later; "training" = collecting training data. Tokens from each vendor's crawler docs.
# Humans arriving FROM an assistant (utm_source / document.referrer) are matched in the browser: web/src/visits.ts.
AI_FETCHERS = (
    ("ChatGPT-User", "OpenAI", "user"),
    ("OAI-SearchBot", "OpenAI", "search"),
    ("GPTBot", "OpenAI", "training"),
    ("Claude-User", "Anthropic", "user"),
    ("Claude-SearchBot", "Anthropic", "search"),
    ("ClaudeBot", "Anthropic", "training"),
    ("Perplexity-User", "Perplexity", "user"),
    ("PerplexityBot", "Perplexity", "search"),
)
SKIP_PATHS = ("/assets/", "/photo/", "/favicon", "/robots.txt", "/sitemap.xml", "/healthz")  # like Transit's matcher


def match_ai_fetcher(user_agent: str | None) -> tuple[str, str, str] | None:
    """(token, vendor, kind) of the first AI_FETCHERS token found in the User-Agent, else None."""
    for f in AI_FETCHERS:
        if user_agent and f[0] in user_agent:
            return f
    return None


# ---------------------------------------------------------------- Fly server regions (rough visitor location)
FLY_REGIONS = {
    "ams": "Amsterdam, Netherlands", "arn": "Stockholm, Sweden", "atl": "Atlanta, US", "bog": "Bogotá, Colombia",
    "bom": "Mumbai, India", "bos": "Boston, US", "cdg": "Paris, France", "den": "Denver, US", "dfw": "Dallas, US",
    "ewr": "Secaucus NJ, US", "eze": "Buenos Aires, Argentina", "fra": "Frankfurt, Germany", "gdl": "Guadalajara, Mexico",
    "gig": "Rio de Janeiro, Brazil", "gru": "São Paulo, Brazil", "hkg": "Hong Kong", "iad": "Ashburn VA, US",
    "jnb": "Johannesburg, South Africa", "lax": "Los Angeles, US", "lhr": "London, UK", "mad": "Madrid, Spain",
    "mia": "Miami, US", "nrt": "Tokyo, Japan", "ord": "Chicago, US", "otp": "Bucharest, Romania", "phx": "Phoenix, US",
    "qro": "Querétaro, Mexico", "scl": "Santiago, Chile", "sea": "Seattle, US", "sin": "Singapore", "sjc": "San Jose, US",
    "syd": "Sydney, Australia", "waw": "Warsaw, Poland", "yul": "Montreal, Canada", "yyz": "Toronto, Canada",
}


def approx_location(region: str | None) -> str | None:
    """Fly routes each visitor to its nearest server, so that server's city is a rough guess at theirs."""
    r = (region or "").strip().lower()[:8]
    if not r:
        return None
    return f"{FLY_REGIONS.get(r, r)} area (nearest Fly server: {r})"


# ---------------------------------------------------------------- human likelihood (ported from firefighter-hazard)
# 0-100: start at 50, add points for signs of a person (mouse moved, scrolled, touched) and subtract them for signs of
# automation (webdriver flag, headless, missing headers). Each rule records why, so the card shows the reasons and the
# weights are easy to tune by eye. Hand-picked, not trained: there is no labelled data. web/src/visits.ts only collects
# the raw signals; the request headers (which only the server sees) feed the same score.
QUIET_PENALTY_AFTER_MS = 5_000  # only call "no input" suspicious after watching this long


def score_human(signals: dict, headers) -> tuple[int, str, list[str]]:
    """(score, label, reasons) from the browser's raw signals + this /api/track request's headers. Missing or
    malformed signals count as unknown, never as evidence."""
    points, reasons = 50, []

    def add(delta: int, why: str):
        nonlocal points
        points += delta
        reasons.append(f"{delta:+d} {why}")

    def num(k: str) -> float | None:
        v = signals.get(k)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    ua = headers.get("user-agent") or ""
    # traces that automation tools leave behind
    if re.search(r"bot|crawler|spider", ua, re.I):
        add(-60, "bot user-agent")
    if re.search(r"HeadlessChrome|PhantomJS|Lighthouse", ua, re.I):
        add(-50, "headless browser user-agent")
    if signals.get("webdriver") is True:
        add(-50, "webdriver flag (automated browser)")
    if num("languagesCount") == 0:
        add(-15, "no browser languages")
    if num("screenWidth") == 0 or num("screenHeight") == 0:
        add(-20, "0px screen")
    if num("outerWidth") == 0 and num("outerHeight") == 0:
        add(-15, "no window frame")
    if signals.get("isMobile") is False and num("pluginsCount") == 0:  # desktop browsers all list a PDF viewer
        add(-10, "no plugins on desktop")
    # real people almost always move, scroll, touch or type
    moves, touches, scrolls, keys = (num(k) or 0 for k in ("pointerMoves", "touches", "scrolls", "keyPresses"))
    if moves >= 5:
        add(25, "mouse moved")
    elif moves > 0:
        add(10, "slight mouse movement")
    if touches > 0:
        add(25, "touched screen")
    if scrolls > 0:
        add(15, "scrolled")
    if keys > 0:
        add(10, "pressed keys")
    earlier = signals.get("interactedEarlier") is True
    if earlier:
        add(20, "interacted earlier this tab session")
    ms = num("msObserved") or 0
    if not (moves or touches or scrolls or keys or earlier) and ms >= QUIET_PENALTY_AFTER_MS:
        add(-10, f"no input in {round(ms / 1000)}s")  # small: reading without touching anything is possible
    if signals.get("wasVisible") is False:
        add(-10, "tab never visible")  # also a Cmd+click background tab, so only a hint
    # headers real browsers send by themselves (/api/track already requires Sec-Fetch-Site)
    if not headers.get("accept-language"):
        add(-20, "no Accept-Language header")
    ch = headers.get("sec-ch-ua")
    if ch and "HeadlessChrome" in ch:
        add(-50, "headless Chrome in sec-ch-ua")
    elif re.search(r"Chrome/\d", ua) and not ch:  # every Chromium browser sends sec-ch-ua; Firefox/Safari don't
        add(-15, "claims Chrome but no sec-ch-ua")
    score = max(0, min(100, points))
    return score, "likely human" if score >= 70 else "unsure" if score >= 40 else "likely bot", reasons


# ---------------------------------------------------------------- routing
def webhook() -> str | None:
    return (os.environ.get("DISCORD_WEBHOOK_URL") or "").strip() or None


def enabled() -> bool:
    return webhook() is not None


def visit_type(event: str, meta: dict, webhook_type: str | None) -> str:
    """The client's webhookType wins; otherwise a card with params or a "from **X**" label is a referral."""
    if webhook_type in ("ai_referral_visit", "referral_visit"):
        return webhook_type
    if meta.get("🔗 Params") or "?" in (meta.get("🔗 URL") or ""):
        return "referral_visit"
    if re.search(r"from \*\*.+\*\*|CUSTOM REFERRAL", event, re.I):
        return "referral_visit"
    return "regular_visit"


# ---------------------------------------------------------------- Discord, off the request path
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="visits")
_slots = threading.BoundedSemaphore(64)  # Discord slow or down: drop cards rather than queue without bound


def _send(url: str, embed: dict):
    try:
        body = json.dumps({"embeds": [embed], "allowed_mentions": {"parse": []}}).encode()
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": "photos-demo/1.0"})
        urllib.request.urlopen(req, timeout=5).close()
    except Exception as e:  # best effort: never surfaces to the visitor, and never prints the URL
        print(f"[visits] Discord webhook failed: {type(e).__name__}", file=sys.stderr, flush=True)
    finally:
        _slots.release()


def post(url: str, embed: dict):
    """Queue one embed for a webhook and return at once."""
    if _slots.acquire(blocking=False):
        _pool.submit(_send, url, embed)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(v, n: int) -> str:
    s = str(v)
    return s if len(s) <= n else s[:n - 1] + "…"


def report_visit(body: dict, ip: str | None, region: str | None, headers=None) -> str | None:
    """/api/track body {event, meta?, webhookType?, humanSignals?} -> a Discord card. Returns the visit type posted,
    or None. headers: the /api/track request's, for the human score."""
    event = body.get("event")
    meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
    if not isinstance(event, str) or not event.strip():
        return None
    meta = {_clip(k, MAX_NAME): _clip(v, MAX_VALUE) for k, v in list(meta.items())[:MAX_FIELDS]
            if isinstance(k, str) and isinstance(v, str) and k and v}
    wt = body.get("webhookType")
    kind = visit_type(event, meta, wt if isinstance(wt, str) else None)
    url = webhook()
    if not url:
        return None
    fields = [{"name": k, "value": v, "inline": True} for k, v in meta.items()]
    signals = body.get("humanSignals")
    if isinstance(signals, dict):
        score, label, reasons = score_human(signals, headers or {})
        fields += [{"name": "🤖 Human likelihood" if label == "likely bot" else "🧍 Human likelihood",
                    "value": f"**{score}/100** ({label})", "inline": True},
                   {"name": "🧾 Why", "value": _clip(", ".join(reasons) or "no signals either way", MAX_VALUE),
                    "inline": False}]
    loc = approx_location(region)
    fields += [{"name": k, "value": v, "inline": True}
               for k, v in (("🌐 IP", ip), ("📍 Rough location", loc)) if v]
    post(url, {"title": _clip(event, MAX_TITLE), "color": TEAL if kind == "ai_referral_visit" else BLURPLE,
               "fields": fields, "timestamp": _now()})
    return kind


def report_ai_fetch(user_agent: str | None, path: str, local: bool) -> tuple[str, str, str] | None:
    """Middleware hook: log every AI fetch as one JSON line; Discord only for "user" fetches. Never blocks.
    Logs the path only (no query string: it could carry search text)."""
    f = match_ai_fetcher(user_agent)
    if not f or path.startswith(SKIP_PATHS):
        return None
    token, vendor, kind = f
    print(json.dumps({"type": "ai_fetch", "vendor": vendor, "bot": token, "kind": kind, "page": path}), flush=True)
    url = webhook() if kind == "user" and not local else None
    if url:
        post(url, {"title": _clip(f"🔎 {token} read {path} for someone", MAX_TITLE), "color": TEAL,
                   "fields": [{"name": "🤖 Assistant", "value": vendor, "inline": True},
                              {"name": "📄 Page", "value": _clip(path, MAX_VALUE), "inline": True},
                              {"name": "🕒 Time", "value": _now(), "inline": True}],
                   "timestamp": _now()})
    return f
