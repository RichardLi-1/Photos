"""Visitor analytics for the hosted demo only (ported from Transit Planner's docs/visitor-tracking.md). Stdlib only.

The local app (search/server.py), iOS, Android and desktop never import this; here every piece is a no-op until its
env var is set. Webhook URLs stay server-side: the browser only ever calls /api/track.
  * Human visits: web/src/visits.ts posts a card (referral source, device, path) to /api/track; report_visit adds
    the client IP and an approximate location, and posts off the request path.
  * AI fetchers (ChatGPT-User, GPTBot, ClaudeBot, ...) run no JavaScript, so the middleware spots them by User-Agent:
    report_ai_fetch writes one JSON log line per hit, and posts to Discord only for "user" fetches.
Location: Fly sends no visitor geo-IP headers, so the card names the Fly edge region that took the connection
(fly-region), labelled approximate. No geo-IP service is called.

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


# ---------------------------------------------------------------- Fly edge regions (approximate visitor location)
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
    r = (region or "").strip().lower()[:8]
    if not r:
        return None
    return f"near {FLY_REGIONS.get(r, r)} (Fly edge {r}; approximate, not geo-IP)"


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


def report_visit(body: dict, ip: str | None, region: str | None) -> str | None:
    """/api/track body {event, meta?, webhookType?} -> a Discord card. Returns the visit type posted, or None."""
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
    loc = approx_location(region)
    meta |= {**({"🌐 IP": ip} if ip else {}), **({"📍 Approx. location": loc} if loc else {})}
    post(url, {"title": _clip(event, MAX_TITLE), "color": TEAL if kind == "ai_referral_visit" else BLURPLE,
               "fields": [{"name": k, "value": v, "inline": True} for k, v in meta.items()], "timestamp": _now()})
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
