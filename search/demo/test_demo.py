"""Model-free tests for the demo pieces.  run (from search/): .venv/bin/python -m pytest demo/test_demo.py -q"""
import io
import json
import os
import sys
import tarfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.join(os.path.dirname(HERE), "experiments")]

from engine import QMatrix, pca_basis, quantize_cols  # noqa: E402
from textenc import GemmaBPE, TokenTable, clean, write_pruned_q8, write_q4, write_q8  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
SGTK = os.path.join(ROOT, "android", "app", "src", "main", "assets", "models", "siglip2_tokenizer.bin")
FIXTURES = os.path.join(ROOT, "android", "app", "src", "test", "resources", "tokenizer_fixtures.json")


def unit(n, d, seed=0):
    X = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_qmatrix_matches_dense():
    X, v = unit(500, 64), unit(1, 64, 1)[0]
    q, s = quantize_cols(X)
    for M in (QMatrix(q, s, chunk=97), QMatrix(q, s, dense=True)):
        assert np.allclose(M @ v, X @ v, atol=0.02)
        assert M[3].shape == (64,) and M[[1, 2]].shape == (2, 64)
        assert np.allclose(M[[1, 2]] @ v, (M @ v)[[1, 2]], atol=1e-5)
        assert (M @ np.stack([v, v], 1)).shape == (500, 2)


def test_qmatrix_pca_reconstructs():
    X = unit(800, 32)
    B, mu = pca_basis(X, 32)
    q, s = quantize_cols((X - mu) @ B.T)
    M = QMatrix(q, s, B, mu)
    v = X[0]
    assert M.shape == (800, 32)
    assert np.corrcoef(M @ v, X @ v)[0, 1] > 0.99
    assert np.allclose(M[5], X[5], atol=0.05)


@pytest.mark.parametrize("fmt", ["q8", "pruned", "q4"])
def test_token_tables_roundtrip(tmp_path, fmt):
    W = np.random.default_rng(0).standard_normal((300, 64)).astype(np.float32)
    p = str(tmp_path / "t.bin")
    keep = np.arange(0, 300, 2)
    {"q8": lambda: write_q8(p, W), "pruned": lambda: write_pruned_q8(p, W, keep, fallback=0),
     "q4": lambda: write_q4(p, W, group=16)}[fmt]()
    T = TokenTable(p)
    ids = np.array([[0, 2, 4], [6, 8, 298]])
    rel = np.linalg.norm(T(ids) - W[ids], axis=-1) / np.linalg.norm(W[ids], axis=-1)
    assert rel.max() < (0.15 if fmt == "q4" else 0.02)
    if fmt == "pruned":  # a pruned id reads the fallback row
        assert np.allclose(T(np.array([1])), T(np.array([0])))


def test_clean_matches_open_clip_rules():
    assert clean("  Sunset over the BEACH!  ") == "sunset over the beach"
    assert clean("Tom &amp; Jerry_x") == "tom jerry x"


@pytest.mark.skipif(not (os.path.exists(SGTK) and os.path.exists(FIXTURES)), reason="android tokenizer assets not built")
def test_bpe_matches_open_clip_fixtures():
    bpe = GemmaBPE(SGTK)
    for f in json.load(open(FIXTURES)):
        assert bpe.encode(clean(f["text"])) == f["ids"], f["text"]


def test_tar_stream_parser_handles_long_names():
    import fetch_pass
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as t:
        for name, data in (("PASS_dataset/5/abc.jpg", b"x" * 700), ("d/" + "y" * 150 + ".jpg", b"z" * 10)):
            ti = tarfile.TarInfo(name); ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    raw = buf.getvalue()

    class Fake:
        pos = 0

        def read(self, n):
            b = raw[self.pos:self.pos + n]; self.pos += len(b); return b
    s = Fake()
    got = []
    for off, name, size, pad in fetch_pass.tar_members(s):
        got.append((name, s.read(size + pad)[:size]))
    assert got == [("PASS_dataset/5/abc.jpg", b"x" * 700), ("d/" + "y" * 150 + ".jpg", b"z" * 10)]


# ---------------------------------------------------------------- hosted-demo analytics (visits.py + demo_server guards)
import visits  # noqa: E402

AI_ENV = ("DISCORD_WEBHOOK_URL", "DISCORD_REGULAR_VISITS_WEBHOOK_URL", "DISCORD_REFERRAL_VISITS_WEBHOOK_URL",
          "DISCORD_AI_VISITS_WEBHOOK_URL")
SAME = {"sec-fetch-site": "same-origin"}


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """demo_server on a fake bundle: a tiny sqlite + a stub searcher, so no model or real bundle is loaded."""
    import sqlite3
    import types

    import engine
    import photosearch.explain
    from photosearch import tools
    d = tmp_path_factory.mktemp("bundle")
    (d / "manifest.json").write_text('{"publishable": true}')
    db = sqlite3.connect(d / "library.sqlite")
    db.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, taken REAL, ocr_done INT, ocr TEXT, place TEXT, album TEXT)")
    db.execute("INSERT INTO photos VALUES (1, 0, 1, NULL, 'Kyoto', 'demo')")
    db.commit()
    web = tmp_path_factory.mktemp("web")
    (web / "index.html").write_text("<!doctype html><body></body>")
    S = types.SimpleNamespace(ids=[1], Z=None, X=None, embed_prompted=lambda t: None,
                              lib=types.SimpleNamespace(db=sqlite3.connect(d / "library.sqlite", check_same_thread=False),
                                                        row_of={1: 0}, encoder=types.SimpleNamespace(
                                                            text=types.SimpleNamespace(_cache={}))))
    with pytest.MonkeyPatch.context() as mp:
        for k in AI_ENV:
            mp.delenv(k, raising=False)
        mp.setenv("DEMO_BUNDLE", str(d))
        mp.setenv("DEMO_WEB", str(web))
        mp.setenv("DEMO_ALLOWED_HOSTS", "testserver")
        mp.setenv("DEMO_CLIENT_IP_HEADER", "fly-client-ip")
        mp.setattr(engine, "open_bundle", lambda *a, **k: S)
        mp.setattr(photosearch.explain, "Explainer", lambda s: None)
        mp.setattr(tools, "_ctx", tools._ctx)  # restored after the module, whatever tools.init does with the stub
        mp.setattr(tools, "_candidates", tools._candidates)
        sys.modules.pop("demo_server", None)
        import demo_server
        from fastapi.testclient import TestClient
        yield demo_server, TestClient(demo_server.app)
        sys.modules.pop("demo_server", None)


@pytest.fixture
def sent(monkeypatch):
    out = []
    monkeypatch.setattr(visits, "post", lambda url, embed: out.append((url, embed)))
    for k in AI_ENV:
        monkeypatch.delenv(k, raising=False)
    return out


def test_track_is_a_noop_without_env(demo, sent):
    _, c = demo
    r = c.post("/api/track", json={"event": "👀 New visitor on /"}, headers=SAME)
    assert r.status_code == 204 and not r.content and sent == []


def test_track_routes_and_enriches(demo, sent, monkeypatch):
    _, c = demo
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook/general")
    monkeypatch.setenv("DISCORD_REFERRAL_VISITS_WEBHOOK_URL", "https://hook/referral")
    h = {**SAME, "fly-client-ip": "203.0.113.9", "fly-region": "nrt"}
    c.post("/api/track", json={"event": "👀 New visitor on /", "meta": {"🔗 URL": "https://x/"}}, headers=h)
    c.post("/api/track", json={"event": "👀 New visitor from **ChatGPT**", "webhookType": "ai_referral_visit"}, headers=h)
    c.post("/api/track", json={"event": "👀 New visitor", "meta": {"🔗 Params": "?promo"}}, headers=h)
    assert [u for u, _ in sent] == ["https://hook/general", "https://hook/referral", "https://hook/referral"]
    (_, regular), (_, ai) = sent[:2]
    assert regular["color"] == visits.BLURPLE and ai["color"] == visits.TEAL  # AI falls back to referral, stays teal
    fields = {f["name"]: f["value"] for f in regular["fields"]}
    assert fields["🌐 IP"] == "203.0.113.9" and "Tokyo" in fields["📍 Approx. location"]
    assert "approximate" in fields["📍 Approx. location"]
    monkeypatch.setenv("DISCORD_AI_VISITS_WEBHOOK_URL", "https://hook/ai")
    c.post("/api/track", json={"event": "x", "webhookType": "ai_referral_visit"}, headers=h)
    assert sent[-1][0] == "https://hook/ai"
    n = len(sent)
    c.post("/api/track", json={"event": "x"}, headers={"fly-client-ip": "203.0.113.9"})  # no Sec-Fetch-Site: not a page
    c.post("/api/track", json={"event": "x"}, headers={"sec-fetch-site": "cross-site"})
    c.post("/api/track", content=b"[1]", headers={**SAME, "content-type": "application/json"})
    c.post("/api/track", json={"event": "x"}, headers={**SAME, "host": "localhost"})
    assert len(sent) == n


def test_single_webhook_catches_everything(demo, sent, monkeypatch):
    _, c = demo
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://hook/only")
    for wt in (None, "referral_visit", "ai_referral_visit"):
        c.post("/api/track", json={"event": "x", **({"webhookType": wt} if wt else {})}, headers=SAME)
    assert [u for u, _ in sent] == ["https://hook/only"] * 3


def test_visit_type_matches_transit():
    vt = visits.visit_type
    assert vt("👀 New visitor on /", {}, None) == "regular_visit"
    assert vt("👀 New visitor on /", {}, "regular_visit") == "regular_visit"
    assert vt("👀 New visitor from **LinkedIn**", {}, None) == "referral_visit"
    assert vt("👀 New visitor — CUSTOM REFERRAL from **X**", {}, None) == "referral_visit"
    assert vt("👋 Quick bounce on /", {"🔗 URL": "https://x/?q"}, None) == "referral_visit"
    assert vt("x", {}, "ai_referral_visit") == "ai_referral_visit"


@pytest.mark.parametrize("ua,kind", [
    ("Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko); compatible; ChatGPT-User/1.0; +https://openai.com/bot", "user"),
    ("Mozilla/5.0 (compatible; Claude-User/1.0; +Claude-User@anthropic.com)", "user"),
    ("Mozilla/5.0 (compatible; Perplexity-User/1.0)", "user"),
    ("Mozilla/5.0 (compatible; OAI-SearchBot/1.0; +https://openai.com/searchbot)", "search"),
    ("Mozilla/5.0 (compatible; Claude-SearchBot/1.0)", "search"),
    ("Mozilla/5.0 (compatible; PerplexityBot/1.0)", "search"),
    ("Mozilla/5.0 (compatible; GPTBot/1.2; +https://openai.com/gptbot)", "training"),
    ("Mozilla/5.0 (compatible; ClaudeBot/1.0; +claudebot@anthropic.com)", "training"),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15", None),
])
def test_ai_fetcher_classification(demo, sent, monkeypatch, capsys, ua, kind):
    assert (visits.match_ai_fetcher(ua) or (None, None, None))[2] == kind
    monkeypatch.setenv("DISCORD_AI_VISITS_WEBHOOK_URL", "https://hook/ai")
    _, c = demo
    assert c.get("/api/stats?q=secret", headers={"user-agent": ua}).status_code == 200
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{"type": "ai_fetch"')]
    assert [x["kind"] for x in lines] == ([kind] if kind else [])
    assert all(x["page"] == "/api/stats" for x in lines)  # path only: a query string could carry search text
    assert len(sent) == (kind == "user")
    if sent:
        assert sent[0][0] == "https://hook/ai" and sent[0][1]["color"] == visits.TEAL


def test_csp_stays_same_origin(demo):
    _, c = demo
    for r in (c.get("/api/stats"), c.post("/api/track", json={"event": "x"}, headers=SAME)):
        csp = r.headers["content-security-policy"]
        assert "connect-src 'self';" in csp and "script-src 'self';" in csp, csp


def test_healthz_still_exempt_from_host_check(demo):
    _, c = demo
    assert c.get("/healthz", headers={"host": "10.0.0.5:8080"}).status_code == 200
    assert c.get("/api/stats", headers={"host": "10.0.0.5:8080"}).status_code == 421
    assert c.post("/api/track", json={"event": "x"}, headers={**SAME, "host": "evil.example"}).status_code == 421
