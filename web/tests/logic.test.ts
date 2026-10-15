// Pure-logic unit tests. Zero dependencies: Node's built-in runner + type stripping (Node >= 22.18 / 23.6).
//   npm test        (also part of `npm run check`)
import assert from "node:assert/strict";
import { test } from "node:test";

import { api, queryKey, type Query } from "../src/api.ts";
import { isQuestion } from "../src/ask.ts";

test("queryKey: text and similar queries", () => {
  assert.deepEqual(queryKey({ kind: "text", q: "beach" }), ["text", "beach"]);
  assert.deepEqual(queryKey({ kind: "similar", id: 42 }), ["similar", 42, false]);
  assert.notDeepEqual(queryKey({ kind: "similar", id: 42, place: true }), queryKey({ kind: "similar", id: 42 }));  // same-place mode caches separately
});

test("queryKey: distinct queries get distinct keys", () => {
  const qs: Query[] = [
    { kind: "text", q: "42" },
    { kind: "similar", id: 42 },
    { kind: "text", q: "beach" },
    { kind: "text", q: "Beach" },
  ];
  const keys = qs.map((q) => JSON.stringify(queryKey(q)));
  assert.equal(new Set(keys).size, keys.length);
});

test("queryKey: image queries key on file identity + text", () => {
  const f = new File([new Uint8Array([1, 2, 3])], "room.jpg", { type: "image/jpeg", lastModified: 1000 });
  const a = queryKey({ kind: "image", file: f, text: "" });
  assert.deepEqual(a, ["image", "room.jpg", 3, 1000, ""]);
  assert.notDeepEqual(queryKey({ kind: "image", file: f, text: "at night" }), a);
  const g = new File([new Uint8Array([1, 2, 3, 4])], "room.jpg", { type: "image/jpeg", lastModified: 1000 });
  assert.notDeepEqual(queryKey({ kind: "image", file: g, text: "" }), a);
});

test("api.thumb / explain build same-origin URLs", async () => {
  assert.equal(api.thumb(7), "/photo/7?w=384");
  assert.equal(api.thumb(7, 128), "/photo/7?w=128");
  const seen: string[] = [];
  const orig = globalThis.fetch;
  globalThis.fetch = (async (url: string) => {
    seen.push(String(url));
    return new Response(JSON.stringify({ mode: "x" }), { status: 200 });
  }) as typeof fetch;
  try {
    await api.explain(5, { q: "a&b=c", ref: 0, tags: ["red tones", "sky"] });
    await api.search({ kind: "text", q: "cats & dogs" });
  } finally {
    globalThis.fetch = orig;
  }
  assert.equal(seen[0], "/api/explain/5?q=a%26b%3Dc&ref=0&tags=red+tones%2Csky");  // ref=0 is kept, input is escaped
  assert.ok(seen[1].startsWith("/api/search?q=cats+%26+dogs&k="));
  assert.ok(seen.every((u) => u.startsWith("/")), "never calls another origin");
});

test("api surfaces server error detail", async () => {
  const orig = globalThis.fetch;
  globalThis.fetch = (async () => new Response(JSON.stringify({ detail: "q required" }), { status: 400 })) as typeof fetch;
  try {
    await assert.rejects(api.search({ kind: "text", q: "" }), (e: Error & { status?: number }) => e.message === "q required" && e.status === 400);
  } finally {
    globalThis.fetch = orig;
  }
});

test("isQuestion: questions go to the VLM", () => {
  for (const s of ["how many dogs?", "What did I eat in Rome", "where was this taken", "Show me sunsets", "find receipts",
    "Did we go skiing in 2022", "beach?", "  when was the wedding  "]) {
    assert.equal(isQuestion(s), true, s);
  }
});

test("isQuestion: plain searches stay fast", () => {
  for (const s of ["beach sunset", "red car", "showroom", "whatever", "finding nemo poster", "dogs in snow", ""]) {
    assert.equal(isQuestion(s), false, s);
  }
});

import { isCommand } from "../src/agent.ts";
import { dateLabel, emptyMessage, parsedChips, spanLabel } from "../src/parsed.ts";

test("isCommand: imperative library commands go to the agent", () => {
  for (const s of ["make an album of my beach photos", "album: Japan trip", "clean up duplicates", "Delete photos of receipts",
    "trash the blurry ones", "restore", "undo", "show my albums", "show trash", "Recently Deleted", "add these to Japan trip",
    "find photos I probably don't need", "what can you do", "get rid of screenshots"]) {
    assert.equal(isCommand(s), true, s);
  }
});

test("isCommand: searches and questions are left alone", () => {
  for (const s of ["beach sunset", "album cover art", "showroom", "show me sunsets", "find receipts", "how many dogs?",
    "deleted scenes poster", "red car", "", "  "]) {
    assert.equal(isCommand(s), false, s);
  }
  // "find receipts" stays a question for Ask; commands win over questions only when they're imperative
  assert.equal(isQuestion("find receipts"), true);
});

test("dateLabel: days, months, years, ranges (before is exclusive)", () => {
  assert.equal(dateLabel("2025-09-29", "2025-09-30"), "Sep 29, 2025");
  assert.equal(dateLabel("2019-01-01", "2020-01-01"), "2019");
  assert.equal(dateLabel("2018-01-01", "2021-01-01"), "2018–2020");
  assert.equal(dateLabel("2025-09-01", "2025-10-01"), "Sep 2025");
  assert.equal(dateLabel("2016-12-01", "2017-03-01"), "Dec 2016 – Feb 2017");
  assert.equal(dateLabel("2021-03-03", "2021-03-10"), "Mar 3, 2021 – Mar 9, 2021");
  assert.equal(dateLabel("2021-03-03", null), "after Mar 3, 2021");
  assert.equal(dateLabel(null, null), null);
});

test("parsedChips: only when something besides text was understood", () => {
  assert.deepEqual(parsedChips({ text: "flowers" }), []);
  assert.deepEqual(parsedChips(null), []);
  assert.deepEqual(parsedChips({ text: "flowers", place: "Germany" }), ["📍 Germany", "“flowers”"]);
  assert.deepEqual(parsedChips({ text: "", after: "2025-09-29", before: "2025-09-30" }), ["📅 Sep 29, 2025"]);
  assert.deepEqual(parsedChips({ text: "", place: "Paris|London", has_text: true }), ["📍 Paris or London", "🔤 has text"]);
});

test("emptyMessage: says what was searched and what the library spans", () => {
  const range: [string, string] = ["2014-03-02", "2025-05-20"];
  assert.equal(spanLabel(range), "2014–May 2025");
  assert.equal(emptyMessage({ text: "", after: "2025-09-29", before: "2025-09-30" }, range),
    "No photos from Sep 29, 2025 — your library spans 2014–May 2025.");
  assert.equal(emptyMessage({ text: "snow", place: "Peru" }, range), "No photos of “snow” in Peru.");
  assert.equal(emptyMessage({ text: "", after: "2030-01-01", before: "2031-01-01" }), "No photos from 2030.");
  assert.equal(emptyMessage(null), "No matches. Try different words.");
});

test("api.agent posts JSON to same-origin agent routes", async () => {
  const seen: { url: string; body: unknown; type: string | null }[] = [];
  const orig = globalThis.fetch;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    seen.push({ url: String(url), body: JSON.parse(String(init?.body)), type: new Headers(init?.headers).get("content-type") });
    return new Response(JSON.stringify({ session_id: "s", reply: "ok", steps: [] }), { status: 200 });
  }) as typeof fetch;
  try {
    await api.agent.message("clean up duplicates");
    await api.agent.confirm("s", "p");
    await api.agent.cancel("s", "p");
    await api.agent.undo("s", "u");
  } finally {
    globalThis.fetch = orig;
  }
  assert.deepEqual(seen.map((s) => s.url), ["/api/agent/message", "/api/agent/confirm", "/api/agent/cancel", "/api/agent/undo"]);
  assert.deepEqual(seen[1].body, { session_id: "s", pending_id: "p" });
  assert.ok(seen.every((s) => s.type === "application/json"));
});
