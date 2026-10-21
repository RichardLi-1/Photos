// Hosted-demo visitor tracking (src/visits.ts): referral classification and what a card may contain.
import assert from "node:assert/strict";
import { test } from "node:test";

import { classify, scrubUrl, visitCard } from "../src/visits.ts";

const MAC = { path: "/", url: "https://photos-demo.fly.dev/", referrer: "", platform: "MacIntel",
  ua: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15" };

test("classify: short params, then utm_source, then AI referrer, then stored", () => {
  assert.equal(classify("?l", "", "Email").source, "LinkedIn");
  assert.equal(classify("?g=1&utm_source=resume", "", null).source, "Resume");  // utm wins over a short param
  assert.equal(classify("?utm_source=richardli.dev&utm_campaign=photos&utm_content=chatgpt.com", "", null).source,
    "richardli.dev (originally chatgpt.com, via the photos page)");
  assert.equal(classify("?utm_source=chatgpt.com", "", null).source, "ChatGPT");
  const ref = classify("", "https://www.perplexity.ai/", "LinkedIn");
  assert.deepEqual([ref.source, ref.ai, ref.fresh], ["Perplexity", "Perplexity", true]);
  const back = classify("", "", "ChatGPT");  // a returning visitor keeps the source but isn't AI traffic forever
  assert.deepEqual([back.source, back.ai, back.fresh], ["ChatGPT", null, false]);
});

test("classify: ?m opts out; the app's own params are never a referral", () => {
  const l = classify("?m&q=beach+at+night&stripes", "", null);
  assert.equal(l.optOut, true);
  assert.equal(l.params.toString(), "");
  assert.equal(visitCard(l, MAC).webhookType, "regular_visit");
});

test("visitCard: routing and labels match Transit", () => {
  assert.equal(visitCard(classify("", "", null), MAC).event, "👀 New visitor on /");
  assert.equal(visitCard(classify("?l", "", null), MAC).webhookType, "referral_visit");
  const ai = visitCard(classify("?utm_source=chatgpt.com", "", null), MAC);
  assert.equal(ai.webhookType, "ai_referral_visit");
  assert.equal(ai.meta["🤖 AI assistant"], "ChatGPT (no referrer — often the mobile app)");
  const custom = visitCard(classify("?promo=spring", "", null), MAC);
  assert.equal(custom.event, "👀 New visitor — CUSTOM REFERRAL from **SPRING**");
  assert.equal(custom.webhookType, "referral_visit");
  assert.equal(visitCard(classify("", "", null), MAC).meta["🖥️ Device"], "🖥️ Desktop · macOS");
});

test("visitCard: a remembered referral says so; a bounce is the same card", () => {
  const fresh = visitCard(classify("?r", "", null), { ...MAC, url: "https://photos-demo.fly.dev/?r" });
  assert.equal(fresh.event, "👀 New visitor from **Resume**");
  assert.equal(fresh.meta["🔁 Referral"], undefined);
  assert.equal(fresh.meta["🔗 Params"], "?r");
  const back = classify("", "", "Resume", "2026-10-20T15:00:00Z");
  assert.equal(back.since, "2026-10-20T15:00:00Z");
  const card = visitCard(back, { ...MAC, timeZone: "America/Toronto" });
  assert.equal(card.event, "👀 Returning visitor from **Resume**");
  assert.match(card.meta["🔁 Referral"], /^Resume, remembered from an earlier visit on .+ \(this link had none\)$/);
  assert.match(card.meta["🕒 Their clock"], /\(America\/Toronto\)$/);
  assert.equal(card.meta["🛤️ Path"], undefined);
  assert.equal(classify("", "", "Resume").since, null);
  assert.equal(visitCard(classify("", "", "Resume"), MAC).meta["🔁 Referral"], "Resume, remembered from an earlier visit (this link had none)");
  const bounce = visitCard(back, MAC, 1400);
  assert.equal(bounce.event, "👋 Returning visitor from **Resume** left after 1s");
  assert.equal(bounce.meta["⏱️ Time on page"], "1s");
  assert.equal(visitCard(classify("", "", null), MAC, 3000).event, "👋 New visitor on / left after 3s");
});

test("scrubUrl / visitCard: no search text or photo ids leave the page", () => {
  assert.equal(scrubUrl("https://x.dev/?q=my+secret"), "https://x.dev/");
  assert.equal(scrubUrl("https://x.dev/?l&q=a&utm_source=b#h"), "https://x.dev/?l&utm_source=b#h");
  assert.equal(scrubUrl("https://x.dev/?utm_source=b&q=a#h"), "https://x.dev/?utm_source=b#h");
  assert.equal(scrubUrl("/photo/123?w=384"), "/photo/:id?w=384");
  const card = visitCard(classify("?q=my+secret&l", "https://www.google.com/search?q=other+secret", null),
    { ...MAC, url: "https://photos-demo.fly.dev/?q=my+secret&l", referrer: "https://www.google.com/search?q=other+secret" });
  assert.doesNotMatch(JSON.stringify(card), /secret/);
});
