// What the search box understood (/api/search -> parsed), as short chips, plus a friendly empty state.
import type { Parsed } from "./api";

const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAY = 86_400_000;

const d = (iso: string) => {
  const [y, m, dd] = iso.slice(0, 10).split("-").map(Number);
  return new Date(Date.UTC(y, (m || 1) - 1, dd || 1));
};
const day = (t: Date) => `${MON[t.getUTCMonth()]} ${t.getUTCDate()}, ${t.getUTCFullYear()}`;
const month = (t: Date) => `${MON[t.getUTCMonth()]} ${t.getUTCFullYear()}`;
const monthsBetween = (a: Date, b: Date) => (b.getUTCFullYear() - a.getUTCFullYear()) * 12 + b.getUTCMonth() - a.getUTCMonth();

/** "2019", "Sep 2025", "Sep 29, 2025", "Dec 2016 – Feb 2017", "after Mar 3, 2021" ... (before is exclusive). */
export function dateLabel(after?: string | null, before?: string | null): string | null {
  if (!after && !before) return null;
  if (after && !before) return `after ${day(d(after))}`;
  if (!after && before) return `before ${day(d(before))}`;
  const a = d(after!), b = d(before!);
  const last = new Date(b.getTime() - DAY);
  const monthAligned = a.getUTCDate() === 1 && b.getUTCDate() === 1;
  if (b.getTime() - a.getTime() === DAY) return day(a);
  if (monthAligned && a.getUTCMonth() === 0 && b.getUTCMonth() === 0) {
    const n = b.getUTCFullYear() - a.getUTCFullYear();
    return n === 1 ? `${a.getUTCFullYear()}` : `${a.getUTCFullYear()}–${last.getUTCFullYear()}`;
  }
  if (monthAligned) return monthsBetween(a, b) === 1 ? month(a) : `${month(a)} – ${month(last)}`;
  return `${day(a)} – ${day(last)}`;
}

/** Chips for the status bubble: place, dates, other filters, then the remaining visual text. Empty when nothing was parsed. */
export function parsedChips(p?: Parsed | null): string[] {
  if (!p) return [];
  const chips: string[] = [];
  if (p.place) chips.push(`📍 ${p.place.split("|").join(" or ")}`);
  const dl = dateLabel(p.after, p.before);
  if (dl) chips.push(`📅 ${dl}`);
  if (p.dates?.length) {
    const ls = p.dates.map(([a, b]) => dateLabel(a, b)).filter(Boolean);
    chips.push(`📅 ${ls.slice(0, 2).join(", ")}${ls.length > 2 ? ` +${ls.length - 2}` : ""}`);
  }
  if (p.album) chips.push(`📁 ${p.album}`);
  if (p.has_text) chips.push("🔤 has text");
  if (p.near) chips.push("📌 nearby");
  if (!chips.length) return [];
  if (p.text) chips.push(`“${p.text}”`);
  return chips;
}

/** "2014–May 2025" from /api/stats date_range. */
export function spanLabel(range?: [string | null, string | null]): string | null {
  if (!range || !range[0] || !range[1]) return null;
  const a = d(range[0]), b = d(range[1]);
  return a.getUTCFullYear() === b.getUTCFullYear() ? `${month(a)} – ${month(b)}` : `${a.getUTCFullYear()}–${month(b)}`;
}

/** Friendly zero-results line, e.g. "No photos from Sep 29, 2025 — your library spans 2014–May 2025." */
export function emptyMessage(p: Parsed | null | undefined, range?: [string | null, string | null]): string {
  const when = p ? dateLabel(p.after, p.before) : null;
  const where = p?.place ? ` in ${p.place.split("|").join(" or ")}` : "";
  const what = p?.text ? ` of “${p.text}”` : "";
  const span = spanLabel(range);
  if (when) {
    const from = /^(after|before) /.test(when) ? when : `from ${when}`;
    return `No photos${what}${where} ${from}${span ? ` — your library spans ${span}` : ""}.`;
  }
  if (where) return `No photos${what}${where}.`;
  return "No matches. Try different words.";
}
