"""Rule-based query understanding (local, ~1 ms, no LLM).

parse("flowers in Germany last summer") -> ("flowers", Filters(place="Germany", after="2026-06-01", before="2026-09-01"))

Dates are resolved against today. A month/season/holiday without a year ("in March", "at christmas") means every year.
"winter 2017" = Dec 2016 - Feb 2017. Nothing parsed -> the text comes back unchanged.
"""
from __future__ import annotations

import calendar
import functools
import re
from datetime import date, timedelta

from .places import Places
from .search import Filters

MONTHS = dict(jan=1, feb=2, mar=3, apr=4, may=5, jun=6, jul=7, aug=8, sep=9, oct=10, nov=11, dec=12)
MON = r"(?P<mon>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
YR = r"(?P<yr>(?:19|20)\d\d|'\d\d)"
OPT_YR = r"(?:,?\s+(?:of\s+)?" + YR + r")?"
REL = r"(?:(?P<rel>last|this|past|previous)\s+)?"
NUM = dict(a=1, an=1, one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10, eleven=11, twelve=12,
           few=3, couple=2)
N = r"(?P<n>\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|few|couple(?: of)?)"
SEASONS = dict(spring=(3, 6), summer=(6, 9), autumn=(9, 12), fall=(9, 12), winter=(12, 3))
NTH = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3, "last": -1}
WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
MIN, MAX = date(1800, 1, 1), date(2200, 1, 1)

# What may sit right before a bare "March" / "summer" / "christmas" for it to count as a date.
CUE = {"in", "during", "on", "at", "from", "over", "since", "around", "before", "after", "until", "till", "between", "through"}
LEAD = CUE | {"of", "the", "taken", "shot", "prior", "to", "up"}  # removed together with a date or place
STOP = set("""a an the and or of in on at to from for with by into onto over under near around about during between since
before after until till than then that this these those it its is are was were be been being am do does did done has have
had i me my we our us you your he she they them their his her what which who whom when where why how all any some
just very really also so as if not no nor too can could would should will shall may might must there here show find
search look looking photo photos picture pictures pic pics image images""".split())
CLUTTER = re.compile(r"\b(?:show me|find me|search for|look(?:ing)? for|what did (?:i|we) (?:photograph|shoot|take|see|do)|"
                     r"(?:i|we)(?: have)? (?:took|take|taken|shot|captured|photographed|snapped|saw|made|had|did)|my|our|"
                     r"photos?|pictures?|pics?|pix|images?|snapshots?|trip|vacation|holidays?|journey|visit)\b|の?写真|的?照片|사진|фото", re.I)


def _yy(s: str, today: date) -> int:
    y = int(s.lstrip("'"))
    return y if y > 99 else (2000 + y if 2000 + y <= today.year + 1 else 1900 + y)


def _month(y, m):
    return date(y, m, 1), date(y + m // 12, m % 12 + 1, 1)


def _add_months(d: date, k: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 + k, 12)
    return date(y, m + 1, 1)


def _season(name, y):
    a, b = SEASONS[name]
    return (date(y - 1, 12, 1), date(y, 3, 1)) if name == "winter" else (date(y, a, 1), date(y, b, 1))


def _easter(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mo, da = divmod(h + l - 7 * m + 114, 31)
    return date(y, mo, da + 1)


def _holiday(h, y):
    h = h.replace("'", "").replace("’", "")
    if h in ("christmas eve",):
        return date(y, 12, 24), date(y, 12, 25)
    if h in ("christmas day",):
        return date(y, 12, 25), date(y, 12, 26)
    if h in ("christmas", "xmas"):
        return date(y, 12, 24), date(y, 12, 27)
    if h in ("new years eve", "nye"):
        return date(y, 12, 31), date(y + 1, 1, 1)
    if h == "new years day":
        return date(y, 1, 1), date(y, 1, 2)
    if h in ("new years", "new year"):
        return date(y - 1, 12, 31), date(y, 1, 2)
    if h == "halloween":
        return date(y, 10, 31), date(y, 11, 1)
    if h.startswith("valentine"):
        return date(y, 2, 14), date(y, 2, 15)
    if h == "easter":
        e = _easter(y)
        return e - timedelta(2), e + timedelta(2)
    if h == "thanksgiving":  # US: 4th Thursday of November, through the weekend
        d = date(y, 11, 1) + timedelta((3 - date(y, 11, 1).weekday()) % 7 + 21)
        return d, d + timedelta(4)
    return date(y, 7, 4), date(y, 7, 5)  # independence day / fourth of july


class Atom:
    """One date expression. Either fixed intervals, or a per-year rule plus (optionally) a year or last/this."""

    def __init__(self, span, fixed=None, per_year=None, year=None, rel=None):
        self.span, self.fixed, self.per_year, self.year, self.rel = list(span), fixed, per_year, year, rel

    @property
    def bare(self):  # "March", "summer", "christmas": every year, only trusted with a cue word or another date
        return self.fixed is None and self.year is None and self.rel is None

    def resolve(self, today) -> list:
        if self.fixed is not None:
            return self.fixed
        if self.year is not None:
            return [self.per_year(self.year)]
        if self.rel:
            ivs = [self.per_year(y) for y in range(today.year + 1, today.year - 3, -1)]
            if self.rel == "this":
                return [next((iv for iv in ivs if iv[0] <= today < iv[1]), self.per_year(today.year))]
            return [next(iv for iv in ivs if iv[1] <= today)]
        return [self.per_year(y) for y in range(1970, today.year + 2)]


def _atoms(t: str, today: date) -> list[Atom]:
    """Date expressions in lowercased text, most specific patterns first, non-overlapping."""
    out, taken = [], []

    def free(m):
        return not any(m.start() < b and m.end() > a for a, b in taken)

    def mon(m):
        return MONTHS[m.group("mon")[:3]]

    def yr(m, g="yr"):
        return _yy(m.group(g), today) if m.group(g) else None

    def iso(m):
        y, mo, d = int(m["y"]), int(m["mo"]), m["d"]
        if not 1 <= mo <= 12:
            return None
        return Atom(m.span(), fixed=[(date(y, mo, int(d)), date(y, mo, int(d)) + timedelta(1)) if d else _month(y, mo)])

    def numeric(m):
        a, b, y = int(m["a"]), int(m["b"]), _yy(m["y"], today)
        mo, d = (b, a) if m["sep"] == "." or a > 12 else (a, b)  # 29.09.2025 / 9/29/2025 (US order when ambiguous)
        return Atom(m.span(), fixed=[(date(y, mo, d), date(y, mo, d) + timedelta(1))])

    def week_of(m):
        k, mo = NTH[m["nth"]], mon(m)
        def f(y):
            a, b = _month(y, mo)
            s = b - timedelta(7) if k < 0 else a + timedelta(7 * k)
            return s, s + timedelta(7)
        return Atom(m.span(), per_year=f, year=yr(m))

    def part(m):
        p = m["part"].split()[0]
        lo, hi = {"early": (0, 1 / 3), "beginning": (0, 1 / 3), "start": (0, 1 / 3), "mid": (1 / 3, 2 / 3),
                  "middle": (1 / 3, 2 / 3), "late": (2 / 3, 1), "end": (2 / 3, 1)}[p]
        def cut(a, b):
            n = (b - a).days
            return a + timedelta(round(n * lo)), a + timedelta(round(n * hi))
        if m["y2"]:
            return Atom(m.span(), fixed=[cut(date(int(m["y2"]), 1, 1), date(int(m["y2"]) + 1, 1, 1))])
        mo = mon(m)
        return Atom(m.span(), per_year=lambda y: cut(*_month(y, mo)), year=yr(m))

    def holiday(m):
        h = m["hol"]
        return Atom(m.span(), per_year=lambda y: _holiday(h, y), year=yr(m), rel=_rel(m))

    def month_day(m):
        mo, d = mon(m), int(m["d"])
        if not 1 <= d <= 31:
            return None
        def f(y):
            x = date(y, mo, min(d, calendar.monthrange(y, mo)[1]))
            return x, x + timedelta(1)
        return Atom(m.span(), per_year=f, year=yr(m))

    def season(m):
        s = m["season"]
        return Atom(m.span(), per_year=lambda y: _season(s, y), year=yr(m), rel=_rel(m))

    def month_year(m):
        mo = mon(m)
        return Atom(m.span(), per_year=lambda y: _month(y, mo), year=yr(m), rel=_rel(m))

    def day_word(m):
        w = m["w"]
        d = today - timedelta(1) if w in ("yesterday", "last night") else today
        return Atom(m.span(), fixed=[(d, d + timedelta(1))])

    def rel_unit(m):
        r, u = m["rel"], m["unit"]
        if r == "past":  # rolling: "the past year"
            return Atom(m.span(), fixed=[(_minus(today, 1, u), today + timedelta(1))])
        k = 0 if r == "this" else -1
        if u in ("week", "weekend"):
            mon0 = today - timedelta(today.weekday()) + timedelta(7 * k)
            return Atom(m.span(), fixed=[(mon0 + timedelta(5), mon0 + timedelta(7)) if u == "weekend" else (mon0, mon0 + timedelta(7))])
        if u == "month":
            a = _add_months(today.replace(day=1), k)
            return Atom(m.span(), fixed=[(a, _add_months(a, 1))])
        if u == "decade":
            y = today.year // 10 * 10 + 10 * k
            return Atom(m.span(), fixed=[(date(y, 1, 1), date(y + 10, 1, 1))])
        return Atom(m.span(), fixed=[(date(today.year + k, 1, 1), date(today.year + k + 1, 1, 1))])

    def ago(m):
        n, u = _n(m["n"]), m["unit"]
        if u == "day":
            d = today - timedelta(n)
            return Atom(m.span(), fixed=[(d, d + timedelta(1))])
        if u == "week":
            d = today - timedelta(7 * n)
            a = d - timedelta(d.weekday())
            return Atom(m.span(), fixed=[(a, a + timedelta(7))])
        if u == "month":
            a = _add_months(today.replace(day=1), -n)
            return Atom(m.span(), fixed=[(a, _add_months(a, 1))])
        return Atom(m.span(), fixed=[(date(today.year - n, 1, 1), date(today.year - n + 1, 1, 1))])

    def past_n(m):
        return Atom(m.span(), fixed=[(_minus(today, _n(m["n"]), m["unit"]), today + timedelta(1))])

    def weekday(m):
        wd = WEEKDAYS.index(m["wd"])
        back = (today.weekday() - wd) % 7 or (7 if m["rel"] == "last" else 0)
        d = today - timedelta(back)
        return Atom(m.span(), fixed=[(d, d + timedelta(1))])

    def decade(m):
        if m["d2"]:
            y = (1900 if int(m["d2"]) >= 3 else 2000) + 10 * int(m["d2"])
        else:
            y = int(m["c"] + m["d"] + "0")
        return Atom(m.span(), fixed=[(date(y, 1, 1), date(y + 10, 1, 1))])

    def year(m):
        y = int(m["y"])
        return Atom(m.span(), fixed=[(date(y, 1, 1), date(y + 1, 1, 1))]) if 1900 <= y <= today.year + 1 else None

    rules = [
        (r"\b(?P<y>(?:19|20)\d\d)[-/.](?P<mo>\d{1,2})(?:[-/.](?P<d>\d{1,2}))?\b", iso),
        (r"\b(?P<a>\d{1,2})(?P<sep>[/.])(?P<b>\d{1,2})(?P=sep)(?P<y>(?:19|20)?\d\d)\b", numeric),
        (r"\b(?:the\s+)?(?P<nth>first|1st|second|2nd|third|3rd|fourth|4th|last)\s+week\s+(?:of|in)\s+" + MON + OPT_YR + r"\b", week_of),
        (r"\b(?P<part>early|mid|late|end of|beginning of|start of|middle of)[\s-]+(?:the\s+)?(?:(?P<y2>(?:19|20)\d\d)|" + MON + OPT_YR + r")\b", part),
        (r"\b" + REL + r"(?P<hol>christmas eve|christmas day|christmas|xmas|new year'?s eve|new year'?s day|new year'?s|new year|nye|"
         r"halloween|thanksgiving|easter|valentine'?s(?: day)?|independence day|(?:fourth|4th) of july|july 4(?:th)?)" + OPT_YR + r"\b", holiday),
        (r"\b" + MON + r"\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?(?![\d:])" + r"(?:,?\s+" + YR + r")?\b", month_day),
        (r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + MON + r"(?:,?\s+" + YR + r")?\b", month_day),
        (r"\b" + REL + r"(?P<season>spring|summer|autumn|fall|winter)(?:time)?" + OPT_YR + r"\b", season),
        (r"\b" + REL + MON + OPT_YR + r"\b", month_year),
        (r"\b(?P<w>today|tonight|yesterday|this morning|this afternoon|this evening|last night)\b", day_word),
        (r"\b(?:the\s+)?(?:last|past|previous)\s+" + N + r"\s+(?P<unit>day|week|month|year)s\b", past_n),
        (r"\b(?P<rel>this|last|past|previous)\s+(?P<unit>weekend|week|month|year|decade)\b", rel_unit),
        (r"\b" + N + r"\s+(?P<unit>day|week|month|year)s?\s+ago\b", ago),
        (r"\b(?P<rel>last|this|on|past)\s+(?P<wd>" + "|".join(WEEKDAYS) + r")\b", weekday),
        (r"\b(?:(?P<c>19|20)(?P<d>\d)0'?s|(?:the\s+|')(?P<d2>\d)0'?s)\b", decade),
        (r"(?<![\w/.:])(?P<y>(?:19|20)\d\d)(?![\w/:])", year),
    ]
    for rx, fn in zip(_compiled(tuple(r for r, _ in rules)), [f for _, f in rules]):
        for m in rx.finditer(t):
            if free(m):
                try:
                    a = fn(m)
                except ValueError:  # Feb 30 etc.
                    a = None
                if a:
                    out.append(a); taken.append(m.span())
    return sorted(out, key=lambda a: a.span[0])


@functools.lru_cache(4)
def _compiled(rules):
    return [re.compile(r) for r in rules]


def _rel(m):
    return m["rel"] and ("this" if m["rel"] == "this" else "last")


def _n(s):
    return int(s) if s.isdigit() else NUM[s.split()[0]]


def _minus(today, n, unit):
    if unit == "day":
        return today - timedelta(n - 1)
    if unit == "week":
        return today - timedelta(7 * n)
    if unit == "month":
        y, mo = divmod(today.year * 12 + today.month - 1 - n, 12)
        return date(y, mo + 1, min(today.day, 28))
    return today.replace(year=today.year - n, day=min(today.day, 28))


def _intersect(a, b):
    out = [(max(x[0], y[0]), min(x[1], y[1])) for x in a for y in b]
    return sorted(iv for iv in out if iv[0] < iv[1])


def _prev_word(t, i):
    m = re.search(r"(\w+)\s+(?:the\s+)?$", t[:i])
    return (m.group(1), m.start(1)) if m else (None, i)


def parse_dates(t: str, today: date) -> tuple[list | None, list]:
    """(intervals or None, spans to remove). t is lowercased."""
    atoms = _atoms(t, today)
    # ranges: "from March to May 2021", "between 2015 and 2017", "2015-2017"
    merged = []
    for a in atoms:
        if merged:
            p = merged[-1]
            gap = t[p.span[1]:a.span[0]]
            pw, _ = _prev_word(t, p.span[0])
            if re.fullmatch(r"\s*(?:-|–|—|to|until|till|through|thru)\s*", gap) or (re.fullmatch(r"\s+and\s+", gap) and pw == "between"):
                merged[-1] = _range(p, a, today)
                continue
        merged.append(a)
    ivs, spans, bare = None, [], []
    for a in merged:
        pw, ps = _prev_word(t, a.span[0])
        if a.bare and pw not in CUE:
            bare.append(a)
            continue
        r = a.resolve(today)
        lo, hi = r[0][0], r[-1][1]
        if pw in ("since",):
            r = [(lo, MAX)]
        elif pw == "after":
            r = [(hi, MAX)]
        elif pw == "before" or (pw == "to" and t[:ps].rstrip().endswith("prior")):
            r = [(MIN, lo)]
        elif pw in ("until", "till"):
            r = [(MIN, hi)]
        ivs = r if ivs is None else _intersect(ivs, r)
        spans.append((ps if pw in LEAD else a.span[0], a.span[1]))
    if ivs is not None:  # "summer photos from 2019": a bare season narrows a real date
        for a in bare:
            ivs = _intersect(ivs, a.resolve(today))
            spans.append(tuple(a.span))
    return ivs, spans


def _range(a: Atom, b: Atom, today) -> Atom:
    span = (a.span[0], b.span[1])
    if a.bare and b.bare:
        def f(y, fa=a.per_year, fb=b.per_year):
            s, e = fa(y), fb(y)
            return s[0], (e if e[1] > s[0] else fb(y + 1))[1]
        return Atom(span, per_year=f)
    if a.bare and b.year is not None:
        a.year = b.year if a.per_year(b.year)[0] <= b.per_year(b.year)[0] else b.year - 1
    if b.bare and a.year is not None:
        b.year = a.year if b.per_year(a.year)[0] >= a.per_year(a.year)[0] else a.year + 1
    ra, rb = a.resolve(today), b.resolve(today)
    return Atom(span, fixed=[(ra[0][0], rb[-1][1])])


@functools.lru_cache(1)
def _gazetteer_only():
    return Places()


def _cut(text, spans):
    for a, b in sorted(spans, reverse=True):
        text = text[:a] + " " + text[b:]
    return text


def clean_visual(text: str) -> str:
    """Remove search clutter ("photos", "I took", "my trip") and dangling function words at the edges."""
    t = CLUTTER.sub(" ", text)
    words = re.findall(r"[\w'’-]+|[^\w\s]", t)
    while words and (words[0].lower() in STOP or not re.match(r"\w", words[0])):
        words.pop(0)
    while words and (words[-1].lower() in STOP or not re.match(r"\w", words[-1])):
        words.pop()
    return re.sub(r"\s+([,.!?;:])", r"\1", " ".join(words)).strip()


def parse(text: str, places: Places | None = None, today: date | None = None) -> tuple[str, Filters]:
    """Split a search-box query into (visual text, Filters with dates/place)."""
    today = today or date.today()
    P = places or _gazetteer_only()
    low = text.lower()
    ivs, dspans = parse_dates(low, today)
    found = P.find(text, skip=dspans)
    names = [n for _, _, n in found]
    names = [n for n in dict.fromkeys(names) if not any(P.is_within(n, o) for o in names if o != n)]  # "Bali, Indonesia" -> Bali
    pspans = []
    for a, b, _ in found:
        m = re.search(r"\b(?:" + "|".join(sorted(LEAD)) + r")\s+(?:the\s+)?$", text[:a], re.I)
        pspans.append((m.start() if m else a, b))
    f = Filters(place="|".join(names) or None)
    if ivs is not None:
        if len(ivs) == 1:
            f.after = ivs[0][0].isoformat() if ivs[0][0] > MIN else None
            f.before = ivs[0][1].isoformat() if ivs[0][1] < MAX else None
        else:
            f.dates = [(a.isoformat(), b.isoformat()) for a, b in ivs] or [("1800-01-01", "1800-01-01")]  # empty: match nothing
    if not (dspans or pspans):
        return text, f
    return clean_visual(_cut(text, dspans + pspans)), f
