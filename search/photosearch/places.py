"""Place names: aliases across languages + country/state codes, matched against the library's free-text `place` column.

`Places(library_place_strings)`:
  .find(text)   place mentions in a query  -> [(start, end, name)]
  .match(name)  predicate over `place` strings ("Germany" matches "Erlangen, Deutschland" and "Berlin")
"""
from __future__ import annotations

import functools
import gettext
import re
import unicodedata

# Translations of country names come from pycountry (ISO 3166 + gettext catalogs); dictionary words are dropped later.
LANGS = "de fr es it pt nl sv nb_NO da fi is pl cs hu ro hr el ru uk tr ja zh_CN zh_TW ko th vi id he ar".split()

# Country canonical name first, then extra aliases (endonyms and short forms pycountry lacks).
COUNTRY_EXTRA = {
    "US": "United States|USA|U.S.|U.S.A.|United States of America|America|美国|美國|미국|アメリカ|Сполучені Штати Америки|Estados Unidos|États-Unis",
    "GB": "United Kingdom|UK|U.K.|Great Britain|Britain|英国|英國|イギリス", "KR": "South Korea|Korea|대한민국|한국",
    "KP": "North Korea", "RU": "Russia|Россия", "VN": "Vietnam|Việt Nam", "TW": "Taiwan|台灣|台湾", "CZ": "Czechia|Czech Republic|Česko",
    "TR": "Turkey|Türkiye", "NL": "Netherlands|Holland|The Netherlands|Nederland", "IR": "Iran|ایران", "SY": "Syria", "LA": "Laos",
    "BO": "Bolivia", "VE": "Venezuela", "TZ": "Tanzania", "MD": "Moldova", "BN": "Brunei", "MK": "North Macedonia|Macedonia",
    "PS": "Palestine", "VA": "Vatican City|Vatican", "CD": "DR Congo|Democratic Republic of the Congo", "CG": "Republic of the Congo",
    "CI": "Ivory Coast|Côte d'Ivoire", "CV": "Cape Verde", "SZ": "Eswatini|Swaziland", "MM": "Myanmar|Burma", "FM": "Micronesia",
    "CN": "China|中国|中國", "JP": "Japan|日本|Nippon", "TH": "Thailand|ประเทศไทย|Thaïlande|Thailande", "MX": "Mexico|México",
    "CH": "Switzerland|Schweiz|Suisse|Svizzera|Svizra", "DE": "Germany|Deutschland", "IS": "Iceland|Ísland",
    "NO": "Norway|Norge|Noreg", "AT": "Austria|Österreich", "IT": "Italy|Italia", "ES": "Spain|España", "GR": "Greece|Ελλάδα|Hellas",
    "AE": "United Arab Emirates|UAE", "NZ": "New Zealand|Aotearoa", "IE": "Ireland|Éire", "BE": "Belgium|België|Belgique",
}

# Regions and cities: "Parent: Name|alias; Name|alias". A filter on the parent also matches these.
SUBPLACES = """
United States: New York|New York City|NYC|Manhattan|Brooklyn|Нью-Йорк|ニューヨーク|纽约; Los Angeles|LA|L.A.; San Francisco|SF; Chicago; Las Vegas; Seattle; Boston; Miami; New Orleans; Washington D.C.|Washington DC; Philadelphia; San Diego; Austin; Portland; Denver; Honolulu; Yosemite|Yosemite National Park; Grand Canyon; Yellowstone; Zion National Park|Zion; Death Valley; Joshua Tree; Big Sur; Lake Tahoe; Glacier National Park; Antelope Canyon; Monument Valley; Arches National Park; Bryce Canyon
United Kingdom: England; Scotland|Alba; Wales|Cymru; Northern Ireland; Lake District; Cornwall; Isle of Skye|Skye
England: London|Londres|Londra|Лондон|ロンドン|伦敦; Manchester; Liverpool; Oxford; Cambridge; Brighton; Bristol; York; Bath
Scotland: Edinburgh; Glasgow; Highlands
Germany: Berlin; Munich|München|Muenchen; Hamburg; Cologne|Köln|Koeln; Frankfurt; Dresden; Stuttgart; Düsseldorf|Dusseldorf; Leipzig; Heidelberg; Nuremberg|Nürnberg; Bavaria|Bayern; Black Forest|Schwarzwald
France: Paris|パリ|巴黎|Париж; Lyon; Marseille|Marseilles; Nice; Bordeaux; Strasbourg; Provence; Normandy|Normandie; Brittany|Bretagne; Corsica|Corse; Chamonix; Mont Saint-Michel
Italy: Rome|Roma; Florence|Firenze; Venice|Venezia; Milan|Milano; Naples|Napoli; Turin|Torino; Tuscany|Toscana; Sicily|Sicilia; Sardinia|Sardegna; Amalfi|Amalfi Coast; Cinque Terre; Dolomites|Dolomiti; Lake Como|Como; Verona; Bologna; Pisa
Spain: Madrid; Barcelona; Seville|Sevilla; Valencia; Granada; Ibiza; Mallorca|Majorca; Tenerife; Canary Islands|Canarias; Bilbao; Málaga|Malaga
Portugal: Lisbon|Lisboa; Porto|Oporto; Madeira; Azores|Açores; Algarve
Netherlands: Amsterdam; Rotterdam; The Hague|Den Haag; Utrecht
Belgium: Brussels|Bruxelles|Brussel; Bruges|Brugge; Antwerp|Antwerpen; Ghent|Gent
Switzerland: Zurich|Zürich; Geneva|Genève|Genf; Bern|Berne; Lucerne|Luzern; Zermatt; Interlaken; Lausanne; Basel; Lugano; Grindelwald; Matterhorn
Austria: Vienna|Wien; Salzburg; Innsbruck; Hallstatt; Tyrol|Tirol
Czechia: Prague|Praha|Prag
Poland: Warsaw|Warszawa; Krakow|Kraków; Gdansk|Gdańsk
Hungary: Budapest
Greece: Athens|Athína|Αθήνα; Santorini|Thira; Mykonos; Crete|Kreta; Rhodes
Turkey: Istanbul|İstanbul; Cappadocia|Kapadokya; Antalya
Russia: Moscow|Москва|Moskva; Saint Petersburg|St. Petersburg|St Petersburg|Санкт-Петербург
Norway: Oslo; Bergen; Lofoten; Tromsø|Tromso; Svalbard
Sweden: Stockholm; Gothenburg|Göteborg
Denmark: Copenhagen|København; Faroe Islands
Finland: Helsinki; Lapland|Lappi
Iceland: Reykjavik|Reykjavík
Ireland: Dublin; Galway
Croatia: Dubrovnik; Split; Zagreb; Plitvice
Japan: Tokyo|東京|Tōkyō; Kyoto|京都; Osaka|大阪; Hokkaido|北海道; Okinawa|沖縄; Nara|奈良; Hiroshima|広島; Fuji|Mount Fuji|富士山; Sapporo|札幌; Kamakura|鎌倉
China: Beijing|Peking|北京; Shanghai|上海; Hong Kong|香港; Guangzhou|广州; Shenzhen|深圳; Chengdu|成都; Xi'an|Xian|西安; Guilin|桂林; Tibet|西藏
South Korea: Seoul|서울; Busan|부산; Jeju|제주
Taiwan: Taipei|台北
Thailand: Bangkok|กรุงเทพ; Chiang Mai|เชียงใหม่; Phuket|ภูเก็ต; Krabi
Vietnam: Hanoi|Hà Nội; Ho Chi Minh City|Saigon; Ha Long Bay|Halong Bay; Hoi An
Indonesia: Bali; Jakarta; Lombok; Java
Malaysia: Kuala Lumpur
Philippines: Manila; Palawan; Boracay
India: Delhi|New Delhi; Mumbai|Bombay; Goa; Agra; Jaipur; Varanasi; Kerala
Nepal: Kathmandu; Everest|Mount Everest
United Arab Emirates: Dubai; Abu Dhabi
Egypt: Cairo; Giza; Luxor
Morocco: Marrakech|Marrakesh; Fes|Fez; Chefchaouen
South Africa: Cape Town; Johannesburg
Kenya: Nairobi; Maasai Mara
Tanzania: Serengeti; Zanzibar; Kilimanjaro
Canada: Toronto; Vancouver; Montreal|Montréal; Quebec|Québec; Banff; Jasper; Calgary; Ottawa; Whistler; British Columbia; Alberta; Ontario; Nova Scotia
Mexico: Mexico City|Ciudad de México|CDMX; Cancun|Cancún; Tulum; Oaxaca
Brazil: Rio de Janeiro|Rio; São Paulo|Sao Paulo
Argentina: Buenos Aires; Patagonia
Peru: Lima; Cusco|Cuzco; Machu Picchu
Chile: Santiago; Atacama; Torres del Paine
Australia: Sydney; Melbourne; Brisbane; Perth; Adelaide; Cairns; Great Barrier Reef; Tasmania; Uluru; Gold Coast
New Zealand: Auckland; Wellington; Queenstown; Milford Sound; Christchurch
"""

US_STATES = ("AL Alabama|AK Alaska|AZ Arizona|AR Arkansas|CA California|CO Colorado|CT Connecticut|DE Delaware|FL Florida|"
             "GA Georgia|HI Hawaii|ID Idaho|IL Illinois|IN Indiana|IA Iowa|KS Kansas|KY Kentucky|LA Louisiana|ME Maine|"
             "MD Maryland|MA Massachusetts|MI Michigan|MN Minnesota|MS Mississippi|MO Missouri|MT Montana|NE Nebraska|"
             "NV Nevada|NH New Hampshire|NJ New Jersey|NM New Mexico|NY New York State|NC North Carolina|ND North Dakota|"
             "OH Ohio|OK Oklahoma|OR Oregon|PA Pennsylvania|RI Rhode Island|SC South Carolina|SD South Dakota|TN Tennessee|"
             "TX Texas|UT Utah|VT Vermont|VA Virginia|WA Washington State|WV West Virginia|WI Wisconsin|WY Wyoming")

# Names that are also ordinary words: only places when Capitalized or after "in/from/at/...".
AMBIGUOUS = set("turkey china chile jordan georgia chad guinea jersey nice victoria split reading bath mobile cork orange "
                "phoenix buffalo columbia rio york java fuji highlands everest giza goa rome lima santiago jasper "
                "cambridge oxford washington america holland".split())
PREPS = {"in", "from", "at", "near", "around", "to", "visiting", "across", "throughout", "within", "inside", "outside"}
# Trailing words that make a place name generic ("Yosemite Valley" -> also findable as "Yosemite").
SUFFIX = re.compile(r"\s+(?:national park|state park|provincial park|national forest|park|valley|beach|island|islands|"
                    r"lake|mountains?|falls|bay|river|city|county|province|region|district|trail|trailhead|coast)$")


def fold(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold()


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


@functools.lru_cache(1)
def common_words() -> frozenset:
    """Lowercase dictionary words (macOS /usr/share/dict). A token outside this set is 'proper-noun-ish'."""
    try:
        return frozenset(w.strip() for w in open("/usr/share/dict/words") if w[:1].islower())
    except OSError:
        return frozenset()


def is_common(w: str) -> bool:
    w, C = w.lower(), common_words()
    return any(x in C for x in (w, w[:-1] if w.endswith("s") else "", w[:-2] if w.endswith("es") else "",
                                w[:-3] + "y" if w.endswith("ies") else "", w[:-3] if w.endswith("ing") else "",
                                w[:-2] if w.endswith("ed") else ""))


def _spaced(s: str) -> bool:  # scripts that separate words with spaces (vs CJK / Thai: substring match)
    return not re.search(r"[฀-๿぀-ヿ㐀-鿿가-힯]", s)


class _Entry:
    def __init__(self, name, parent=None):
        self.name, self.parent, self.aliases, self.codes, self.last_codes, self.children = name, parent, set(), set(), set(), []


@functools.lru_cache(1)
def gazetteer() -> tuple[dict, dict]:
    """(entries by canonical name, entry by alias key). Keys are folded, without dots; short all-caps ones keep case."""
    import pycountry
    E, by_alias = {}, {}

    def add(e, alias, strict=False, code=False):
        alias = alias.strip()
        if alias.replace(".", "").isupper() and len(alias.replace(".", "")) <= 4:  # USA, UK, NYC, LA: query abbreviations
            k = alias.replace(".", "")
            by_alias.setdefault(k, e)
            if code:
                e.codes.add(k)
            if not is_common(k.lower()):
                by_alias.setdefault(k.lower(), e)
            return
        a = fold(alias)
        if not strict and (len(a) < (3 if _spaced(a) else 2) or (a in common_words() and a != fold(e.name))):
            return  # e.g. Danish "Island" (= Iceland)
        for v in {a, strip_accents(a)}:
            if v == a or not is_common(v):
                e.aliases |= {v, v.replace(".", "")}
                by_alias.setdefault(v.replace(".", ""), e)

    tr = []
    for lang in LANGS:
        try:
            tr.append(gettext.translation("iso3166-1", pycountry.LOCALES_DIR, languages=[lang]).gettext)
        except OSError:
            pass
    countries = list(pycountry.countries)
    for c in countries:  # canonical + curated aliases first, so they win alias collisions ("Korea" -> South Korea)
        extra = [x for x in COUNTRY_EXTRA.get(c.alpha_2, "").split("|") if x]
        name = extra[0] if extra else (getattr(c, "common_name", None) or c.name.split(",")[0])
        e = E[name] = _Entry(name)
        e.codes.add(c.alpha_3); e.last_codes.add(c.alpha_2)
        for n in [name] + extra[1:]:
            add(e, n, strict=True, code=True)
    for c in countries:
        e = E[(COUNTRY_EXTRA.get(c.alpha_2) or getattr(c, "common_name", None) or c.name.split(",")[0]).split("|")[0]]
        for n in (getattr(c, "common_name", None), c.name.split(",")[0], getattr(c, "official_name", None)):
            if n:
                add(e, n, strict=True)
        for t in tr:
            for n in (c.name, getattr(c, "official_name", None), getattr(c, "common_name", None)):
                if n and t(n) != n:
                    add(e, t(n).split(",")[0])
    for line in SUBPLACES.strip().splitlines():
        parent, items = line.split(":", 1)
        p = E[parent.strip()]
        for it in items.split(";"):
            names = [n.strip() for n in it.split("|")]
            if names[0] not in E:
                E[names[0]] = _Entry(names[0], p); p.children.append(E[names[0]])
            for n in names:
                add(E[names[0]], n, strict=True)
    us, states = E["United States"], set()
    for s in US_STATES.split("|"):
        code, name = s.split(" ", 1)
        e = E[name] = _Entry(name, us); us.children.append(e)
        e.codes.add(code); states.add(code)
        add(e, name, strict=True); add(e, name.replace(" State", ""), strict=True)
    for e in E.values():  # "Austin, TX" / "Los Angeles, CA": a US state, not Texas... or Canada
        e.last_codes -= states
    return E, by_alias


def _matcher(aliases, codes, last):
    words = sorted(a for a in aliases if _spaced(a))
    rx = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, words)) + r")(?!\w)") if words else None
    cjk = [a for a in aliases if not _spaced(a)]

    def ok(place):
        f = fold(place)
        if (rx and (rx.search(f) or rx.search(strip_accents(f)))) or any(a in f for a in cjk):
            return True
        comps = _components(place)
        return bool(comps) and (any(c in codes for c in comps) or comps[-1] in last)
    return ok


def _components(place: str) -> list[str]:
    return [re.sub(r"\s+\d[\d -]*$", "", c.strip()) for c in place.split(",") if c.strip()]  # "CA 92037" -> "CA"


class Places:
    def __init__(self, library_places=()):
        self.E, self.by_alias = gazetteer()
        self.vocab = set()  # phrases from the library's own place names (folded)
        for p in library_places:
            for c in _components(p or ""):
                c = fold(c)
                if not c or c.isdigit():
                    continue
                self.vocab.add(c)
                self.vocab.add(SUFFIX.sub("", c))
                for w in re.findall(r"[^\W\d_]{3,}", c):
                    if not is_common(w):
                        self.vocab.add(w)
        self._cache, self._countries, self._codes = {}, {}, {}
        for e in self.E.values():
            if e.parent is None:
                for c in e.codes | e.last_codes:
                    self._codes.setdefault(c, []).append(e)
        self._cjk = [(a, e) for a, e in self.by_alias.items() if not _spaced(a) and e.parent is None]

    # ---------- query side ----------
    def find(self, text: str, skip=()) -> list[tuple[int, int, str]]:
        """Place mentions in a query: [(start, end, canonical name)], longest first, outside the `skip` spans."""
        toks = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\W\d_]+(?:[.'’-][^\W\d_]+)*\.?", text)]
        used = [False] * len(toks)
        for a, b in skip:
            for i, t in enumerate(toks):
                used[i] |= t[0] < b and t[1] > a
        out = []
        for n in (4, 3, 2, 1):
            for i in range(len(toks) - n + 1):
                if any(used[i:i + n]):
                    continue
                win = toks[i:i + n]
                raw = text[win[0][0]:win[-1][1]]
                phrase = fold(" ".join(t[2] for t in win)).replace(".", "")
                prep = i > 0 and fold(toks[i - 1][2]) in PREPS or (i > 1 and fold(toks[i - 1][2]) == "the" and fold(toks[i - 2][2]) in PREPS)
                cap = all(t[2][:1].isupper() for t in win)
                name = None
                if raw.replace(".", "") in self.by_alias and raw.replace(".", "").isupper():  # NYC, USA, UK, LA
                    name = self.by_alias[raw.replace(".", "")].name
                elif phrase in self.by_alias and (phrase not in AMBIGUOUS or cap or prep):
                    name = self.by_alias[phrase].name
                elif phrase in self.vocab and len(phrase) >= 3:
                    words = phrase.split()
                    if all(not is_common(w) for w in words) or (cap and prep):  # "Lofoten" / "in Hot Springs"
                        name = text[win[0][0]:win[-1][1]].rstrip(".")
                if name:
                    out.append((win[0][0], win[-1][1], name))
                    used[i:i + n] = [True] * n
        for alias, e in self.by_alias.items():  # CJK names have no word boundaries
            if not _spaced(alias) and alias in text:
                s = text.index(alias)
                if not any(a <= s < b for a, b, _ in out) and not any(a <= s < b for a, b in skip):
                    out.append((s, s + len(alias), e.name))
        return sorted(out)

    def is_within(self, outer: str, inner: str) -> bool:
        e, o = self.E.get(inner), self.E.get(outer)
        while e is not None and o is not None:
            e = e.parent
            if e is o:
                return True
        return False

    # ---------- library side ----------
    def match(self, name: str):
        """Predicate over a `place` string. Known names expand to aliases, sub-places and codes; others match as words."""
        if name in self._cache:
            return self._cache[name]
        e = self.E.get(name) or self.by_alias.get(fold(name)) or self.by_alias.get(name)
        own = _matcher(e.aliases, e.codes, e.last_codes) if e else _matcher({fold(name)}, (), ())
        subs, todo = [], list(e.children) if e else []
        while todo:
            x = todo.pop(); subs.append(x); todo += x.children
        sub = _matcher(set().union(*(x.aliases for x in subs)), set().union(*(x.codes for x in subs)), ()) if subs else None
        home, x = set(), e
        while x is not None:
            home.add(x.name); x = x.parent

        def ok(place: str | None) -> bool:
            if not place:
                return False
            if own(place):
                return True  # a sub-place counts unless the string names another country ("Melbourne, United States")
            return bool(sub and sub(place)) and not (self.countries_in(place) - home)
        self._cache[name] = ok
        return ok

    def countries_in(self, place: str) -> set[str]:
        if place not in self._countries:
            out = set()
            for c in _components(place):
                if c.isupper() and len(c) <= 3:
                    out |= {e.name for e in self._codes.get(c, ())}
                w = fold(c).replace(".", "").split()
                for n in range(1, min(4, len(w)) + 1):
                    for i in range(len(w) - n + 1):
                        e = self.by_alias.get(" ".join(w[i:i + n]))
                        if e is not None and e.parent is None:
                            out.add(e.name)
                out |= {e.name for a, e in self._cjk if a in c}
            self._countries[place] = out
        return self._countries[place]
