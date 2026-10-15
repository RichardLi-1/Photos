import Foundation

/// A search box string split into semantic text + metadata filters. All parsing is local (NSDataDetector, regexes).
struct ParsedQuery: Equatable, Sendable {
    var text: String
    var after: Date?
    var before: Date?
    var place: String?

    var hasFilters: Bool { after != nil || before != nil || place != nil }
}

enum QueryParser {
    private static let months = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
                                 "october", "november", "december"]
    private static let prep = #"(?:\b(?:in|on|from|during|taken)\s+)?"#

    /// "beach in july 2023" → text "beach", July 2023. "receipts last month", "dogs 2019", "cake on march 3".
    /// `places`: known reverse-geocoded names ("Paris, Île-de-France, France"); a query word matching one adds a filter.
    static func parse(_ query: String, now: Date = .now, calendar cal: Calendar = .current, places: [String] = []) -> ParsedQuery {
        var text = query
        var range: (Date, Date)?

        func take(_ pattern: String, _ make: ([String]) -> (Date, Date)?) {
            guard range == nil, let re = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]),
                  let m = re.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) else { return }
            let groups = (0..<m.numberOfRanges).map { i -> String in
                Range(m.range(at: i), in: text).map { String(text[$0]).lowercased() } ?? ""
            }
            guard let r = make(groups), let whole = Range(m.range, in: text) else { return }
            range = r
            text.removeSubrange(whole)
        }

        func span(_ start: Date?, _ unit: Calendar.Component, _ n: Int = 1) -> (Date, Date)? {
            guard let start, let end = cal.date(byAdding: unit, value: n, to: start) else { return nil }
            return (start, end)
        }
        func startOf(_ unit: Calendar.Component, _ date: Date) -> Date? { cal.dateInterval(of: unit, for: date)?.start }

        // Relative phrases.
        take(prep + #"\b(today|yesterday|this\s+(week|month|year)|last\s+(week|month|year)|(\d{1,2})\s+years?\s+ago)\b"#) { g in
            let units: [String: Calendar.Component] = ["week": .weekOfYear, "month": .month, "year": .year]
            if g[1] == "today" { return span(startOf(.day, now), .day) }
            if g[1] == "yesterday" { return span(cal.date(byAdding: .day, value: -1, to: startOf(.day, now) ?? now), .day) }
            if let u = units[g[2]] { return span(startOf(u, now), u) }
            if let u = units[g[3]] { return span(cal.date(byAdding: u, value: -1, to: startOf(u, now) ?? now), u) }
            if let n = Int(g[4]) { return span(cal.date(byAdding: .year, value: -n, to: startOf(.year, now) ?? now), .year) }
            return nil
        }
        // "july", "july 2023" (no day): the whole month; without a year, the most recent July.
        take(prep + #"\b("# + months.joined(separator: "|") + #")(?:\s+((?:19|20)\d{2}))?\b(?!\s+\d)"#) { g in
            guard let month = months.firstIndex(of: g[1]).map({ $0 + 1 }) else { return nil }
            var year = Int(g[2]) ?? cal.component(.year, from: now)
            if g[2].isEmpty && month > cal.component(.month, from: now) { year -= 1 }
            return span(cal.date(from: DateComponents(year: year, month: month, day: 1)), .month)
        }
        // A bare year.
        take(prep + #"\b((?:19|20)\d{2})\b"#) { g in
            span(Int(g[1]).flatMap { cal.date(from: DateComponents(year: $0, month: 1, day: 1)) }, .year)
        }
        // Anything else NSDataDetector understands ("march 3", "last friday", "12/25/2022"): that day.
        if range == nil, let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.date.rawValue),
           let m = detector.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)), var date = m.date,
           var whole = Range(m.range, in: text) {
            // the detector prefers the next occurrence; photos are in the past unless a year was given
            if date > now, text[whole].range(of: #"\d{4}"#, options: .regularExpression) == nil {
                date = cal.date(byAdding: .year, value: -1, to: date) ?? date
            }
            range = span(startOf(.day, date), .day)
            if let p = text.range(of: #"\b(in|on|from|during|taken)\s+$"#, options: [.regularExpression, .caseInsensitive],
                                  range: text.startIndex..<whole.lowerBound) {
                whole = p.lowerBound..<whole.upperBound
            }
            text.removeSubrange(whole)
        }

        var out = ParsedQuery(text: text.split(whereSeparator: \.isWhitespace).joined(separator: " "),
                              after: range?.0, before: range?.1)
        out.place = matchPlace(in: out.text, places: places)
        return out
    }

    /// Longest place component ("Paris") that appears as whole words in the text.
    static func matchPlace(in text: String, places: [String]) -> String? {
        let words = " " + text.lowercased().split { !($0.isLetter || $0.isNumber) }.joined(separator: " ") + " "
        let parts = Set(places.flatMap { $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) } })
        return parts.filter { $0.count >= 3 && words.contains(" " + $0.lowercased() + " ") }.max { $0.count < $1.count }
    }
}
