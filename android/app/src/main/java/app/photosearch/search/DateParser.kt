package app.photosearch.search

import java.time.DayOfWeek
import java.time.LocalDate
import java.time.temporal.TemporalAdjusters

/** [start, end) in local dates; either side may be open. */
data class DateRange(val start: LocalDate?, val end: LocalDate?)

/** The query with its date phrase removed (what the embedding sees), plus the date filter it named. */
data class ParsedQuery(val text: String, val range: DateRange?, val phrase: String? = null)

/**
 * English date phrases in a search ("dogs last summer", "receipts from march 2023", "beach in 2019", "2 weeks ago").
 * Same conventions as the prototype's planner: "last summer" = the most recent June 1 - Sept 1 that has ended,
 * "last year" = the previous calendar year, "in 2019" = 2019-01-01 to 2020-01-01. One date phrase per query.
 */
object DateParser {
    private val MONTHS = listOf("january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december")
    private val MONTH_RE = (MONTHS + MONTHS.map { it.take(3) } + listOf("sept")).sortedByDescending { it.length }.joinToString("|")
    private val NUM = mapOf("a" to 1, "an" to 1, "one" to 1, "two" to 2, "three" to 3, "four" to 4, "five" to 5,
        "six" to 6, "seven" to 7, "eight" to 8, "nine" to 9, "ten" to 10, "a couple of" to 2, "a few" to 3)
    private const val YEAR = """(19\d\d|20\d\d)"""

    private fun month(s: String): Int = if (s == "sept") 9 else MONTHS.indexOfFirst { it.startsWith(s) } + 1

    private class Rule(pattern: String, val f: (MatchResult, LocalDate) -> DateRange?) {
        val re = Regex("""(?<![\w])$pattern(?![\w])""", RegexOption.IGNORE_CASE)
    }

    private fun year(y: Int) = DateRange(LocalDate.of(y, 1, 1), LocalDate.of(y + 1, 1, 1))
    private fun monthRange(y: Int, m: Int) = LocalDate.of(y, m, 1).let { DateRange(it, it.plusMonths(1)) }
    private fun weekOf(d: LocalDate) = d.with(TemporalAdjusters.previousOrSame(DayOfWeek.MONDAY)).let { DateRange(it, it.plusWeeks(1)) }

    /** Season (meteorological, northern hemisphere) whose start falls in [y]; winter runs Dec y - Mar y+1. */
    private fun season(name: String, y: Int): DateRange = when (name) {
        "spring" -> DateRange(LocalDate.of(y, 3, 1), LocalDate.of(y, 6, 1))
        "summer" -> DateRange(LocalDate.of(y, 6, 1), LocalDate.of(y, 9, 1))
        "fall", "autumn" -> DateRange(LocalDate.of(y, 9, 1), LocalDate.of(y, 12, 1))
        else -> DateRange(LocalDate.of(y, 12, 1), LocalDate.of(y + 1, 3, 1))
    }

    private val RULES = listOf(
        Rule("""(?:between|from)\s+$YEAR\s+(?:and|to|until|-)\s+$YEAR""") { m, _ ->
            val a = m.groupValues[1].toInt(); val b = m.groupValues[2].toInt()
            DateRange(LocalDate.of(minOf(a, b), 1, 1), LocalDate.of(maxOf(a, b) + 1, 1, 1))
        },
        Rule("""(?:before|until|prior to)\s+$YEAR""") { m, _ -> DateRange(null, LocalDate.of(m.groupValues[1].toInt(), 1, 1)) },
        Rule("""after\s+$YEAR""") { m, _ -> DateRange(LocalDate.of(m.groupValues[1].toInt() + 1, 1, 1), null) },
        Rule("""since\s+$YEAR""") { m, _ -> DateRange(LocalDate.of(m.groupValues[1].toInt(), 1, 1), null) },
        Rule("""(\d{1,2}|a couple of|a few|an|a|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|week|month|year)s?\s+ago""") { m, t ->
            val n = m.groupValues[1].toIntOrNull() ?: NUM.getValue(m.groupValues[1].lowercase())
            when (m.groupValues[2].lowercase()) {
                "day" -> t.minusDays(n.toLong()).let { DateRange(it, it.plusDays(1)) }
                "week" -> weekOf(t.minusWeeks(n.toLong()))
                "month" -> t.minusMonths(n.toLong()).let { monthRange(it.year, it.monthValue) }
                else -> year(t.year - n)
            }
        },
        Rule("""today""") { _, t -> DateRange(t, t.plusDays(1)) },
        Rule("""yesterday""") { _, t -> DateRange(t.minusDays(1), t) },
        Rule("""(this|last|past)\s+weekend""") { m, t ->
            // photos are never in the future: "this weekend" = the current/most recent one, "last" = the one before if we're in one
            val sat = t.with(TemporalAdjusters.previousOrSame(DayOfWeek.SATURDAY))
            val inWeekend = t.dayOfWeek == DayOfWeek.SATURDAY || t.dayOfWeek == DayOfWeek.SUNDAY
            val start = if (m.groupValues[1].lowercase() != "this" && inWeekend) sat.minusWeeks(1) else sat
            DateRange(start, start.plusDays(2))
        },
        Rule("""(this|last|past)\s+week""") { m, t -> if (m.groupValues[1].lowercase() == "this") weekOf(t) else weekOf(t.minusWeeks(1)) },
        Rule("""(this|last|past)\s+month""") { m, t ->
            val d = if (m.groupValues[1].lowercase() == "this") t else t.minusMonths(1)
            monthRange(d.year, d.monthValue)
        },
        Rule("""(this|last|past)\s+year""") { m, t -> year(if (m.groupValues[1].lowercase() == "this") t.year else t.year - 1) },
        Rule("""(?:(this|last)\s+)?(spring|summer|fall|autumn|winter)(?:\s+(?:of\s+)?$YEAR)?""") { m, t ->
            val name = m.groupValues[2].lowercase()
            val mod = m.groupValues[1].lowercase()
            m.groupValues[3].toIntOrNull()?.let { return@Rule season(name, it) }
            val cur = season(name, t.year).let { if (name == "winter" && t.monthValue < 3) season(name, t.year - 1) else it }
            when (mod) {
                "this" -> cur
                "last" -> if (cur.end!! <= t) cur else season(name, cur.start!!.year - 1) // most recent that has ended
                else -> if (cur.start!! <= t) cur else season(name, cur.start.year - 1)    // most recent that has started
            }
        },
        Rule("""(christmas|xmas|new year'?s eve|halloween|valentine'?s day)(?:\s+$YEAR)?""") { m, t ->
            val (mo, d, len) = when {
                m.groupValues[1].lowercase().let { it.startsWith("christmas") || it == "xmas" } -> Triple(12, 24, 3L)
                m.groupValues[1].lowercase().startsWith("new") -> Triple(12, 31, 2L)
                m.groupValues[1].lowercase() == "halloween" -> Triple(10, 31, 1L)
                else -> Triple(2, 14, 1L)
            }
            val y = m.groupValues[2].toIntOrNull() ?: if (LocalDate.of(t.year, mo, d) <= t) t.year else t.year - 1
            LocalDate.of(y, mo, d).let { DateRange(it, it.plusDays(len)) }
        },
        Rule("""(?:(in|during|this|last)\s+)?($MONTH_RE)\.?(?:\s+(?:of\s+)?$YEAR)?""") { m, t ->
            val word = m.groupValues[2].lowercase()
            val y = m.groupValues[3].toIntOrNull()
            val mod = m.groupValues[1].lowercase()
            // "may"/"mar"/"march"/"jan" etc. are also ordinary words: only treat them as dates with a year or a cue word
            if (y == null && mod.isEmpty() && word in setOf("may", "march", "mar", "jan", "jun", "aug", "sept", "sep", "oct", "nov", "dec", "feb", "apr", "jul")) return@Rule null
            val mo = month(word)
            if (y != null) return@Rule monthRange(y, mo)
            val thisYear = LocalDate.of(t.year, mo, 1)
            when (mod) {
                "this" -> monthRange(t.year, mo)
                "last" -> if (thisYear.plusMonths(1) <= t) monthRange(t.year, mo) else monthRange(t.year - 1, mo)
                else -> if (thisYear <= t) monthRange(t.year, mo) else monthRange(t.year - 1, mo)
            }
        },
        Rule("""(?:the\s+)?(?:(19|20)?(\d)0)'?s""") { m, _ ->
            val century = m.groupValues[1].ifEmpty { if (m.groupValues[2].toInt() >= 3) "19" else "20" }
            val start = (century + m.groupValues[2] + "0").toInt()
            DateRange(LocalDate.of(start, 1, 1), LocalDate.of(start + 10, 1, 1))
        },
        Rule(YEAR) { m, t -> m.groupValues[1].toInt().takeIf { it <= t.year + 1 }?.let(::year) },
    )

    private val DANGLING = Regex("""(?i)(?:^|\s)(?:taken|shot|in|from|during|on|of|at|around|between)\s*$""")

    fun parse(query: String, today: LocalDate = LocalDate.now()): ParsedQuery {
        for (rule in RULES) {
            for (m in rule.re.findAll(query)) {
                val range = rule.f(m, today) ?: continue
                var text = query.removeRange(m.range)
                // drop a preposition left hanging in front of the phrase ("dogs in", "receipts from")
                var before = text.substring(0, m.range.first).trimEnd()
                while (true) { val d = DANGLING.find(before) ?: break; if (d.value.isBlank()) break; before = before.removeRange(d.range).trimEnd() }
                text = (before + " " + text.substring(m.range.first)).trim()
                return ParsedQuery(text.replace(Regex("\\s+"), " ").trim(' ', ',', '.'), range, m.value)
            }
        }
        return ParsedQuery(query.trim(), null)
    }

}
