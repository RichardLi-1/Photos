package app.photosearch

import app.photosearch.search.DateParser
import app.photosearch.search.DateRange
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.time.LocalDate

class DateParserTest {
    private val today = LocalDate.of(2026, 10, 15) // a Thursday

    private fun d(s: String) = LocalDate.parse(s)
    private fun check(q: String, text: String, start: String?, end: String?) {
        val p = DateParser.parse(q, today)
        assertEquals("text of '$q'", text, p.text)
        assertEquals("range of '$q'", DateRange(start?.let(::d), end?.let(::d)), p.range)
    }

    @Test fun years() {
        check("beach in 2019", "beach", "2019-01-01", "2020-01-01")
        check("2019 beach", "beach", "2019-01-01", "2020-01-01")
        check("photos from 2021 of my dog", "photos of my dog", "2021-01-01", "2022-01-01")
        check("concert before 2020", "concert", null, "2020-01-01")
        check("concert after 2020", "concert", "2021-01-01", null)
        check("cats since 2022", "cats", "2022-01-01", null)
        check("trips between 2018 and 2020", "trips", "2018-01-01", "2021-01-01")
        check("party in the 90s", "party", "1990-01-01", "2000-01-01")
    }

    @Test fun relative() {
        check("dinner today", "dinner", "2026-10-15", "2026-10-16")
        check("receipt from yesterday", "receipt", "2026-10-14", "2026-10-15")
        check("dogs last year", "dogs", "2025-01-01", "2026-01-01")
        check("this year", "", "2026-01-01", "2027-01-01")
        check("whiteboard last week", "whiteboard", "2026-10-05", "2026-10-12")
        check("whiteboard last month", "whiteboard", "2026-09-01", "2026-10-01")
        check("cake 3 days ago", "cake", "2026-10-12", "2026-10-13")
        check("hike two weeks ago", "hike", "2026-09-28", "2026-10-05")
        check("snow a year ago", "snow", "2025-01-01", "2026-01-01")
        check("brunch last weekend", "brunch", "2026-10-10", "2026-10-12")
    }

    @Test fun seasonsAndMonths() {
        check("dogs last summer", "dogs", "2026-06-01", "2026-09-01")
        check("skiing last winter", "skiing", "2025-12-01", "2026-03-01")
        check("beach summer 2019", "beach", "2019-06-01", "2019-09-01")
        check("leaves in autumn", "leaves", "2026-09-01", "2026-12-01")
        check("receipts from march 2023", "receipts", "2023-03-01", "2023-04-01")
        check("flowers in may", "flowers", "2026-05-01", "2026-06-01")
        check("snow in december", "snow", "2025-12-01", "2026-01-01")
        check("tree at christmas", "tree", "2025-12-24", "2025-12-27")
        check("costumes halloween 2024", "costumes", "2024-10-31", "2024-11-01")
    }

    @Test fun notDates() {
        assertNull(DateParser.parse("may the force be with you", today).range)
        assertNull(DateParser.parse("march of the penguins", today).range)
        assertNull(DateParser.parse("red vintage car", today).range)
        assertNull(DateParser.parse("route 66 sign", today).range)
        assertEquals("route 66 sign", DateParser.parse("route 66 sign", today).text)
    }
}
