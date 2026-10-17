package app.photosearch.ml

import java.text.Normalizer
import java.util.Locale

/**
 * open_clip's 'canonicalize' clean for SigLIP2, which runs before the tokenizer:
 *   ftfy.fix_text -> html.unescape twice -> strip -> '_' to ' ' -> drop ASCII punctuation -> lower -> collapse whitespace.
 * ftfy is approximated by what matters for typed queries: curly quotes, ligatures, full-width forms, control chars, NFC.
 * (Mojibake repair is not reproduced; a search box never receives double-encoded UTF-8.)
 */
object TextClean {
    private const val PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    fun canonicalize(text: String): String {
        var t = ftfyLite(text)
        t = htmlUnescape(htmlUnescape(t))
        t = t.replace('_', ' ')
        val sb = StringBuilder(t.length)
        for (ch in t) if (PUNCT.indexOf(ch) < 0) sb.append(ch)
        t = sb.toString().lowercase(Locale.ROOT)
        return splitWhitespace(t).joinToString(" ")
    }

    /** Python str.split(): runs of Unicode whitespace (incl. NBSP and the \x1c-\x1f separators). */
    fun splitWhitespace(s: String): List<String> {
        val out = ArrayList<String>()
        var start = -1
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            val ws = isPyWhitespace(cp)
            if (ws && start >= 0) { out += s.substring(start, i); start = -1 }
            if (!ws && start < 0) start = i
            i += Character.charCount(cp)
        }
        if (start >= 0) out += s.substring(start)
        return out
    }

    private fun isPyWhitespace(cp: Int) =
        Character.isWhitespace(cp) || Character.isSpaceChar(cp) || cp == 0x85 || cp in 0x1c..0x1f

    private fun ftfyLite(s: String): String {
        val sb = StringBuilder(s.length)
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            i += Character.charCount(cp)
            when {
                cp == 0x2018 || cp == 0x2019 || cp == 0x201A || cp == 0x201B -> sb.append('\'') // uncurl_quotes
                cp == 0x201C || cp == 0x201D || cp == 0x201E || cp == 0x201F -> sb.append('"')
                cp in 0xFB00..0xFB06 -> sb.append(Normalizer.normalize(String(Character.toChars(cp)), Normalizer.Form.NFKC)) // ligatures
                cp in 0xFF01..0xFF5E -> sb.appendCodePoint(cp - 0xFEE0) // full-width ASCII -> ASCII
                cp == 0x3000 -> sb.append(' ')
                cp in 0xFF61..0xFF9F -> sb.append(Normalizer.normalize(String(Character.toChars(cp)), Normalizer.Form.NFKC)) // half-width kana
                cp == '\r'.code -> sb.append('\n')
                cp < 0x20 && cp != '\t'.code && cp != '\n'.code && cp != 0x0C -> {} // remove_control_chars
                cp in 0x7F..0x9F -> {}
                cp == 0x200B || cp == 0xFEFF -> {}
                else -> sb.appendCodePoint(cp)
            }
        }
        return Normalizer.normalize(sb, Normalizer.Form.NFC)
    }

    private val ENTITIES = mapOf(
        "amp" to "&", "lt" to "<", "gt" to ">", "quot" to "\"", "apos" to "'", "nbsp" to " ",
        "copy" to "©", "reg" to "®", "eacute" to "é", "hellip" to "…", "mdash" to "—", "ndash" to "–",
    )
    private val ENTITY = Regex("&(#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z]{2,8});")

    fun htmlUnescape(s: String): String {
        if ('&' !in s) return s
        return ENTITY.replace(s) { m ->
            val e = m.groupValues[1]
            when {
                e.startsWith("#x") || e.startsWith("#X") -> e.substring(2).toIntOrNull(16)?.let(::cp) ?: m.value
                e.startsWith("#") -> e.substring(1).toIntOrNull()?.let(::cp) ?: m.value
                else -> ENTITIES[e] ?: m.value
            }
        }
    }

    private fun cp(c: Int): String = if (c in 1..0x10FFFF && c !in 0xD800..0xDFFF) String(Character.toChars(c)) else "�"
}
