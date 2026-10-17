package app.photosearch.ml

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * SigLIP2's Gemma BPE tokenizer (HF tokenizer.json semantics), from the binary written by ml/android/export_tokenizer.py.
 *
 * clean (TextClean) -> ' ' becomes '▁' -> one word -> Unicode scalars (byte fallback <0xXX>, else <unk>, fused)
 * -> repeatedly merge the adjacent pair with the lowest rank (leftmost on ties) -> append <eos> -> truncate/pad to 64.
 */
class GemmaTokenizer(buf: ByteBuffer) {
    val contextLength: Int
    val padId: Int
    val eosId: Int
    val unkId: Int
    private val charCps: IntArray
    private val charIds: IntArray
    private val byteIds: IntArray
    private val mergeKeys: LongArray
    private val mergeRanks: IntArray
    private val mergeIds: IntArray

    init {
        val b = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        val magic = ByteArray(4).also { b.get(it) }
        require(String(magic, Charsets.US_ASCII) == "SGTK") { "not a SigLIP tokenizer file" }
        require(b.int == 1) { "unsupported tokenizer version" }
        contextLength = b.int; padId = b.int; eosId = b.int; unkId = b.int
        val nChars = b.int
        charCps = IntArray(nChars).also { b.asIntBuffer().get(it); b.position(b.position() + 4 * nChars) }
        charIds = IntArray(nChars).also { b.asIntBuffer().get(it); b.position(b.position() + 4 * nChars) }
        byteIds = IntArray(256).also { b.asIntBuffer().get(it); b.position(b.position() + 4 * 256) }
        val n = b.int
        mergeKeys = LongArray(n).also { b.asLongBuffer().get(it); b.position(b.position() + 8 * n) }
        mergeRanks = IntArray(n).also { b.asIntBuffer().get(it); b.position(b.position() + 4 * n) }
        mergeIds = IntArray(n).also { b.asIntBuffer().get(it) }
    }

    /** Token ids for one query, always exactly [contextLength] long (pad = 0). */
    fun encode(text: String): IntArray {
        val ids = bpe(TextClean.canonicalize(text).replace(' ', '▁'))
        val out = IntArray(contextLength) { padId }
        val n = minOf(ids.size, contextLength - 1)
        for (i in 0 until n) out[i] = ids[i]
        out[n] = eosId
        return out
    }

    internal fun bpe(word: String): IntArray {
        if (word.isEmpty()) return IntArray(0)
        val syms = ArrayList<Int>(word.length)
        var i = 0
        while (i < word.length) {
            val cp = word.codePointAt(i)
            i += Character.charCount(cp)
            val k = charCps.binarySearch(cp)
            if (k >= 0) { syms += charIds[k]; continue }
            val bytes = String(Character.toChars(cp)).toByteArray(Charsets.UTF_8)
            val fb = bytes.map { byteIds[it.toInt() and 0xff] }
            if (fb.all { it >= 0 }) syms += fb
            else if (syms.isEmpty() || syms.last() != unkId) syms += unkId // fuse_unk
        }
        // Naive lowest-rank-first merging; queries are short so O(n^2) is fine and exactly matches HF's priority queue.
        while (syms.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestAt = -1
            var bestId = -1
            for (j in 0 until syms.size - 1) {
                val m = lookup(syms[j], syms[j + 1])
                if (m >= 0 && mergeRanks[m] < bestRank) { bestRank = mergeRanks[m]; bestAt = j; bestId = mergeIds[m] }
            }
            if (bestAt < 0) break
            syms[bestAt] = bestId
            syms.removeAt(bestAt + 1)
        }
        return syms.toIntArray()
    }

    private fun lookup(left: Int, right: Int): Int {
        val key = (left.toLong() shl 32) or (right.toLong() and 0xffffffffL)
        val k = mergeKeys.binarySearch(key)
        return if (k >= 0) k else -1
    }
}
