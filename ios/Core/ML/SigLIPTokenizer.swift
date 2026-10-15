import Foundation

/// SigLIP2's Gemma BPE tokenizer, reimplemented to match HF `tokenizers` + open_clip's "canonicalize" cleaning.
///
/// Pipeline (see ml/export_siglip2_tokenizer.py): clean (lowercase, drop ASCII punctuation, collapse whitespace)
/// → spaces become "▁" → BPE over Unicode scalars with ranked merges (byte fallback `<0xXX>`, unknowns fused to
/// `<unk>`) → truncate to 63 → append `<eos>` → pad with 0 to 64.
/// Not supported (unreachable after cleaning): added tokens such as runs of "\n" or "▁▁", ftfy mojibake repair.
/// Foundation-only, so it also compiles in a plain macOS `swiftc` check.
final class SigLIPTokenizer: @unchecked Sendable {
    let contextLength: Int
    let padID: Int32, eosID: Int32, unkID: Int32
    private let chars: [UInt32: Int32]
    private let byteIDs: [Int32]
    private let merges: Data        // u32 n | u64 keys[n] (sorted) | u32 ranks[n] | u32 merged[n]
    private let count: Int

    private struct Meta: Decodable {
        let context_length: Int
        let pad_id: Int32, eos_id: Int32, unk_id: Int32
        let chars: [String: Int32]
        let bytes: [Int32]
    }

    init(metaURL: URL, mergesURL: URL) throws {
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        contextLength = meta.context_length
        padID = meta.pad_id; eosID = meta.eos_id; unkID = meta.unk_id
        chars = Dictionary(uniqueKeysWithValues: meta.chars.compactMap { k, v in UInt32(k).map { ($0, v) } })
        byteIDs = meta.bytes
        merges = try Data(contentsOf: mergesURL, options: .alwaysMapped)
        count = Int(merges.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
        guard merges.count == 4 + count * 16 else { throw CocoaError(.fileReadCorruptFile) }
    }

    convenience init(bundle: Bundle = .main) throws {
        guard let meta = bundle.url(forResource: "siglip2_tokenizer", withExtension: "json"),
              let merges = bundle.url(forResource: "siglip2_merges", withExtension: "bin")
        else { throw CocoaError(.fileNoSuchFile) }
        try self.init(metaURL: meta, mergesURL: merges)
    }

    /// open_clip `_clean_canonicalize` minus ftfy: unescape HTML, "_"→" ", drop ASCII punctuation, lowercase, collapse spaces.
    static func clean(_ text: String) -> String {
        var s = text
        for (entity, char) in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&#39;", "'"), ("&nbsp;", "\u{a0}")] {
            s = s.replacingOccurrences(of: entity, with: char)
        }
        s = s.trimmingCharacters(in: .whitespacesAndNewlines).replacingOccurrences(of: "_", with: " ")
        let punctuation = Set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".unicodeScalars)
        s = String(String.UnicodeScalarView(s.unicodeScalars.filter { !punctuation.contains($0) })).lowercased()
        return s.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    /// Token ids padded to `contextLength`, exactly what the text encoder expects.
    func encode(_ text: String) -> [Int32] {
        var ids = bpe(Self.clean(text).replacingOccurrences(of: " ", with: "\u{2581}"))
        ids = Array(ids.prefix(contextLength - 1)) + [eosID]
        return ids + Array(repeating: padID, count: contextLength - ids.count)
    }

    /// BPE: start from scalars (byte fallback for unknowns), repeatedly merge the adjacent pair with the lowest rank
    /// (leftmost on ties) — the same order as HF's priority queue.
    func bpe(_ word: String) -> [Int32] {
        guard !word.isEmpty else { return [] }
        var symbols: [Int32] = []
        for scalar in word.unicodeScalars {
            if let id = chars[scalar.value] {
                symbols.append(id)
                continue
            }
            let bytes = Array(String(scalar).utf8).map { byteIDs[Int($0)] }
            if bytes.allSatisfy({ $0 >= 0 }) {
                symbols += bytes
            } else if symbols.last != unkID {  // fuse_unk
                symbols.append(unkID)
            }
        }
        while symbols.count > 1 {
            var best: (rank: UInt32, index: Int, id: Int32)?
            for i in 0..<(symbols.count - 1) {
                if let m = merge(symbols[i], symbols[i + 1]), m.rank < (best?.rank ?? .max) {
                    best = (m.rank, i, m.id)
                }
            }
            guard let best else { break }
            symbols[best.index] = best.id
            symbols.remove(at: best.index + 1)
        }
        return symbols
    }

    /// Binary search over the memory-mapped, sorted merge keys.
    private func merge(_ left: Int32, _ right: Int32) -> (rank: UInt32, id: Int32)? {
        let key = UInt64(UInt32(bitPattern: left)) << 32 | UInt64(UInt32(bitPattern: right))
        return merges.withUnsafeBytes { raw -> (UInt32, Int32)? in
            var lo = 0, hi = count - 1
            while lo <= hi {
                let mid = (lo + hi) / 2
                let k = raw.loadUnaligned(fromByteOffset: 4 + mid * 8, as: UInt64.self)
                if k == key {
                    let rank = raw.loadUnaligned(fromByteOffset: 4 + count * 8 + mid * 4, as: UInt32.self)
                    let id = raw.loadUnaligned(fromByteOffset: 4 + count * 12 + mid * 4, as: UInt32.self)
                    return (rank, Int32(bitPattern: id))
                }
                if k < key { lo = mid + 1 } else { hi = mid - 1 }
            }
            return nil
        }
    }
}
