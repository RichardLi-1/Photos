import CoreGraphics
import Foundation

/// What to index: folders the user picked (already security-scope-accessed by the caller) and, optionally, Photos.
struct IndexSources: Sendable, Equatable {
    var folders: [URL] = []
    var photosLibrary = false
    /// Folders whose bookmark could not be resolved right now (e.g. an unplugged drive): their photos are kept, not dropped.
    var unavailable: [URL] = []
}

struct IndexStatus: Sendable, Equatable {
    enum State: Sendable, Equatable { case idle, scanning, running, paused(String), finished }
    var state: State = .idle
    var total = 0
    var embedded = 0
    var skipped = 0
    var fraction: Double { total == 0 ? 0 : Double(embedded + skipped) / Double(total) }
    var isBusy: Bool { state == .scanning || state == .running }
}

/// Incremental, resumable indexing on the Mac: folders (ImageIO) + optional Photos library → Core ML embedding (batched)
/// + Vision OCR → the shared SQLite `LibraryStore`. Progress is committed per batch; a later pass resumes from the DB.
/// Mirrors ios/Core/Indexing/Indexer.swift, without UIKit/PhotoKit-only assumptions.
actor DesktopIndexer {
    private let store: LibraryStore
    private let engine: SearchEngine
    private let encoder: SigLIPEncoder
    private let reader: IndexReader
    private let onBatch: @Sendable () async -> Void
    private var sources = IndexSources()
    private var running: Task<Void, Never>?
    private var rerun = false
    private var userPaused = false
    private var continuations: [UUID: AsyncStream<IndexStatus>.Continuation] = [:]
    private(set) var status = IndexStatus() {
        didSet { if status != oldValue { for c in continuations.values { c.yield(status) } } }
    }

    static let batchSize = 8
    static let maxSide = 1024  // same rendition size as the iOS app; the encoder squashes it to 384×384

    init(store: LibraryStore, engine: SearchEngine, encoder: SigLIPEncoder, reader: IndexReader,
         onBatch: @escaping @Sendable () async -> Void = {}) {
        self.store = store; self.engine = engine; self.encoder = encoder; self.reader = reader; self.onBatch = onBatch
    }

    func updates() -> AsyncStream<IndexStatus> {
        let id = UUID()
        return AsyncStream { c in
            c.yield(status)
            continuations[id] = c
            c.onTermination = { _ in Task { await self.drop(id) } }
        }
    }

    private func drop(_ id: UUID) { continuations[id] = nil }

    func setPaused(_ paused: Bool) {
        userPaused = paused
        if !paused { start() }
    }

    /// Starts a pass over `sources` (or the last ones). If a pass is running, another one follows it.
    func start(_ newSources: IndexSources? = nil) {
        if let newSources { sources = newSources }
        guard running == nil else { rerun = true; return }
        running = Task(priority: .utility) { [weak self] in
            await self?.runPass()
            await self?.finishPass()
        }
    }

    func stop() {
        running?.cancel()
        running = nil
        rerun = false
    }

    func runToCompletion(_ newSources: IndexSources? = nil) async {
        start(newSources)
        while let task = running { await task.value }
    }

    private func finishPass() {
        running = nil
        if rerun { rerun = false; start() }
    }

    // MARK: - One pass

    private func runPass() async {
        status.state = .scanning
        do {
            let infos = try await listing()
            let (_, removed) = try await store.sync(infos)
            try await engine.remove(removed)
            try await refreshCounts()
            while !Task.isCancelled {
                if let reason = await pauseReason() {
                    status.state = .paused(reason)
                    try await Task.sleep(for: .seconds(userPaused ? 1 : 30))
                    continue
                }
                status.state = .running
                let batch = try await store.pending(limit: Self.batchSize)
                if batch.isEmpty { break }
                try await process(batch)
                try await refreshCounts()
                await onBatch()
            }
            if !Task.isCancelled { status.state = .finished }
        } catch {
            status.state = .idle  // cancelled or storage failure; the next pass resumes from the database
        }
    }

    /// Every photo the sources currently contain. Files already indexed with the same modification time reuse their
    /// stored metadata, so a re-scan of a big folder only opens new or edited files.
    private func listing() async throws -> [AssetInfo] {
        let known = try await reader.known()
        let sources = self.sources
        return await Task.detached(priority: .utility) {
            var out: [AssetInfo] = []
            var seen = Set<String>()
            for folder in sources.folders {
                for entry in FolderScanner.list(folder) {
                    let id = PhotoRef.file(entry.url).localID
                    guard seen.insert(id).inserted else { continue }  // overlapping folders
                    if let k = known[id], let m = entry.modified?.timeIntervalSince1970, let old = k.modified, abs(m - old) <= 1 {
                        out.append(k.info)
                    } else {
                        out.append(FolderScanner.info(entry))
                    }
                }
            }
            // Keep (don't drop) photos from folders that are offline right now. Turning the Photos library off, or
            // removing a folder, does drop its photos from the index.
            let offline = sources.unavailable.map { $0.standardizedFileURL.absoluteString }
            if !offline.isEmpty {
                for (id, k) in known where !seen.contains(id) && offline.contains(where: { id.hasPrefix($0) }) {
                    out.append(k.info); seen.insert(id)
                }
            }
            if sources.photosLibrary {
                out += PhotosLibrarySource.list().filter { seen.insert($0.localID).inserted }
            }
            return out
        }.value
    }

    private func pauseReason() async -> String? {
        if userPaused { return String(localized: "Paused") }
        let info = ProcessInfo.processInfo
        if info.thermalState == .serious || info.thermalState == .critical { return String(localized: "Mac is hot") }
        if info.isLowPowerModeEnabled { return String(localized: "Low Power Mode") }
        if let free = try? URL.applicationSupportDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            .volumeAvailableCapacityForImportantUsage, free < 1_000_000_000 { return String(localized: "Disk almost full") }
        return nil
    }

    private func refreshCounts() async throws {
        let c = try await store.counts()
        status.total = c.total
        status.embedded = c.embedded
        status.skipped = c.failed
    }

    /// Decode + OCR concurrently (ImageIO and Vision are thread-safe per request), then one batched Core ML call.
    private func process(_ batch: [LibraryStore.Pending]) async throws {
        let decoded: [(LibraryStore.Pending, CGImage?, String?)] = await withTaskGroup(of: (Int, CGImage?, String?).self) { group in
            for (i, p) in batch.enumerated() {
                group.addTask {
                    guard let ref = PhotoRef(localID: p.localID), let image = await ref.cgImage(maxSide: Self.maxSide) else {
                        return (i, nil, nil)
                    }
                    return (i, image, p.needsOCR ? TextRecognizer.recognize(image) : nil)
                }
            }
            var out = [(LibraryStore.Pending, CGImage?, String?)](repeating: (batch[0], nil, nil), count: batch.count)
            for await (i, image, text) in group { out[i] = (batch[i], image, text) }
            return out
        }
        var results: [LibraryStore.Result] = []
        var images: [(index: Int, image: CGImage)] = []
        for (p, image, text) in decoded {
            var r = LibraryStore.Result(id: p.id, embedding: nil, ocr: nil, ocrDone: false, failed: false)
            guard let image else {
                r.failed = true
                results.append(r)
                continue
            }
            if p.needsOCR { r.ocr = text; r.ocrDone = true }
            if p.needsEmbedding { images.append((results.count, image)) }
            results.append(r)
        }
        if !images.isEmpty {
            let vectors = try encoder.embed(images: images.map(\.image))
            for (k, item) in images.enumerated() { results[item.index].embedding = vectors[k] }
        }
        try await store.save(results)
        let records = try await store.records(results.filter { $0.embedding != nil }.map(\.id))
        let vectorByID = Dictionary(uniqueKeysWithValues: results.compactMap { r in r.embedding.map { (r.id, $0) } })
        await engine.upsert(records.compactMap { rec in vectorByID[rec.id].map { (rec, $0) } })
    }
}
