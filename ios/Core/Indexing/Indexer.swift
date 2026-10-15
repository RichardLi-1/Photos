import Foundation
import Photos
import UIKit

struct IndexProgress: Sendable, Equatable {
    enum State: Sendable, Equatable {
        case idle, preparing, running, finished
        case paused(PauseReason)
    }
    var state: State = .idle
    var total = 0
    var embedded = 0
    var skipped = 0

    var fraction: Double { total == 0 ? 0 : Double(embedded + skipped) / Double(total) }
}

enum PauseReason: Sendable, Equatable {
    case thermal, lowPower, lowBattery, lowDisk, userPaused
}

/// When indexing should hold off: thermal pressure, Low Power Mode, low battery on battery power, little free disk.
enum DeviceConditions {
    @MainActor
    static func pauseReason() -> PauseReason? {
        let info = ProcessInfo.processInfo
        if info.thermalState == .serious || info.thermalState == .critical { return .thermal }
        if info.isLowPowerModeEnabled { return .lowPower }
        let device = UIDevice.current
        device.isBatteryMonitoringEnabled = true
        if device.batteryState == .unplugged && device.batteryLevel >= 0 && device.batteryLevel < 0.2 { return .lowBattery }
        if let free = try? URL.applicationSupportDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
            .volumeAvailableCapacityForImportantUsage, free < 500_000_000 { return .lowDisk }
        return nil
    }
}

/// Incremental, resumable indexing of the photo library: PhotoKit → Core ML embedding (batched) + Vision OCR → SQLite.
/// Progress is committed per batch, so it picks up where it left off after a kill, a pause or a background expiry.
actor Indexer {
    private let store: LibraryStore
    private let engine: SearchEngine
    private let encoder: SigLIPEncoder
    private let library = PhotoLibrary.shared
    private let places = PlaceResolver()
    private var running: Task<Void, Never>?
    private var userPaused = false
    private var continuations: [UUID: AsyncStream<IndexProgress>.Continuation] = [:]
    private(set) var progress = IndexProgress() {
        didSet { for c in continuations.values { c.yield(progress) } }
    }

    static let batchSize = 8

    init(store: LibraryStore, engine: SearchEngine, encoder: SigLIPEncoder) {
        self.store = store; self.engine = engine; self.encoder = encoder
    }

    func updates() -> AsyncStream<IndexProgress> {
        let id = UUID()
        return AsyncStream { c in
            c.yield(progress)
            continuations[id] = c
            c.onTermination = { _ in Task { await self.drop(id) } }
        }
    }

    private func drop(_ id: UUID) { continuations[id] = nil }

    func setPaused(_ paused: Bool) {
        userPaused = paused
        if !paused { start() }
    }

    /// Starts (or keeps) a background-priority indexing pass.
    func start() {
        guard running == nil else { return }
        running = Task(priority: .background) { [weak self] in
            await self?.runPass()
            await self?.finishPass()
        }
    }

    func stop() {
        running?.cancel()
        running = nil
    }

    /// Runs one pass to completion or cancellation (used by the BGProcessingTask too).
    func runToCompletion() async {
        start()
        await running?.value
    }

    private func finishPass() { running = nil }

    private func runPass() async {
        progress.state = .preparing
        let fetch = library.fetchImages()
        let infos = library.assetInfos(fetch)
        do {
            let (_, removed) = try await store.sync(infos)
            try await engine.remove(removed)
            try await refreshCounts()
            while !Task.isCancelled {
                if let reason = await pauseReason() {
                    progress.state = .paused(reason)
                    try await Task.sleep(for: .seconds(reason == .userPaused ? 2 : 30))
                    continue
                }
                progress.state = .running
                let batch = try await store.pending(limit: Self.batchSize)
                if batch.isEmpty { break }
                try await process(batch)
                try await refreshCounts()
                await Task.yield()
            }
            if !Task.isCancelled && PlaceResolver.isEnabled { try await resolvePlaces() }
            if !Task.isCancelled { progress.state = .finished }
        } catch {
            progress.state = .idle  // cancelled or storage failure; the next pass resumes from the database
        }
    }

    private func pauseReason() async -> PauseReason? {
        if userPaused { return .userPaused }
        return await DeviceConditions.pauseReason()
    }

    private func refreshCounts() async throws {
        let c = try await store.counts()
        progress.total = c.total
        progress.embedded = c.embedded
        progress.skipped = c.failed
    }

    private func process(_ batch: [LibraryStore.Pending]) async throws {
        let assets = Dictionary(uniqueKeysWithValues: library.assets(batch.map(\.localID)).map { ($0.localIdentifier, $0) })
        var results: [LibraryStore.Result] = []
        var images: [(index: Int, image: CGImage)] = []
        for p in batch {
            var r = LibraryStore.Result(id: p.id, embedding: nil, ocr: nil, ocrDone: false, failed: false)
            guard let asset = assets[p.localID], let image = await library.cgImage(for: asset) else {
                r.failed = true
                results.append(r)
                continue
            }
            if p.needsOCR {
                r.ocr = TextRecognizer.recognize(image)
                r.ocrDone = true
            }
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

    /// Opt-in only (see PlaceResolver): fills in place names for photos with GPS.
    private func resolvePlaces() async throws {
        while !Task.isCancelled, PlaceResolver.isEnabled, await pauseReason() == nil {
            let todo = try await store.pendingPlaces(limit: 20)
            if todo.isEmpty { return }
            for item in todo {
                let name: String?
                do { name = try await places.place(lat: item.lat, lon: item.lon) } catch { return }  // offline: retry later
                try await store.setPlace(name, for: item.id)
                await engine.updatePlace(name, for: item.id)
            }
        }
    }
}
