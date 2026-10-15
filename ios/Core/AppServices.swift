import BackgroundTasks
import Foundation

/// Process-wide, UI-independent services (shared by the UI, the BGProcessingTask and PhotoTools).
actor AppServices {
    static let shared = AppServices()

    nonisolated let store: LibraryStore
    nonisolated let engine: SearchEngine
    private var encoderTask: Task<SigLIPEncoder, Error>?
    private var indexerInstance: Indexer?

    private init() {
        let store: LibraryStore
        do {
            store = try LibraryStore()
        } catch {
            // Application Support unavailable (should not happen): keep working with a throwaway index.
            store = try! LibraryStore(directory: FileManager.default.temporaryDirectory.appendingPathComponent("PhotoIndex"))
        }
        self.store = store
        engine = SearchEngine(store: store)
    }

    /// Loads the Core ML models once (first load compiles them for the Neural Engine, which takes a few seconds).
    func encoder() async throws -> SigLIPEncoder {
        if encoderTask == nil {
            encoderTask = Task.detached(priority: .userInitiated) {
                let encoder = try SigLIPEncoder()
                let explainer = try? ConceptExplainer(calibration: encoder.calibration)
                await self.engine.attach(encoder: encoder, explainer: explainer)
                try await self.engine.reload()
                return encoder
            }
        }
        do {
            return try await encoderTask!.value
        } catch {
            encoderTask = nil
            throw error
        }
    }

    func indexer() async throws -> Indexer {
        if let indexerInstance { return indexerInstance }
        let i = Indexer(store: store, engine: engine, encoder: try await encoder())
        indexerInstance = i
        return i
    }

    var loadedIndexer: Indexer? { indexerInstance }
}

/// Indexing while the app is in the background: a BGProcessingTask (needs power, no network), registered in
/// Info.plist under BGTaskSchedulerPermittedIdentifiers.
enum BackgroundIndexing {
    static let identifier = "RLProjects.Photos.index"

    /// Call before the app finishes launching.
    static func register() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: identifier, using: nil) { task in
            guard let task = task as? BGProcessingTask else { return task.setTaskCompleted(success: false) }
            handle(task)
        }
    }

    static func schedule() {
        let request = BGProcessingTaskRequest(identifier: identifier)
        request.requiresExternalPower = true
        request.requiresNetworkConnectivity = false
        try? BGTaskScheduler.shared.submit(request)
    }

    private static func handle(_ task: BGProcessingTask) {
        schedule()  // keep going on later nights until the library is done
        let work = Task {
            let indexer = try? await AppServices.shared.indexer()
            await indexer?.runToCompletion()
            task.setTaskCompleted(success: !Task.isCancelled)
        }
        task.expirationHandler = {
            work.cancel()
            Task { await AppServices.shared.loadedIndexer?.stop() }
        }
    }
}
