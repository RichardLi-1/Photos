import Foundation

/// Process-wide services for the Mac app (the counterpart of ios/Core/AppServices.swift): the shared SQLite store and
/// search engine from ios/Core, the Core ML encoder, the "same place" head and the folder/Photos indexer.
/// The index lives in the app's sandbox container (Application Support), never next to the user's photos.
actor DesktopServices {
    nonisolated let store: LibraryStore
    nonisolated let engine: SearchEngine
    nonisolated let reader: IndexReader
    nonisolated let directory: URL
    private var encoderTask: Task<SigLIPEncoder, Error>?
    private var indexerInstance: DesktopIndexer?
    private var placeIndexInstance: PlaceIndex?
    private var headFailed = false

    static let shared: DesktopServices = {
        do { return try DesktopServices(directory: LibraryStore.defaultDirectory()) } catch {
            // Application Support unavailable (should not happen): keep working with a throwaway index.
            return try! DesktopServices(directory: FileManager.default.temporaryDirectory.appendingPathComponent("PhotoIndex"))
        }
    }()

    init(directory: URL) throws {
        self.directory = directory
        store = try LibraryStore(directory: directory)
        engine = SearchEngine(store: store)
        reader = try IndexReader(directory: directory)
    }

    /// Loads the Core ML models once (the first load specialises them for the Neural Engine and takes a few seconds).
    func encoder() async throws -> SigLIPEncoder {
        if encoderTask == nil {
            let engine = engine
            encoderTask = Task.detached(priority: .userInitiated) {
                let encoder = try SigLIPEncoder()
                let explainer = try? ConceptExplainer(calibration: encoder.calibration)
                await engine.attach(encoder: encoder, explainer: explainer)
                try await engine.reload()
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

    func indexer() async throws -> DesktopIndexer {
        if let indexerInstance { return indexerInstance }
        let place = placeIndex()
        let i = DesktopIndexer(store: store, engine: engine, encoder: try await encoder(), reader: reader) {
            await place?.invalidate()
        }
        indexerInstance = i
        return i
    }

    /// nil when the head's weights aren't bundled (then "Same place" is hidden).
    func placeIndex() -> PlaceIndex? {
        if let placeIndexInstance { return placeIndexInstance }
        guard !headFailed, let head = try? PlaceHead() else { headFailed = true; return nil }
        let p = PlaceIndex(head: head, store: store)
        placeIndexInstance = p
        return p
    }
}
