import Observation
import Photos
import SwiftUI

/// What the grid shows: the whole library (lazy PHFetchResult) or ranked search results.
struct GridItems {
    var fetch: PHFetchResult<PHAsset>?
    var assets: [PHAsset] = []
    var version = 0  // bumps whenever the source changes, so the grid resets scroll and identity

    var count: Int { fetch?.count ?? assets.count }
    subscript(i: Int) -> PHAsset { fetch?.object(at: i) ?? assets[i] }
}

enum ActiveQuery: Equatable {
    case text(String)
    case image(UIImage, refine: String)
    case similar(localID: String)
}

@MainActor @Observable
final class AppModel {
    var access: PHAuthorizationStatus = PhotoLibrary.shared.status
    var items = GridItems()
    var query: ActiveQuery?
    var queryTags: [String] = []
    var status: LocalizedStringResource?
    var isSearching = false
    var searchFailures = 0          // drives the error haptic
    var progress = IndexProgress()
    var modelError: String?
    var picked: String?             // localID of the photo whose match card is open
    var searchText = ""

    private let services = AppServices.shared
    private var searchTask: Task<Void, Never>?
    private var observer: LibraryObserver?
    private var progressTask: Task<Void, Never>?
    private var explainText: String?        // the semantic part of the last text query

    var isLimited: Bool { access == .limited }
    var hasAccess: Bool { access == .authorized || access == .limited }

    // MARK: - Lifecycle

    func start() async {
        access = PhotoLibrary.shared.status
        guard hasAccess else { return }
        if observer == nil {  // first time with access; later changes arrive through the observer
            observer = LibraryObserver { [weak self] in Task { @MainActor in self?.libraryChanged() } }
            reloadBrowse()
        }
        await startIndexing()
    }

    func requestAccess() async {
        access = await PhotoLibrary.shared.requestAccess()
        await start()
    }

    private func startIndexing() async {
        do {
            let indexer = try await services.indexer()
            modelError = nil
            if progressTask == nil {
                progressTask = Task { [weak self] in
                    for await p in await indexer.updates() { self?.progress = p }
                }
            }
            await indexer.start()
        } catch {
            modelError = String(describing: error)
        }
    }

    func setIndexingPaused(_ paused: Bool) async {
        await (try? services.indexer())?.setPaused(paused)
    }

    private func libraryChanged() {
        access = PhotoLibrary.shared.status
        if query == nil { reloadBrowse() }
        Task { await (try? services.indexer())?.start() }
    }

    private func reloadBrowse() {
        items = GridItems(fetch: PhotoLibrary.shared.fetchImages(), version: items.version + 1)
    }

    // MARK: - Search

    func submit() {
        let text = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        if case .image(let image, _) = query {
            run(.image(image, refine: text))
        } else if !text.isEmpty {
            run(.text(text))
        }
    }

    func search(image: UIImage) { run(.image(image, refine: searchText.trimmingCharacters(in: .whitespaces))) }

    func moreLikeThis(_ localID: String) {
        picked = nil
        searchText = ""
        run(.similar(localID: localID))
    }

    func clear() {
        searchTask?.cancel()
        query = nil; queryTags = []; status = nil; searchText = ""; picked = nil; isSearching = false
        reloadBrowse()
    }

    private func run(_ q: ActiveQuery) {
        searchTask?.cancel()
        query = q
        picked = nil
        isSearching = true
        status = "Searching…"
        searchTask = Task { [weak self] in
            guard let self else { return }
            do {
                _ = try await services.encoder()
                let places = (try? await services.store.placeNames()) ?? []
                var request = SearchRequest()
                switch q {
                case .text(let raw):
                    let parsed = QueryParser.parse(raw, places: places)
                    request.text = parsed.text.nilIfEmpty
                    explainText = request.text
                    request.filters = SearchFilters(after: parsed.after, before: parsed.before, place: parsed.place)
                case .image(let image, let refine):
                    guard let cg = PhotoLibrary.cgImage(from: image) else { return }
                    let encoder = try await services.encoder()
                    request.imageVector = try await Task.detached(priority: .userInitiated) { try encoder.embed(images: [cg])[0] }.value
                    request.text = refine.nilIfEmpty
                case .similar(let localID):
                    request.similarTo = try await services.store.id(forLocalID: localID)
                    if request.similarTo == nil { throw SearchEngine.Failure.notIndexed }
                }
                let response = try await services.engine.search(request)
                try Task.checkCancellation()
                queryTags = response.queryTags
                items = GridItems(assets: PhotoLibrary.shared.assets(response.hits.map(\.localID)), version: items.version + 1)
                let ms = Int(response.elapsed.components.seconds * 1000 + response.elapsed.components.attoseconds / 1_000_000_000_000_000)
                status = response.hits.isEmpty ? "No matches" : "\(response.hits.count) best matches · \(ms) ms · on this iPhone"
            } catch is CancellationError {
                return
            } catch SearchEngine.Failure.notIndexed {
                status = "That photo isn’t indexed yet"
                searchFailures += 1
            } catch {
                status = "Search isn’t available yet"
                searchFailures += 1
            }
            isSearching = false
        }
    }

    // MARK: - Match card

    /// Why `localID` matched the current query (the web prototype's match card).
    func explain(_ localID: String) async -> Explanation? {
        guard let id = try? await services.store.id(forLocalID: localID) else { return nil }
        let context: MatchContext
        switch query {
        case .text: context = explainText.map { .text($0) } ?? .browse
        case .image: context = .photo(tags: queryTags)
        case .similar(let ref): context = (try? await services.store.id(forLocalID: ref)).map { .similar($0) } ?? .browse
        case nil: context = .browse
        }
        return try? await services.engine.explain(id, context: context)
    }
}

/// Bridges PHPhotoLibraryChangeObserver (new photos, deletions, limited-selection changes) to a closure.
final class LibraryObserver: NSObject, PHPhotoLibraryChangeObserver, @unchecked Sendable {
    private let onChange: @Sendable () -> Void
    init(_ onChange: @escaping @Sendable () -> Void) {
        self.onChange = onChange
        super.init()
        PHPhotoLibrary.shared().register(self)
    }
    deinit { PHPhotoLibrary.shared().unregisterChangeObserver(self) }
    func photoLibraryDidChange(_ changeInstance: PHChange) { onChange() }
}
