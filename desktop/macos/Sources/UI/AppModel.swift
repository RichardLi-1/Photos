import AppKit
import Foundation
import Observation
import UniformTypeIdentifiers

/// What the grid is showing.
enum GridMode: Equatable {
    case browse
    case text(String)
    case similar(Int64)
    case samePlace(Int64)
    case photo(String)  // a dropped image file name
}

/// A folder the user granted access to, remembered across launches as a security-scoped bookmark (read-only).
struct FolderSource: Identifiable, Hashable {
    var id: String { url.path }
    let url: URL
    var available: Bool
}

@MainActor @Observable
final class AppModel {
    let services: DesktopServices
    private(set) var folders: [FolderSource] = []
    private(set) var includePhotos = UserDefaults.standard.bool(forKey: Keys.photos)
    private(set) var status = IndexStatus()
    private(set) var modelsReady = false
    private(set) var modelError: String?
    private(set) var placeAvailable = false

    var queryText = ""
    private(set) var mode: GridMode = .browse
    private(set) var hits: [SearchHit] = []
    private(set) var queryTags: [String] = []
    private(set) var understood: String?
    private(set) var elapsed: Duration?
    private(set) var searching = false
    var selection: Int64?
    var errorMessage: String?

    private var accessed: [URL] = []
    private var searchTask: Task<Void, Never>?
    private var lastBrowseRefresh = ContinuousClock.now

    private enum Keys { static let bookmarks = "folderBookmarks", photos = "includePhotosLibrary" }

    init(services: DesktopServices = .shared) {
        self.services = services
        restoreFolders()
    }

    // MARK: - Startup

    func start() async {
        do {
            _ = try await services.encoder()
            modelsReady = true
        } catch {
            modelError = String(localized: "The on-device models could not be loaded: \(error.localizedDescription)")
            return
        }
        placeAvailable = await services.placeIndex() != nil
        await refreshBrowse()
        guard let indexer = try? await services.indexer() else { return }
        Task { [weak self] in
            for await s in await indexer.updates() {
                guard let self else { return }
                let finished = s.state == .finished && self.status.state != .finished
                self.status = s
                if case .browse = self.mode, finished || ContinuousClock.now - self.lastBrowseRefresh > .seconds(2) {
                    await self.refreshBrowse()
                }
            }
        }
        await reindex()
    }

    private var sources: IndexSources {
        IndexSources(folders: folders.filter(\.available).map(\.url), photosLibrary: includePhotos && PhotosLibrarySource.isAuthorized,
                     unavailable: folders.filter { !$0.available }.map(\.url))
    }

    func reindex() async {
        guard modelsReady, let indexer = try? await services.indexer() else { return }
        await indexer.start(sources)
    }

    func setPaused(_ paused: Bool) async {
        await (try? services.indexer())?.setPaused(paused)
    }

    // MARK: - Sources

    func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        panel.prompt = String(localized: "Add")
        panel.message = String(localized: "Choose folders of photos to search. Photo Search only reads them; nothing is moved, changed or uploaded.")
        guard panel.runModal() == .OK else { return }
        addFolders(panel.urls)
    }

    func addFolders(_ urls: [URL]) {
        for url in urls {
            guard !folders.contains(where: { $0.url.standardizedFileURL == url.standardizedFileURL }) else { continue }
            _ = url.startAccessingSecurityScopedResource()
            accessed.append(url)
            folders.append(FolderSource(url: url, available: true))
        }
        saveBookmarks()
        Task { await reindex() }
    }

    func removeFolder(_ folder: FolderSource) {
        folders.removeAll { $0.id == folder.id }
        if let i = accessed.firstIndex(of: folder.url) { accessed.remove(at: i).stopAccessingSecurityScopedResource() }
        saveBookmarks()
        Task {
            await reindex()
            await refreshBrowse()
        }
    }

    func setIncludePhotos(_ on: Bool) async {
        if on && !PhotosLibrarySource.isAuthorized {
            let s = await PhotosLibrarySource.requestAccess()
            guard s == .authorized || s == .limited else {
                errorMessage = String(localized: "Photo Search doesn’t have access to Photos. You can allow it in System Settings → Privacy & Security → Photos.")
                return
            }
        }
        includePhotos = on
        UserDefaults.standard.set(on, forKey: Keys.photos)
        await reindex()
    }

    private func restoreFolders() {
        var anyStale = false
        for data in UserDefaults.standard.array(forKey: Keys.bookmarks) as? [Data] ?? [] {
            var stale = false
            guard let url = try? URL(resolvingBookmarkData: data, options: [.withSecurityScope, .withoutUI], relativeTo: nil,
                                     bookmarkDataIsStale: &stale) else { continue }
            let ok = url.startAccessingSecurityScopedResource() && FileManager.default.fileExists(atPath: url.path)
            if ok { accessed.append(url) }
            anyStale = anyStale || (stale && ok)
            folders.append(FolderSource(url: url, available: ok))
        }
        if anyStale { saveBookmarks() }  // the folder moved or was renamed: refresh its bookmark
    }

    private func saveBookmarks() {
        let datas = folders.compactMap { try? $0.url.bookmarkData(options: [.withSecurityScope, .securityScopeAllowOnlyReadAccess],
                                                                    includingResourceValuesForKeys: nil, relativeTo: nil) }
        UserDefaults.standard.set(datas, forKey: Keys.bookmarks)
    }

    // MARK: - Search

    func submitSearch() {
        let q = queryText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return showAll() }
        run(.text(q)) { [services] in
            let places = (try? await services.store.placeNames()) ?? []
            let parsed = QueryParser.parse(q, places: places)
            var req = SearchRequest(text: parsed.text.isEmpty ? nil : parsed.text)
            req.filters = SearchFilters(after: parsed.after, before: parsed.before, place: parsed.place)
            let res = try await services.engine.search(req)
            return (res, Self.describe(parsed))
        }
    }

    func moreLikeThis(_ id: Int64) {
        run(.similar(id)) { [services] in
            var req = SearchRequest()
            req.similarTo = id
            return (try await services.engine.search(req), nil)
        }
    }

    func samePlace(_ id: Int64) {
        run(.samePlace(id)) { [services] in
            guard let place = await services.placeIndex() else { throw SearchEngine.Failure.modelsUnavailable }
            let clock = ContinuousClock(), t = clock.now
            let hits = try await place.samePlace(as: id)
            return (SearchResponse(hits: hits, elapsed: clock.now - t), nil)
        }
    }

    /// Search by a photo dropped onto the window (it doesn't have to be in the library; nothing is stored).
    func searchByImage(_ url: URL) {
        run(.photo(url.lastPathComponent)) { [services] in
            let encoder = try await services.encoder()
            guard let image = FolderScanner.cgImage(url, maxSide: DesktopIndexer.maxSide) else { throw SigLIPEncoder.Failure.badOutput }
            var req = SearchRequest()
            req.imageVector = try encoder.embed(images: [image]).first
            return (try await services.engine.search(req), nil)
        }
    }

    func showAll() {
        searchTask?.cancel()
        mode = .browse
        understood = nil; elapsed = nil; queryTags = []
        selection = nil
        Task { await refreshBrowse() }
    }

    private func run(_ newMode: GridMode, _ work: @escaping @Sendable () async throws -> (SearchResponse, String?)) {
        searchTask?.cancel()
        mode = newMode
        selection = nil
        searching = true
        searchTask = Task {
            do {
                let (res, understood) = try await work()
                guard !Task.isCancelled else { return }
                hits = res.hits
                queryTags = res.queryTags
                elapsed = res.elapsed
                self.understood = understood
            } catch {
                guard !Task.isCancelled else { return }
                hits = []
                errorMessage = String(localized: "Search failed: \(error.localizedDescription)")
            }
            searching = false
        }
    }

    private func refreshBrowse() async {
        lastBrowseRefresh = .now
        guard case .browse = mode else { return }
        hits = (try? await services.reader.browse()) ?? []
    }

    nonisolated private static func describe(_ p: ParsedQuery) -> String? {
        guard p.hasFilters else { return nil }
        var parts: [String] = []
        if !p.text.isEmpty { parts.append("“\(p.text)”") }
        if let a = p.after, let b = p.before {
            let end = b.addingTimeInterval(-1)
            parts.append(Calendar.current.isDate(a, inSameDayAs: end) ? a.formatted(date: .abbreviated, time: .omitted)
                         : "\(a.formatted(date: .abbreviated, time: .omitted)) – \(end.formatted(date: .abbreviated, time: .omitted))")
        }
        if let place = p.place { parts.append(place) }
        return parts.joined(separator: " · ")
    }

    // MARK: - Photo actions

    func explain(_ id: Int64) async -> Explanation? {
        let context: MatchContext
        switch mode {
        case .text(let q): context = .text(q)
        case .similar(let ref), .samePlace(let ref): context = .similar(ref)
        case .photo: context = .photo(tags: queryTags)
        case .browse: context = .browse
        }
        return try? await services.engine.explain(id, context: context)
    }

    func hit(_ id: Int64?) -> SearchHit? { id.flatMap { id in hits.first { $0.id == id } } }

    func open(_ hit: SearchHit) {
        if let url = PhotoRef(localID: hit.localID)?.fileURL { NSWorkspace.shared.open(url) }
        else { NSWorkspace.shared.open(URL(fileURLWithPath: "/System/Applications/Photos.app")) }
    }

    func revealInFinder(_ hit: SearchHit) {
        if let url = PhotoRef(localID: hit.localID)?.fileURL { NSWorkspace.shared.activateFileViewerSelecting([url]) }
    }

    /// Folders dropped on the window are added as sources; an image file searches by that photo.
    func handleDrop(_ urls: [URL]) {
        var dirs: [URL] = []
        for url in urls {
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue { dirs.append(url) }
            else if FolderScanner.extensions.contains(url.pathExtension.lowercased()) { searchByImage(url); return }
        }
        if !dirs.isEmpty { addFolders(dirs) }
    }
}
