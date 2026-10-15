import Foundation
import Photos

/// Capabilities for a future on-device agent; the agent loop itself lives elsewhere. Tool names and arguments mirror
/// the prototype's search/photosearch/tools.py (`search_photos`, `create_album`, `add_to_album`, `trash_photos`).
///
/// Photo ids are PHAsset local identifiers. Safety rails: per-call caps; `trash_photos` previews by default
/// (`dry_run` true) and a real run goes through PhotoKit, where iOS itself asks the user to confirm and moves the photos
/// to Recently Deleted (restorable for 30 days).
struct PhotoTools: Sendable {
    static let maxLimit = 200, maxTrash = 200, maxAlbumBatch = 1000

    struct ToolError: LocalizedError { let errorDescription: String? ; init(_ m: String) { errorDescription = m } }

    struct Photo: Codable, Sendable, Equatable {
        let id: String
        let date: Date?
        let place: String?
        let score: Float?
    }

    struct TrashResult: Codable, Sendable {
        var dryRun: Bool
        var photoIDs: [String]
        var notFound: [String]
        var trashed: Int
        var reason: String
        var note = "iOS asks the user to confirm; photos go to Recently Deleted for 30 days"
    }

    var services: AppServices = .shared
    private var library: PhotoLibrary { .shared }

    // MARK: - Tools

    func searchPhotos(query: String? = nil, similarTo: String? = nil, after: Date? = nil, before: Date? = nil,
                      place: String? = nil, hasText: Bool = false, limit: Int = 24) async throws -> [Photo] {
        let text = query?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        guard text != nil || similarTo != nil else { throw ToolError("give query (text) and/or similar_to (a photo id)") }
        _ = try await services.encoder()
        var request = SearchRequest(text: text, limit: max(1, min(Self.maxLimit, limit)))
        request.filters = SearchFilters(after: after, before: before, place: place, hasText: hasText)
        if let similarTo {
            guard let id = try await services.store.id(forLocalID: similarTo) else { throw ToolError("photo \(similarTo) not found / not indexed") }
            request.similarTo = id
        }
        let hits = try await services.engine.search(request).hits
        let records = Dictionary(uniqueKeysWithValues: try await services.store.records(hits.map(\.id)).map { ($0.id, $0) })
        return hits.map { Photo(id: $0.localID, date: records[$0.id]?.taken, place: records[$0.id]?.place, score: $0.score) }
    }

    /// Creates a user album (fails if one with that name exists) and optionally fills it.
    func createAlbum(name: String, photoIDs: [String] = []) async throws -> (albumID: String, added: Int) {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, name.count <= 100 else { throw ToolError("album name must be 1–100 characters") }
        try requireAccess()
        let options = PHFetchOptions()
        options.predicate = NSPredicate(format: "localizedTitle ==[c] %@", name)
        guard PHAssetCollection.fetchAssetCollections(with: .album, subtype: .albumRegular, options: options).count == 0 else {
            throw ToolError("an album named \(name) already exists")
        }
        var placeholder: PHObjectPlaceholder?
        try await PHPhotoLibrary.shared().performChanges {
            placeholder = PHAssetCollectionChangeRequest.creationRequestForAssetCollection(withTitle: name).placeholderForCreatedAssetCollection
        }
        guard let albumID = placeholder?.localIdentifier else { throw ToolError("could not create the album") }
        let added = photoIDs.isEmpty ? 0 : try await addToAlbum(albumID: albumID, photoIDs: photoIDs)
        return (albumID, added)
    }

    /// Adds photos to an album; returns how many were added (already-present photos are skipped).
    func addToAlbum(albumID: String, photoIDs: [String]) async throws -> Int {
        let ids = try validated(photoIDs, cap: Self.maxAlbumBatch)
        try requireAccess()
        guard let album = PHAssetCollection.fetchAssetCollections(withLocalIdentifiers: [albumID], options: nil).firstObject,
              album.canPerform(.addContent) else { throw ToolError("album \(albumID) not found or not editable") }
        var present = Set<String>()
        PHAsset.fetchAssets(in: album, options: nil).enumerateObjects { a, _, _ in present.insert(a.localIdentifier) }
        let assets = library.assets(ids).filter { !present.contains($0.localIdentifier) }
        guard !assets.isEmpty else { return 0 }
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetCollectionChangeRequest(for: album)?.addAssets(assets as NSArray)
        }
        return assets.count
    }

    /// Dry run (default): what would be trashed. Real run: PhotoKit deletion, which iOS confirms with the user.
    func trashPhotos(photoIDs: [String], reason: String, dryRun: Bool = true) async throws -> TrashResult {
        let ids = try validated(photoIDs, cap: Self.maxTrash)
        let reason = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !reason.isEmpty else { throw ToolError("reason required") }
        let assets = library.assets(ids)
        let found = Set(assets.map(\.localIdentifier))
        var result = TrashResult(dryRun: dryRun, photoIDs: assets.map(\.localIdentifier), notFound: ids.filter { !found.contains($0) },
                                 trashed: 0, reason: String(reason.prefix(200)))
        guard !dryRun, !assets.isEmpty else { return result }
        try requireAccess()
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.deleteAssets(assets as NSArray)
        }
        result.trashed = assets.count
        return result
    }

    // MARK: - JSON entry point for an agent loop

    /// `call("trash_photos", {"photo_ids": [...], "reason": "blurry"})` → JSON object; never throws (errors → {"error"}).
    func call(_ name: String, arguments: [String: Any]) async -> [String: Any] {
        func strings(_ k: String) -> [String] { (arguments[k] as? [Any])?.compactMap { $0 as? String } ?? [] }
        func date(_ k: String) -> Date? { (arguments[k] as? String).flatMap(Self.parseDate) }
        do {
            switch name {
            case "search_photos":
                let photos = try await searchPhotos(query: arguments["query"] as? String, similarTo: arguments["similar_to"] as? String,
                                                    after: date("after"), before: date("before"), place: arguments["place"] as? String,
                                                    hasText: arguments["has_text"] as? Bool ?? false, limit: arguments["limit"] as? Int ?? 24)
                return ["results": photos.map { p -> [String: Any] in
                    ["id": p.id, "date": p.date.map { ISO8601DateFormatter().string(from: $0) } as Any, "place": p.place as Any,
                     "score": p.score.map(Double.init) as Any]
                }]
            case "create_album":
                let r = try await createAlbum(name: arguments["name"] as? String ?? "", photoIDs: strings("photo_ids"))
                return ["album": ["id": r.albumID, "name": arguments["name"] as? String ?? ""], "added": r.added]
            case "add_to_album":
                return ["added": try await addToAlbum(albumID: arguments["album_id"] as? String ?? "", photoIDs: strings("photo_ids"))]
            case "trash_photos":
                let r = try await trashPhotos(photoIDs: strings("photo_ids"), reason: arguments["reason"] as? String ?? "",
                                              dryRun: arguments["dry_run"] as? Bool ?? true)
                return [r.dryRun ? "would_trash" : "trashed": r.dryRun ? r.photoIDs.count : r.trashed, "photo_ids": r.photoIDs,
                        "not_found": r.notFound, "reason": r.reason, "dry_run": r.dryRun, "note": r.note]
            default:
                return ["error": "unknown tool \(name)"]
            }
        } catch {
            return ["error": error.localizedDescription]
        }
    }

    /// Tool definitions in the {name, description, input_schema} shape LLM tool-calling APIs accept.
    static let schemas: [[String: Any]] = [
        tool("search_photos", "Search the user's photo library by meaning (text) and/or visual similarity to a photo, with optional date/place filters.",
             ["query": str("What to look for, in natural language"), "similar_to": str("A photo id to find similar photos to"),
              "after": str("ISO date, inclusive"), "before": str("ISO date, exclusive"), "place": str("Substring of the place name"),
              "has_text": ["type": "boolean"], "limit": ["type": "integer", "minimum": 1, "maximum": maxLimit]], required: []),
        tool("create_album", "Create a new album, optionally with photos.",
             ["name": str("Album name"), "photo_ids": ids()], required: ["name"]),
        tool("add_to_album", "Add photos to an existing album.",
             ["album_id": str("Album id from create_album"), "photo_ids": ids()], required: ["album_id", "photo_ids"]),
        tool("trash_photos", "Move photos to Recently Deleted. Defaults to a dry run; iOS asks the user to confirm a real run.",
             ["photo_ids": ids(), "reason": str("Why, shown to the user"), "dry_run": ["type": "boolean", "default": true]],
             required: ["photo_ids", "reason"]),
    ]

    // MARK: - Helpers

    private static func tool(_ name: String, _ description: String, _ props: [String: Any], required: [String]) -> [String: Any] {
        ["name": name, "description": description, "input_schema": ["type": "object", "properties": props, "required": required]]
    }
    private static func str(_ d: String) -> [String: Any] { ["type": "string", "description": d] }
    private static func ids() -> [String: Any] { ["type": "array", "items": ["type": "string"]] }

    static func parseDate(_ s: String) -> Date? {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withFullDate]
        return ISO8601DateFormatter().date(from: s) ?? f.date(from: String(s.prefix(10)))
    }

    private func validated(_ ids: [String], cap: Int) throws -> [String] {
        var seen = Set<String>()
        let out = ids.filter { seen.insert($0).inserted }
        guard !out.isEmpty else { throw ToolError("photo_ids is empty") }
        guard out.count <= cap else { throw ToolError("too many photo_ids: \(out.count) > cap \(cap) per call; split into batches") }
        return out
    }

    private func requireAccess() throws {
        let status = library.status
        guard status == .authorized || status == .limited else { throw ToolError("no photo library access") }
    }
}
