#if DEBUG
import Foundation
import Photos

/// DEBUG-only end-to-end check, driven by launch arguments (used with `xcrun simctl launch --console-pty`):
///   -selftest "query one" "query two" ...
/// Waits for indexing to finish, runs each query through the same PhotoTools/SearchEngine path as the UI, and prints
/// one JSON line per query with the top hits' original file names (local metadata only), plus a "more like this" and
/// an explanation for the first query's top hit. Compiled out of Release builds.
enum SelfTest {
    static func runIfRequested() async {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "-selftest") else { return }
        let queries = Array(args[(i + 1)...]).filter { !$0.hasPrefix("-") }
        func emit(_ object: [String: Any]) {
            if let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]),
               let line = String(data: data, encoding: .utf8) {
                print("SELFTEST " + line)
                fflush(stdout)
            }
        }
        do {
            let services = AppServices.shared
            let clock = ContinuousClock()
            let t0 = clock.now
            let indexer = try await services.indexer()
            emit(["event": "models_loaded", "seconds": (clock.now - t0).seconds])
            let t1 = clock.now
            await indexer.runToCompletion()
            let p = await indexer.progress
            emit(["event": "indexed", "total": p.total, "embedded": p.embedded, "skipped": p.skipped, "seconds": (clock.now - t1).seconds])

            let tools = PhotoTools()
            for q in queries {
                let parsed = QueryParser.parse(q, places: (try? await services.store.placeNames()) ?? [])
                let t = clock.now
                let hits = try await tools.searchPhotos(query: parsed.text.nilIfEmpty ?? q, after: parsed.after, before: parsed.before,
                                                        place: parsed.place, limit: 3)
                emit(["event": "search", "query": q, "ms": (clock.now - t).seconds * 1000, "filters": parsed.hasFilters,
                      "top": hits.map { name($0.id) }, "scores": hits.map { Double($0.score ?? 0) }])
            }
            if let q = queries.first, let top = try await tools.searchPhotos(query: q, limit: 1).first,
               let id = try await services.store.id(forLocalID: top.id) {
                let e = try await services.engine.explain(id, context: .text(q))
                emit(["event": "explain", "query": q, "photo": name(top.id), "shared": e.shared, "text_hits": e.textHits])
                let similar = try await tools.searchPhotos(similarTo: top.id, limit: 3)
                emit(["event": "similar", "to": name(top.id), "top": similar.map { name($0.id) }])
            }
            let schemaNames = PhotoTools.schemas.compactMap { $0["name"] as? String }
            emit(["event": "tools", "names": schemaNames])
        } catch {
            emit(["event": "error", "message": String(describing: error)])
        }
        emit(["event": "done"])
    }

    private static func name(_ localID: String) -> String {
        guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [localID], options: nil).firstObject else { return localID }
        return PHAssetResource.assetResources(for: asset).first?.originalFilename ?? localID
    }
}

private extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) / 1e18 }
}
#endif
