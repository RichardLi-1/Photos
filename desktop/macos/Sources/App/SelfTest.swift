import Darwin
import Foundation

/// Headless end-to-end check of the shipped app, run from Terminal:
///
///     tar -cf - -C photos a.jpg b.jpg … | PhotoSearch.app/Contents/MacOS/PhotoSearch --self-test --tar - \
///         --query "a dog running in a garden" --query "…"
///     PhotoSearch.app/Contents/MacOS/PhotoSearch --self-test --dir /path/inside/the/container
///
/// The app is sandboxed with no file entitlements beyond user-selected ones, so it can't open arbitrary paths given on
/// the command line. Images therefore arrive as a tar stream on stdin (an inherited file descriptor) and are unpacked
/// into the container's temp directory. Everything (index, models' compile cache) stays inside the sandbox, and the
/// temporary index is deleted afterwards. It also proves the sandbox blocks networking.
enum SelfTest {
    static func run(_ args: [String]) async -> Int32 {
        var queries: [String] = []
        var tarPath: String?
        var dir: String?
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--query" where i + 1 < args.count: queries.append(args[i + 1]); i += 1
            case "--tar" where i + 1 < args.count: tarPath = args[i + 1]; i += 1
            case "--dir" where i + 1 < args.count: dir = args[i + 1]; i += 1
            default: break
            }
            i += 1
        }
        if queries.isEmpty { queries = ["a dog running in a garden", "snowmobiles in a snowy forest", "a couple sitting on the beach"] }

        var ok = true
        func check(_ cond: Bool, _ label: String) { print(cond ? "  PASS  \(label)" : "  FAIL  \(label)"); ok = ok && cond }
        let clock = ContinuousClock()
        func ms(_ d: Duration) -> String { String(format: "%.0f ms", Double(d.components.attoseconds) / 1e15 + Double(d.components.seconds) * 1000) }

        print("Photo Search self-test")
        let env = ProcessInfo.processInfo.environment
        let sandboxed = env["APP_SANDBOX_CONTAINER_ID"] != nil
        print("  macOS \(ProcessInfo.processInfo.operatingSystemVersionString), \(ProcessInfo.processInfo.physicalMemory / 1_073_741_824) GB RAM")
        check(sandboxed, "running inside the App Sandbox (container \(env["APP_SANDBOX_CONTAINER_ID"] ?? "none"))")
        let net = networkProbe()
        check(net.blocked, "network is blocked: \(net.detail)")

        let work = FileManager.default.temporaryDirectory.appendingPathComponent("selftest-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: work) }
        let imagesDir: URL
        do {
            if let tarPath {
                imagesDir = work.appendingPathComponent("images", isDirectory: true)
                let input = tarPath == "-" ? FileHandle.standardInput : try FileHandle(forReadingFrom: URL(fileURLWithPath: tarPath))
                let n = try TarReader.extract(from: input, to: imagesDir)
                print("  unpacked \(n) images from the tar stream")
            } else if let dir {
                imagesDir = URL(fileURLWithPath: dir, isDirectory: true)
            } else {
                print("  FAIL  no input: pass --tar - (stdin) or --dir PATH")
                return 2
            }
        } catch {
            print("  FAIL  reading input: \(error)")
            return 2
        }

        do {
            let services = try DesktopServices(directory: work.appendingPathComponent("index", isDirectory: true))
            var t = clock.now
            let encoder = try await services.encoder()
            print("  models loaded in \(ms(clock.now - t)) (Core ML, compute units: all)")

            if let place = await services.placeIndex() {
                _ = place
                let head = try PlaceHead()
                let parity = head.parity()
                check(parity > 0.9999, String(format: "same-place head matches PyTorch (min cos %.6f)", parity))
            } else {
                check(false, "same-place head bundled")
            }

            let indexer = try await services.indexer()
            t = clock.now
            await indexer.runToCompletion(IndexSources(folders: [imagesDir]))
            let took = clock.now - t
            let status = await indexer.status
            let secs = Double(took.components.seconds) + Double(took.components.attoseconds) / 1e18
            print(String(format: "  indexed %d photos (%d skipped) in %@ — %.1f photos/s (decode + Vision OCR + SigLIP2 embedding)",
                         status.embedded, status.skipped, ms(took), Double(status.embedded) / max(secs, 0.001)))
            check(status.embedded > 0 && status.skipped == 0, "every photo indexed")

            var firstTop: SearchHit?
            for q in queries {
                let parsed = QueryParser.parse(q)
                var req = SearchRequest(text: parsed.text.isEmpty ? nil : parsed.text, limit: 5)
                req.filters = SearchFilters(after: parsed.after, before: parsed.before)
                t = clock.now
                let res = try await services.engine.search(req)
                let cold = clock.now - t
                t = clock.now
                _ = try await services.engine.search(req)
                let warm = clock.now - t
                print("  query “\(q)”: \(ms(cold)) first (text encoder), \(ms(warm)) repeat")
                for (rank, h) in res.hits.prefix(3).enumerated() {
                    print(String(format: "      %d. %@  %.3f", rank + 1, PhotoRef(localID: h.localID)?.displayName ?? h.localID, h.score ?? 0))
                }
                if firstTop == nil { firstTop = res.hits.first }
            }
            check(firstTop != nil, "text search returns results")

            if let top = firstTop {
                let e = try await services.engine.explain(top.id, context: .text(queries[0]))
                print("  why it matched: \(e.shared.joined(separator: " · "))\(e.overall ? " (overall look)" : "")")
                t = clock.now
                var req = SearchRequest(limit: 3)
                req.similarTo = top.id
                let similar = try await services.engine.search(req)
                print("  More like this (\(ms(clock.now - t))): " + similar.hits.map { PhotoRef(localID: $0.localID)?.displayName ?? "?" }.joined(separator: ", "))
                if let place = await services.placeIndex() {
                    t = clock.now
                    let same = try await place.samePlace(as: top.id, limit: 3)
                    print("  Same place (\(ms(clock.now - t)), incl. projecting the library through the head): "
                          + same.map { PhotoRef(localID: $0.localID)?.displayName ?? "?" }.joined(separator: ", "))
                    check(!same.isEmpty, "same place returns results")
                }
            }
            _ = encoder
        } catch {
            check(false, "engine error: \(error)")
        }
        print(ok ? "SELF-TEST PASSED" : "SELF-TEST FAILED")
        return ok ? 0 : 1
    }

    /// Tries a TCP connect to 127.0.0.1:9 (discard port, on this Mac, so nothing could leave it anyway). Inside the
    /// sandbox without `com.apple.security.network.client` the kernel refuses with EPERM before any packet is built.
    static func networkProbe() -> (blocked: Bool, detail: String) {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return (true, "socket() refused: \(String(cString: strerror(errno)))") }
        defer { close(fd) }
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(9).bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let rc = withUnsafePointer(to: &addr) { $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) } }
        let err = errno
        if rc == 0 { return (false, "connect() succeeded") }
        return (err == EPERM || err == EACCES, "connect() → \(String(cString: strerror(err))) (errno \(err))")
    }
}

/// Minimal ustar/pax reader: regular files only, flattened into one directory, image extensions only.
enum TarReader {
    enum Failure: Error { case truncated }

    static func extract(from handle: FileHandle, to dir: URL) throws -> Int {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var count = 0
        var pendingName: String?
        while true {
            let header = try read(handle, 512)
            if header.allSatisfy({ $0 == 0 }) { break }
            func field(_ off: Int, _ len: Int) -> String {
                let bytes = header[off..<off + len].prefix { $0 != 0 }
                return String(decoding: bytes, as: UTF8.self)
            }
            let size = Int(field(124, 12).trimmingCharacters(in: .whitespaces), radix: 8) ?? 0
            let type = header[156]
            var name = field(0, 100)
            let prefix = field(345, 155)
            if !prefix.isEmpty { name = prefix + "/" + name }
            let body = try read(handle, (size + 511) / 512 * 512).prefix(size)
            switch type {
            case UInt8(ascii: "x"):  // pax header: may carry a long path
                for line in String(decoding: body, as: UTF8.self).split(separator: "\n") {
                    if let r = line.range(of: " path=") { pendingName = String(line[r.upperBound...]) }
                }
            case UInt8(ascii: "0"), 0:
                let base = ((pendingName ?? name) as NSString).lastPathComponent
                pendingName = nil
                guard !base.hasPrefix("."), FolderScanner.extensions.contains((base as NSString).pathExtension.lowercased()) else { continue }
                try Data(body).write(to: dir.appendingPathComponent(base))
                count += 1
            default:
                pendingName = nil
            }
        }
        return count
    }

    private static func read(_ h: FileHandle, _ n: Int) throws -> Data {
        var out = Data()
        while out.count < n {
            guard let chunk = try h.read(upToCount: n - out.count), !chunk.isEmpty else { throw Failure.truncated }
            out += chunk
        }
        return out
    }
}
