import AppKit
import SwiftUI

/// Decoded thumbnails, memory-only (never written to disk), evicted under pressure by NSCache.
@MainActor
final class ThumbnailCache {
    static let shared = ThumbnailCache()
    private let cache: NSCache<NSString, NSImage> = {
        let c = NSCache<NSString, NSImage>()
        c.totalCostLimit = 256 << 20
        return c
    }()

    func image(_ localID: String, maxSide: Int) async -> NSImage? {
        let key = "\(maxSide)|\(localID)" as NSString
        if let hit = cache.object(forKey: key) { return hit }
        guard let ref = PhotoRef(localID: localID) else { return nil }
        let cg = await Task.detached(priority: .userInitiated) { await ref.cgImage(maxSide: maxSide) }.value
        guard let cg else { return nil }
        let image = NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
        cache.setObject(image, forKey: key, cost: cg.bytesPerRow * cg.height)
        return image
    }
}

struct Thumbnail: View {
    let localID: String
    var maxSide = 360
    var contentMode: ContentMode = .fill
    @State private var image: NSImage?
    @State private var failed = false

    var body: some View {
        ZStack {
            if let image {
                Image(nsImage: image).resizable().aspectRatio(contentMode: contentMode)
            } else {
                Rectangle().fill(.quaternary)
                    .overlay { if failed { Image(systemName: "photo.badge.exclamationmark").foregroundStyle(.secondary) } }
            }
        }
        .task(id: localID) {
            image = await ThumbnailCache.shared.image(localID, maxSide: maxSide)
            failed = image == nil
        }
    }
}
