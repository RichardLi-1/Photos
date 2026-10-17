import Photos
import SwiftUI

/// Edge-to-edge square photo grid (the web prototype's "wall"). Thumbnails come from PHCachingImageManager; tapping a
/// photo opens its match card.
struct PhotoGrid: View {
    @Bindable var model: AppModel
    var onOpen: (String) -> Void

    @Environment(\.dynamicTypeSize) private var typeSize

    private var columns: [GridItem] {
        // larger text → fewer, bigger tiles (still ≥ 44pt hit targets at any size)
        [GridItem(.adaptive(minimum: typeSize.isAccessibilitySize ? 160 : 110), spacing: 2)]
    }

    var body: some View {
        ScrollView {
            LazyVGrid(columns: columns, spacing: 2) {
                ForEach(0..<model.items.count, id: \.self) { i in
                    let asset = model.items[i]
                    PhotoCell(asset: asset, isPicked: model.picked == asset.localIdentifier) {
                        model.picked = asset.localIdentifier
                    }
                    .popover(isPresented: Binding(get: { model.picked == asset.localIdentifier },
                                                  set: { if !$0, model.picked == asset.localIdentifier { model.picked = nil } })) {
                        MatchCard(model: model, localID: asset.localIdentifier, onOpen: onOpen)
                            .presentationCompactAdaptation(.popover)
                    }
                }
            }
            .id(model.items.version)
        }
        .scrollDismissesKeyboard(.immediately)
    }
}

struct PhotoCell: View {
    let asset: PHAsset
    var isPicked: Bool
    var onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            Color(.secondarySystemBackground)
                .aspectRatio(1, contentMode: .fit)
                .overlay { Thumbnail(asset: asset) }
                .clipped()
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Self.label(for: asset))
        .accessibilityHint("Shows why this photo matched")
        .accessibilityAddTraits(.isImage)
    }

    static func label(for asset: PHAsset) -> Text {
        if let date = asset.creationDate {
            return Text("Photo, \(date.formatted(date: .long, time: .shortened))")
        }
        return Text("Photo")
    }
}

/// Async thumbnail (opportunistic: a fast low-res image first, then the sharp one). Never downloads from iCloud.
struct Thumbnail: View {
    let asset: PHAsset
    var contentMode: ContentMode = .fill
    var targetSide: CGFloat = 220

    @State private var image: UIImage?
    @Environment(\.displayScale) private var scale
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            if let image {
                Image(uiImage: image).resizable().aspectRatio(contentMode: contentMode)
                    .transition(.opacity)
            }
        }
        .animation(reduceMotion ? nil : .easeOut(duration: 0.25), value: image != nil)
        .task(id: asset.localIdentifier) {
            for await next in Self.images(for: asset, side: targetSide * scale) { image = next }
        }
    }

    static func images(for asset: PHAsset, side: CGFloat) -> AsyncStream<UIImage> {
        AsyncStream { cont in
            let options = PHImageRequestOptions()
            options.deliveryMode = .opportunistic
            options.resizeMode = .fast
            options.isNetworkAccessAllowed = false
            let manager = PhotoLibrary.shared.imageManager
            let id = manager.requestImage(for: asset, targetSize: CGSize(width: side, height: side), contentMode: .aspectFill,
                                          options: options) { image, info in
                if let image { cont.yield(image) }
                if (info?[PHImageResultIsDegradedKey] as? Bool) != true { cont.finish() }
            }
            cont.onTermination = { _ in manager.cancelImageRequest(id) }
        }
    }
}
