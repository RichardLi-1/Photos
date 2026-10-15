import Photos
import UIKit

/// Thin async wrapper over PhotoKit. Never allows network access: iCloud-only originals fall back to the local
/// thumbnail, and are skipped if there is none.
final class PhotoLibrary: @unchecked Sendable {
    static let shared = PhotoLibrary()
    let imageManager = PHCachingImageManager()

    var status: PHAuthorizationStatus { PHPhotoLibrary.authorizationStatus(for: .readWrite) }

    func requestAccess() async -> PHAuthorizationStatus {
        await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    }

    /// Images the app can see (all, or the user's limited selection), newest first.
    func fetchImages() -> PHFetchResult<PHAsset> {
        let options = PHFetchOptions()
        options.predicate = NSPredicate(format: "mediaType == %d", PHAssetMediaType.image.rawValue)
        options.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        return PHAsset.fetchAssets(with: options)
    }

    func assetInfos(_ fetch: PHFetchResult<PHAsset>) -> [AssetInfo] {
        var out: [AssetInfo] = []
        out.reserveCapacity(fetch.count)
        fetch.enumerateObjects { a, _, _ in
            out.append(AssetInfo(localID: a.localIdentifier, taken: a.creationDate, modified: a.modificationDate,
                                 lat: a.location?.coordinate.latitude, lon: a.location?.coordinate.longitude,
                                 width: a.pixelWidth, height: a.pixelHeight))
        }
        return out
    }

    func assets(_ localIDs: [String]) -> [PHAsset] {
        let fetch = PHAsset.fetchAssets(withLocalIdentifiers: localIDs, options: nil)
        var by: [String: PHAsset] = [:]
        fetch.enumerateObjects { a, _, _ in by[a.localIdentifier] = a }
        return localIDs.compactMap { by[$0] }
    }

    /// A local, display-ready rendition whose longer side is ≤ `maxSide` px (used for both embedding and OCR).
    func cgImage(for asset: PHAsset, maxSide: CGFloat = 1024) async -> CGImage? {
        if let image = await request(asset, maxSide: maxSide, mode: .highQualityFormat) { return image }
        return await request(asset, maxSide: maxSide, mode: .fastFormat)  // cloud-only original: local preview
    }

    private func request(_ asset: PHAsset, maxSide: CGFloat, mode: PHImageRequestOptionsDeliveryMode) async -> CGImage? {
        let options = PHImageRequestOptions()
        options.deliveryMode = mode
        options.resizeMode = .exact
        options.isNetworkAccessAllowed = false  // privacy: never download from iCloud
        options.isSynchronous = false
        options.version = .current
        return await withCheckedContinuation { cont in
            PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: maxSide, height: maxSide),
                                                  contentMode: .aspectFit, options: options) { image, _ in
                cont.resume(returning: image?.cgImage)  // fast/high-quality modes call back exactly once
            }
        }
    }

    /// Full-resolution UIImage for an asset from the user's own picker/camera input (not the library).
    static func cgImage(from image: UIImage) -> CGImage? {
        if let cg = image.cgImage, image.imageOrientation == .up { return cg }
        let renderer = UIGraphicsImageRenderer(size: image.size, format: { let f = UIGraphicsImageRendererFormat(); f.scale = 1; return f }())
        return renderer.image { _ in image.draw(at: .zero) }.cgImage
    }
}
