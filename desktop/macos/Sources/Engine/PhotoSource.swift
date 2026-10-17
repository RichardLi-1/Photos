import AppKit
import Foundation
import ImageIO
import Photos

/// Where an indexed photo lives. The shared `LibraryStore` keys photos by a string `localID`; on the Mac that is either a
/// file URL ("file:///Users/…/IMG_1.jpg", a photo in a folder the user picked) or "photos:<PHAsset id>" (Photos library).
enum PhotoRef: Hashable, Sendable {
    case file(URL)
    case asset(String)

    static let assetPrefix = "photos:"

    init?(localID: String) {
        if localID.hasPrefix(Self.assetPrefix) {
            self = .asset(String(localID.dropFirst(Self.assetPrefix.count)))
        } else if let url = URL(string: localID), url.isFileURL {
            self = .file(url)
        } else {
            return nil
        }
    }

    var localID: String {
        switch self {
        case .file(let url): url.absoluteString
        case .asset(let id): Self.assetPrefix + id
        }
    }

    var fileURL: URL? { if case .file(let u) = self { u } else { nil } }

    var displayName: String {
        switch self {
        case .file(let url): url.lastPathComponent
        case .asset: String(localized: "Photos library")
        }
    }

    /// A local rendition whose longer side is ≤ `maxSide` px, upright (EXIF orientation applied). Never uses the network.
    func cgImage(maxSide: Int) async -> CGImage? {
        switch self {
        case .file(let url): FolderScanner.cgImage(url, maxSide: maxSide)
        case .asset(let id): await PhotosLibrarySource.cgImage(localIdentifier: id, maxSide: maxSide)
        }
    }
}

/// Lists and reads image files under user-chosen folders with ImageIO (headers only for metadata; downsampled decode).
enum FolderScanner {
    static let extensions: Set<String> = ["jpg", "jpeg", "png", "heic", "heif", "tif", "tiff", "gif", "webp", "bmp", "avif"]

    struct Entry: Sendable { let url: URL; let modified: Date? }

    /// Every image file below `root` (hidden files and packages such as .photoslibrary are skipped).
    static func list(_ root: URL) -> [Entry] {
        let keys: [URLResourceKey] = [.isRegularFileKey, .contentModificationDateKey]
        guard let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: keys,
                                                          options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { return [] }
        var out: [Entry] = []
        for case let url as URL in walker where extensions.contains(url.pathExtension.lowercased()) {
            let values = try? url.resourceValues(forKeys: Set(keys))
            guard values?.isRegularFile == true else { continue }
            out.append(Entry(url: url.standardizedFileURL, modified: values?.contentModificationDate))
        }
        return out
    }

    /// Metadata from the image header: EXIF capture time (local time, like search/photosearch/library.py), GPS, size.
    /// No EXIF date → `taken` stays nil (file times say when a copy was made, not when the photo was taken).
    static func info(_ entry: Entry) -> AssetInfo {
        var taken: Date?, lat: Double?, lon: Double?, width = 0, height = 0
        if let src = CGImageSourceCreateWithURL(entry.url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
           let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any] {
            width = props[kCGImagePropertyPixelWidth] as? Int ?? 0
            height = props[kCGImagePropertyPixelHeight] as? Int ?? 0
            let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any]
            let tiff = props[kCGImagePropertyTIFFDictionary] as? [CFString: Any]
            let raw = (exif?[kCGImagePropertyExifDateTimeOriginal] ?? exif?[kCGImagePropertyExifDateTimeDigitized]
                       ?? tiff?[kCGImagePropertyTIFFDateTime]) as? String
            taken = raw.flatMap(exifDate)
            if let gps = props[kCGImagePropertyGPSDictionary] as? [CFString: Any],
               let la = gps[kCGImagePropertyGPSLatitude] as? Double, let lo = gps[kCGImagePropertyGPSLongitude] as? Double {
                lat = (gps[kCGImagePropertyGPSLatitudeRef] as? String) == "S" ? -la : la
                lon = (gps[kCGImagePropertyGPSLongitudeRef] as? String) == "W" ? -lo : lo
            }
        }
        return AssetInfo(localID: PhotoRef.file(entry.url).localID, taken: taken, modified: entry.modified,
                         lat: lat, lon: lon, width: width, height: height)
    }

    private static let exifFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyy:MM:dd HH:mm:ss"
        return f
    }()

    static func exifDate(_ s: String) -> Date? {
        guard s.count >= 19 else { return nil }
        return exifFormatter.date(from: String(s.prefix(19)))
    }

    /// Downsampled, upright decode (ImageIO picks a cheap path, e.g. JPEG subsampling, instead of decoding full size).
    static func cgImage(_ url: URL, maxSide: Int) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxSide,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, options as CFDictionary)
    }
}

/// The macOS Photos library through PhotoKit (optional; the user turns it on). Like the iOS app it never allows network
/// access: iCloud-only originals fall back to the local preview, or are skipped.
enum PhotosLibrarySource {
    static var status: PHAuthorizationStatus { PHPhotoLibrary.authorizationStatus(for: .readWrite) }
    static var isAuthorized: Bool { status == .authorized || status == .limited }

    static func requestAccess() async -> PHAuthorizationStatus {
        await PHPhotoLibrary.requestAuthorization(for: .readWrite)
    }

    static func list() -> [AssetInfo] {
        guard isAuthorized else { return [] }
        let options = PHFetchOptions()
        options.predicate = NSPredicate(format: "mediaType == %d", PHAssetMediaType.image.rawValue)
        var out: [AssetInfo] = []
        PHAsset.fetchAssets(with: options).enumerateObjects { a, _, _ in
            out.append(AssetInfo(localID: PhotoRef.asset(a.localIdentifier).localID, taken: a.creationDate, modified: a.modificationDate,
                                 lat: a.location?.coordinate.latitude, lon: a.location?.coordinate.longitude,
                                 width: a.pixelWidth, height: a.pixelHeight))
        }
        return out
    }

    static func cgImage(localIdentifier: String, maxSide: Int) async -> CGImage? {
        guard let asset = PHAsset.fetchAssets(withLocalIdentifiers: [localIdentifier], options: nil).firstObject else { return nil }
        if let image = await request(asset, maxSide: maxSide, mode: .highQualityFormat) { return image }
        return await request(asset, maxSide: maxSide, mode: .fastFormat)
    }

    private static func request(_ asset: PHAsset, maxSide: Int, mode: PHImageRequestOptionsDeliveryMode) async -> CGImage? {
        let options = PHImageRequestOptions()
        options.deliveryMode = mode
        options.resizeMode = .exact
        options.isNetworkAccessAllowed = false  // privacy: never download from iCloud
        options.isSynchronous = false
        options.version = .current
        return await withCheckedContinuation { cont in
            PHImageManager.default().requestImage(for: asset, targetSize: CGSize(width: maxSide, height: maxSide),
                                                  contentMode: .aspectFit, options: options) { image, _ in
                cont.resume(returning: image?.cgImage(forProposedRect: nil, context: nil, hints: nil))
            }
        }
    }
}
