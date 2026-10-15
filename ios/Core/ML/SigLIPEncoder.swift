import CoreGraphics
import CoreML
import CoreVideo
import Foundation

/// SigLIP2-B/16-384 image + text towers, run on-device with Core ML (models built by ml/convert_siglip2.py).
/// Both return L2-normalized 768-d vectors in one shared space; nothing leaves the device.
final class SigLIPEncoder: @unchecked Sendable {
    static let dim = 768
    static let imageSize = 384

    struct Calibration: Decodable, Sendable {
        let logit_scale: Float
        let logit_bias: Float
    }

    enum Failure: Error { case missingResource(String), badOutput }

    let tokenizer: SigLIPTokenizer
    let calibration: Calibration
    private let imageModel: MLModel
    private let textModel: MLModel

    init(bundle: Bundle = .main) throws {
        func url(_ name: String, _ ext: String) throws -> URL {
            guard let u = bundle.url(forResource: name, withExtension: ext) else { throw Failure.missingResource("\(name).\(ext)") }
            return u
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Neural Engine when available
        imageModel = try MLModel(contentsOf: url("SigLIP2ImageEncoder", "mlmodelc"), configuration: config)
        textModel = try MLModel(contentsOf: url("SigLIP2TextEncoder", "mlmodelc"), configuration: config)
        tokenizer = try SigLIPTokenizer(bundle: bundle)
        calibration = try JSONDecoder().decode(Calibration.self, from: Data(contentsOf: url("siglip2_meta", "json")))
    }

    /// Embeds a batch of images (each squash-resized to 384×384 like open_clip's SigLIP preprocessing).
    func embed(images: [CGImage]) throws -> [[Float]] {
        let inputs = try images.map { image -> MLFeatureProvider in
            try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: Self.pixelBuffer(image))])
        }
        let out = try imageModel.predictions(fromBatch: MLArrayBatchProvider(array: inputs))
        return try (0..<out.count).map { try Self.vector(out.features(at: $0)) }
    }

    func embed(text: String) throws -> [Float] {
        let ids = tokenizer.encode(text)
        let array = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
        array.withUnsafeMutableBufferPointer(ofType: Int32.self) { buf, _ in
            for (i, id) in ids.enumerated() { buf[i] = id }
        }
        let out = try textModel.prediction(from: MLDictionaryFeatureProvider(dictionary: ["input_ids": array]))
        return try Self.vector(out)
    }

    private static func vector(_ features: MLFeatureProvider) throws -> [Float] {
        guard let a = features.featureValue(for: "embedding")?.multiArrayValue, a.count == dim else { throw Failure.badOutput }
        var v = [Float](repeating: 0, count: dim)
        switch a.dataType {
        case .float32:
            a.withUnsafeBufferPointer(ofType: Float.self) { buf in for i in 0..<dim { v[i] = buf[i] } }
        default:
            for i in 0..<dim { v[i] = a[i].floatValue }
        }
        return SearchMath.normalized(v)
    }

    /// 384×384 BGRA buffer, image stretched to fill (SigLIP "squash", no crop), high-quality interpolation.
    static func pixelBuffer(_ image: CGImage, size: Int = imageSize) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true, kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, attrs, &buffer) == kCVReturnSuccess, let buffer else {
            throw Failure.badOutput
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: size, height: size, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { throw Failure.badOutput }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        return buffer
    }
}
