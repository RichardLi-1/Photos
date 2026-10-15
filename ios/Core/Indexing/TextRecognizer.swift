import CoreGraphics
import Vision

/// On-device OCR with Vision (VNRecognizeTextRequest), matching the prototype's ocr.py: fast level, min confidence 0.4.
enum TextRecognizer {
    static func recognize(_ image: CGImage, minConfidence: Float = 0.4) -> String? {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .fast
        request.usesLanguageCorrection = false
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        guard (try? handler.perform([request])) != nil else { return nil }
        let lines = (request.results ?? []).compactMap { $0.topCandidates(1).first }
            .filter { $0.confidence >= minConfidence }.map(\.string)
        return lines.isEmpty ? nil : lines.joined(separator: "\n")
    }
}
