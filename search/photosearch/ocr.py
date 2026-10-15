"""On-device OCR with Apple's Vision framework (runs on the Neural Engine; no model download)."""
from __future__ import annotations

import Quartz
import Vision
from Foundation import NSURL


def ocr(path: str, fast: bool = True, min_conf: float = 0.4) -> str:
    url = NSURL.fileURLWithPath_(path)
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    if src is None:
        return ""
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    if img is None:
        return ""
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast if fast else Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(not fast)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    ok, _ = handler.performRequests_error_([req], None)
    if not ok:
        return ""
    lines = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)
        if cand and cand[0].confidence() >= min_conf:
            lines.append(str(cand[0].string()))
    return "\n".join(lines)
