"""HSEmotion: kare başına 8 duygu olasılığı + valence + arousal (PyTorch'suz, onnxruntime).

Model: enet_b0_8_va_mtl (AffectNet, çok görevli). Çıktı vektörü = 8 duygu logit'i + valence + arousal.
Orijinal .pt dosyası tek seferlik olarak ONNX'e çevrildi (models/enet_b0_8_va_mtl.onnx); PyTorch
sürümüyle fark < 2e-6 (bkz. PROGRESS.md "Hafif sürüm"). Böylece çalışma anında torch gerekmiyor.
Ön işleme hsemotion 0.3.0 ile aynı: 224x224, ImageNet normalizasyonu.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from agent import config
from agent.tools.common import list_frames, load_rgb, second_from_path

EMOTIONS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class _FaceCropper:
    """MediaPipe BlazeFace ile en büyük yüzü bulup kırpar."""

    def __init__(self):
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import face_detector

        opts = face_detector.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_buffer=config.FACE_DETECTOR_MODEL.read_bytes()),
            min_detection_confidence=config.FACE_DETECTION_MIN_CONF,
        )
        self._det = face_detector.FaceDetector.create_from_options(opts)

    def crop(self, rgb: np.ndarray):
        import mediapipe as mp

        res = self._det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.detections:
            return None, None, None
        d = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        b = d.bounding_box
        h, w = rgb.shape[:2]
        mx, my = int(b.width * config.FACE_CROP_MARGIN), int(b.height * config.FACE_CROP_MARGIN)
        x0, y0 = max(0, b.origin_x - mx), max(0, b.origin_y - my)
        x1, y1 = min(w, b.origin_x + b.width + mx), min(h, b.origin_y + b.height + my)
        score = float(d.categories[0].score) if d.categories else None
        return rgb[y0:y1, x0:x1], [x0, y0, x1, y1], score

    def close(self):
        self._det.close()


class HSEmotionTool:
    def __init__(self, device: str | None = None):
        import onnxruntime as ort

        self.device = "cpu"
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._sess = ort.InferenceSession(str(config.HSEMOTION_ONNX), sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        self._size = 224 if "_b0_" in config.HSEMOTION_MODEL_NAME else 260
        self._cropper = _FaceCropper()

    def _prep(self, face: np.ndarray) -> np.ndarray:
        # torchvision Resize (bilinear, antialias) + ToTensor + Normalize ile aynı adımlar
        img = Image.fromarray(face).resize((self._size, self._size), Image.BILINEAR)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
        return x.transpose(2, 0, 1)

    def _predict(self, faces: list[np.ndarray]) -> np.ndarray:
        x = np.stack([self._prep(f) for f in faces]).astype(np.float32)
        return self._sess.run(None, {self._input: x})[0]  # (N, 10): 8 logit + valence + arousal

    def analyze_frame(self, path: Path) -> dict:
        t0 = time.perf_counter()
        rgb = load_rgb(path)
        face, box, det_score = self._cropper.crop(rgb)
        t_det = time.perf_counter()
        out = {"second": second_from_path(path), "frame": path.name, "face_found": face is not None}
        if face is None:
            out.update(dominant_emotion=None, emotions=None, valence=None, arousal=None,
                       note="Yüz bulunamadı")
        else:
            scores = self._predict([face])[0]
            logits = scores[:8]
            p = np.exp(logits - logits.max())
            p /= p.sum()
            out.update(
                face_box=box,
                face_detection_score=round(det_score, 4) if det_score is not None else None,
                dominant_emotion=EMOTIONS[int(p.argmax())],
                emotions={e: round(float(v), 4) for e, v in zip(EMOTIONS, p)},
                valence=round(float(scores[8]), 4),
                arousal=round(float(scores[9]), 4),
            )
        t1 = time.perf_counter()
        out["timing_ms"] = {"detect": round((t_det - t0) * 1000, 1),
                            "total": round((t1 - t0) * 1000, 1)}
        return out

    def analyze(self, frames) -> dict:
        return {
            "tool": "hsemotion",
            "model": config.HSEMOTION_MODEL_NAME,
            "device": self.device,
            "frames": [self.analyze_frame(p) for p in list_frames(frames)],
        }

    def close(self):
        self._cropper.close()


def analyze_frames(frames, device: str | None = None) -> dict:
    tool = HSEmotionTool(device)
    try:
        return tool.analyze(frames)
    finally:
        tool.close()
