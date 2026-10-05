"""MediaPipe Face Landmarker: baş pozu (yaw/pitch/roll), göz/iris konumu, göz kırpma,
"ekrana bakıyor mu" (0-1). Yüz bulunamazsa face_found=False ve açık bir not döner.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import face_landmarker

from agent import config
from agent.tools.common import list_frames, load_rgb, second_from_path

# 478 noktalı mesh indeksleri (kişinin kendi sağ/sol gözü)
_R_EYE = {"outer": 33, "inner": 133, "top": 159, "bottom": 145, "iris": 468}
_L_EYE = {"outer": 263, "inner": 362, "top": 386, "bottom": 374, "iris": 473}
_BLENDSHAPES_KEEP = [
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
]


# MediaPipe blendshape -> FACS Action Unit yaklaşık eşlemesi (py-feat yerine; PyTorch gerektirmez).
# Birden fazla blendshape varsa ortalaması (sol/sağ) alınır. Değerler 0-1; "AU ölçümü" değil, vekil.
AU_FROM_BLENDSHAPES = {
    "AU1": ["browInnerUp"],
    "AU2": ["browOuterUpLeft", "browOuterUpRight"],
    "AU4": ["browDownLeft", "browDownRight"],
    "AU6": ["cheekSquintLeft", "cheekSquintRight"],
    "AU7": ["eyeSquintLeft", "eyeSquintRight"],
    "AU9": ["noseSneerLeft", "noseSneerRight"],
    "AU12": ["mouthSmileLeft", "mouthSmileRight"],
    "AU14": ["mouthDimpleLeft", "mouthDimpleRight"],
    "AU15": ["mouthFrownLeft", "mouthFrownRight"],
    "AU17": ["mouthShrugLower"],
    "AU20": ["mouthStretchLeft", "mouthStretchRight"],
    "AU23": ["mouthPressLeft", "mouthPressRight"],
    "AU25": ["jawOpen"],
    "AU26": ["jawOpen"],
    "AU28": ["mouthRollLower"],
    "AU43": ["eyeBlinkLeft", "eyeBlinkRight"],
    "AU45": ["eyeBlinkLeft", "eyeBlinkRight"],
}


def aus_from_blendshapes(bs: dict) -> dict:
    out = {}
    for au, names in AU_FROM_BLENDSHAPES.items():
        vals = [bs[n] for n in names if n in bs]
        if vals:
            out[au] = round(float(sum(vals) / len(vals)), 4)
    return out


def _head_pose_deg(matrix: np.ndarray) -> dict:
    """Facial transformation matrix (4x4) -> Euler açıları (derece)."""
    angles = cv2.RQDecomp3x3(np.asarray(matrix)[:3, :3])[0]
    pitch, yaw, roll = (float(a) for a in angles)
    return {"pitch": round(pitch, 2), "yaw": round(yaw, 2), "roll": round(roll, 2)}


def _iris_ratio(lm, eye: dict) -> dict:
    """İrisin göz içindeki konumu: 0.5 = merkez. x: outer->inner, y: top->bottom."""
    def ratio(a, b, v):
        d = b - a
        return (v - a) / d if abs(d) > 1e-6 else 0.5
    iris = lm[eye["iris"]]
    return {
        "x": round(ratio(lm[eye["outer"]].x, lm[eye["inner"]].x, iris.x), 3),
        "y": round(ratio(lm[eye["top"]].y, lm[eye["bottom"]].y, iris.y), 3),
    }


class MediaPipeTool:
    def __init__(self):
        opts = face_landmarker.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=config.FACE_LANDMARKER_MODEL.read_bytes()),
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._lm = face_landmarker.FaceLandmarker.create_from_options(opts)

    def analyze_frame(self, path: Path) -> dict:
        t0 = time.perf_counter()
        rgb = load_rgb(path)
        res = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        out = {"second": second_from_path(path), "frame": path.name,
               "face_found": bool(res.face_landmarks)}
        if not res.face_landmarks:
            out.update(looking_at_screen=None, looking_at_screen_score=None, head_pose=None, aus=None,
                       note="Yüz bulunamadı: dikkat/bakış ölçülemedi (0 değil, bilinmiyor)")
        else:
            lm = res.face_landmarks[0]
            bs = {c.category_name: float(c.score) for c in res.face_blendshapes[0]}
            pose = _head_pose_deg(res.facial_transformation_matrixes[0])

            # Göz sapması (yaklaşık derece). İşaret konvansiyonu doğrulanmadığı için baş açısıyla
            # toplanmıyor; iki sapma ayrı ayrı cezalandırılıyor (muhafazakâr).
            eye_h = ((bs["eyeLookOutLeft"] - bs["eyeLookInLeft"])
                     + (bs["eyeLookInRight"] - bs["eyeLookOutRight"])) / 2
            eye_v = ((bs["eyeLookUpLeft"] + bs["eyeLookUpRight"])
                     - (bs["eyeLookDownLeft"] + bs["eyeLookDownRight"])) / 2
            eye_yaw, eye_pitch = eye_h * config.EYE_LOOK_TO_DEG, eye_v * config.EYE_LOOK_TO_DEG

            def g(yaw, pitch):
                return math.exp(-(yaw / config.GAZE_SIGMA_YAW_DEG) ** 2
                                - (pitch / config.GAZE_SIGMA_PITCH_DEG) ** 2)
            head_score = g(pose["yaw"], pose["pitch"])
            eye_score = g(eye_yaw, eye_pitch)
            blink = (bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2
            eyes_closed = blink > config.EYE_CLOSED_BLINK_THRESHOLD
            score = 0.0 if eyes_closed else head_score * eye_score

            out.update(
                head_pose=pose,
                eye_gaze_deg={"yaw": round(eye_yaw, 2), "pitch": round(eye_pitch, 2)},
                iris_ratio={"right_eye": _iris_ratio(lm, _R_EYE), "left_eye": _iris_ratio(lm, _L_EYE)},
                blink={"left": round(bs["eyeBlinkLeft"], 4), "right": round(bs["eyeBlinkRight"], 4),
                       "eyes_closed": eyes_closed},
                looking_at_screen_score=round(score, 4),
                looking_at_screen=score >= config.LOOKING_AT_SCREEN_THRESHOLD,
                score_parts={"head": round(head_score, 4), "eyes": round(eye_score, 4)},
                blendshapes={k: round(bs[k], 4) for k in _BLENDSHAPES_KEEP if k in bs},
                aus=aus_from_blendshapes(bs),
            )
        out["timing_ms"] = {"total": round((time.perf_counter() - t0) * 1000, 1)}
        return out

    def analyze(self, frames) -> dict:
        return {
            "tool": "mediapipe_face_landmarker",
            "device": "cpu",  # Python Tasks API Windows'ta GPU delegate desteklemiyor
            "frames": [self.analyze_frame(p) for p in list_frames(frames)],
        }

    def close(self):
        self._lm.close()


def analyze_frames(frames) -> dict:
    tool = MediaPipeTool()
    try:
        return tool.analyze(frames)
    finally:
        tool.close()
