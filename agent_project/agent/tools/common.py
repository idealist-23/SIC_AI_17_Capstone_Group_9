"""Tool'ların ortak yardımcıları: kare listeleme, saniye çıkarma, görüntü yükleme."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SECOND_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)


def list_frames(frames: str | Path | list) -> list[Path]:
    """Klasör yolu, tek kare yolu veya kare listesi -> sıralı Path listesi."""
    if isinstance(frames, (list, tuple)):
        return [Path(f) for f in frames]
    p = Path(frames)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    return [p]


def second_from_path(path: Path) -> int | None:
    """main.py kareleri `frame_{ss}.jpg` diye kaydediyor; başka adlar için None."""
    m = _SECOND_RE.search(path.stem)
    return int(m.group(1)) if m else None


def load_rgb(path: Path) -> np.ndarray:
    # cv2.imread Windows'ta ASCII dışı yollarda (ör. "Masaüstü") başarısız oluyor; PIL kullan.
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
