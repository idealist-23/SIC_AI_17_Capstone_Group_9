"""Faz 5: Reklam tarafı, Warwick TSAM ("Decoding Viewer Emotions in Video Ads", Antonov et al. 2024).

Model, 5 sn'lik video+ses klibinden izleyicilerin en çok bildirdiği duyguyu tahmin eder (8 sınıf; HSEmotion ile
aynı etiket kümesi). Burada "beklenen duygu" olarak kullanılır ve reklam başına önbelleğe alınır.

Lisans: University of Warwick TSAM lisansı. YALNIZCA akademik araştırma / ticari olmayan değerlendirme.
Kod: third_party/DecodingViewerEmotions (değiştirilmeden import edilir). Ağırlıklar: models/tsam/tsam_weights.tar.

Ön işleme, repo'nun kendi scriptiyle aynı (mvlib/video2frames.py):
  kareler: ffmpeg -vf "scale=-1:256,fps=10" -q:v 0 %06d.jpg ; ses: ffmpeg -i video.mp4 id.wav
ffmpeg, .venv içindeki imageio-ffmpeg ikilisinden gelir (sistem kurulumu yok).

Kullanım: python -m agent.ad_analysis sample_ad.mp4   -> agent_results/ad_cache/<sha1>.json
"""
from __future__ import annotations

import codecs
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from agent import config

np = torch = None  # TSAM yolunda (önbellek yoksa) analyze_ad içinde yüklenir

REPO = config.ROOT / "third_party" / "DecodingViewerEmotions"
WEIGHTS = config.MODELS_DIR / "tsam" / "tsam_weights.tar"
CACHE_DIR = config.RESULTS_DIR / "ad_cache"
# Sınıf indeksi -> duygu (mvlib/mvideo_lib.py Emotions.list; emotion_ids 1..8 sıralı, arka plan sınıfı yok)
LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _prepare(video: Path, work: Path, vid: str, fps: int):
    frames = work / "frames" / vid
    audios = work / "audios"
    frames.mkdir(parents=True, exist_ok=True)
    audios.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg()
    subprocess.run([ff, "-loglevel", "error", "-y", "-i", str(video), "-vf", f"scale=-1:256,fps={fps}",
                    "-q:v", "0", str(frames / "%06d.jpg")], check=True)
    subprocess.run([ff, "-loglevel", "error", "-y", "-i", str(video), str(audios / f"{vid}.wav")], check=True)
    return frames, audios


def _load_checkpoint():
    """weights_only=True + açık izin listesi. Pickle içeriği statik olarak incelendi (pickletools):
    yalnızca tensör, OrderedDict, numpy scalar/dtype ve codecs.encode referansları var."""
    from numpy._core.multiarray import scalar
    safe = [(scalar, "numpy.core.multiarray.scalar"), np.dtype, codecs.encode,
            type(np.dtype("float64")), type(np.dtype("int64"))]
    with torch.serialization.safe_globals(safe):
        return torch.load(WEIGHTS, map_location="cpu", weights_only=True)


def _soundfile_load(path, *args, **kwargs):
    """torchaudio.load yerine: torchaudio 2.11 torchcodec/FFmpeg DLL'lerine ihtiyaç duyabiliyor.
    Aynı dönüş biçimi: (Tensor[kanal, örnek] float32, sr)."""
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


def _run_model(work: Path, device: str) -> list[dict]:
    sys.path.insert(0, str(REPO))
    try:
        import torchaudio
        from lib.dataset.dataset_predict import PredictFullVideoDataSet
        from lib.model.backbone import BackBone
        from lib.model.model import VCM
        from lib.model.prepare_input import X_input

        args = json.loads((REPO / "config" / "default.json").read_text(encoding="utf-8"))
        args["TSM"]["num_class"] = len(LABELS)
        args["TSM"]["main"]["backbone_weights"] = "__yok__"  # omurga checkpoint'ten yüklenecek
        args["dataset"]["data_dir"] = str(work)
        args["dataset"]["dir_frames"] = "frames"
        args["dataset"]["dir_audios"] = "audios"
        # predict.py / config_args.adjust_args_in ile aynı türetilmiş alanlar
        args["cuda_ids"] = [0]
        args["dataset"]["k_frames"] = 1
        args["video_segments"] = args["TSM"]["video_segments"]
        args["audio_segments"] = args["TSM"]["audio_segments"]
        args["segment_frames"] = 1

        model = VCM(args["TSM"], X_input(args["TSM"]), BackBone(args["TSM"]))
        state = _load_checkpoint()["model_state"]
        for name in model.model_layers:
            model.model_layers[name].load_state_dict(state[name])
        model.to(device).eval()

        orig_load = torchaudio.load
        torchaudio.load = _soundfile_load
        try:
            ds = PredictFullVideoDataSet(None, args, mode_train_val="predict")
            # predict.py'deki gibi DataLoader (batch boyutunu o ekliyor); Windows'ta num_workers=0
            loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            out = []
            with torch.inference_mode():
                for _id, t, xv, xa in loader:
                    t = int(t[0])
                    logits = model(xv.to(device), xa.to(device))
                    p = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
                    out.append({"start": int(t), "end": int(t) + args["emotion_jumps"]["clip_length"],
                                "expected_emotion": LABELS[int(p.argmax())],
                                "probs": {LABELS[j]: round(float(p[j]), 4) for j in range(len(LABELS))}})
        finally:
            torchaudio.load = orig_load
        return out
    finally:
        sys.path.remove(str(REPO))


def analyze_ad(video: str | Path, use_cache: bool = True, device: str | None = None) -> dict:
    video = Path(video)
    digest = _sha1(video)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{digest}.json"
    if use_cache and cache.exists():
        res = json.loads(cache.read_text(encoding="utf-8"))
        res["cached"] = True
        return res
    # Önbellek yoksa TSAM'ı çalıştırmak için ağır bağımlılıklar gerekir (torch, torchaudio, third_party repo).
    # Hafif sürümde bunlar kurulu değil; çağıran taraf (main_agent) hatayı yakalayıp içgörüsüz devam eder.
    global np, torch
    import numpy as np
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    work = CACHE_DIR / f"work_{digest[:12]}"
    t0 = time.perf_counter()
    try:
        _prepare(video, work, "ad", fps=10)
        t1 = time.perf_counter()
        segments = _run_model(work, device)
        t2 = time.perf_counter()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    res = {
        "model": "TSAM (Warwick, tsam_weights.tar; epoch 7, balanced acc 0.467)",
        "license": "Yalnızca akademik / ticari olmayan değerlendirme",
        "video": video.name, "sha1": digest, "device": device,
        "note": "Kayan 5 sn pencereler (1 sn adım). expected_emotion = izleyicilerin en çok bildirmesi beklenen duygu.",
        "segments": segments,
        "timing_s": {"preprocess": round(t1 - t0, 2), "model": round(t2 - t1, 2)},
        "cached": False,
    }
    cache.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    r = analyze_ad(a.video, use_cache=not a.no_cache)
    print(json.dumps({k: v for k, v in r.items() if k != "segments"}, indent=2, ensure_ascii=False))
    for s in r["segments"]:
        top = sorted(s["probs"].items(), key=lambda kv: -kv[1])[:3]
        print(f"  {s['start']:>2}-{s['end']:<2} sn  {s['expected_emotion']:10s} {top}")
