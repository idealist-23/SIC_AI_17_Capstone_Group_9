"""CLI: python -m agent.run --frames test_frames/ [--gemini real|mock|off] [--clicks clicks.csv]
Sonuç: agent_results/<session_id>/{result.json, per_second.csv, chart.png}
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from agent import config, gemini_client


def _load_clicks(path: str | None) -> list:
    if not path:
        return []
    import pandas as pd
    return pd.read_csv(path).to_dict(orient="records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="Kare klasörü (frame_XX.jpg) ya da görsel dosyası")
    ap.add_argument("--gemini", choices=["real", "mock", "off"], default=None,
                    help="Varsayılan: GEMINI_API_KEY varsa real, yoksa off")
    ap.add_argument("--mock-gemini", action="store_true", help="--gemini mock kısayolu")
    ap.add_argument("--clicks", help="main.py'nin click CSV'si (video_time sütunu)")
    ap.add_argument("--ad-analysis", help="Reklam analizi JSON (segments: start/end/expected_emotion)")
    ap.add_argument("--ad-video", help="Reklam videosu (ör. sample_ad.mp4): TSAM ile beklenen duygu çıkarılır")
    ap.add_argument("--session-id", default=None)
    args = ap.parse_args()

    mode = "mock" if args.mock_gemini else (args.gemini or ("real" if gemini_client.has_key() else "off"))
    if mode == "real" and not gemini_client.has_key():
        raise SystemExit("--gemini real için GEMINI_API_KEY ortam değişkeni gerekli")
    session_id = args.session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.RESULTS_DIR / session_id
    ad = json.loads(Path(args.ad_analysis).read_text(encoding="utf-8")) if args.ad_analysis else None
    if args.ad_video and ad is None:
        from agent.ad_analysis import analyze_ad  # TSAM; sonuç reklam başına önbellekte
        ad = analyze_ad(args.ad_video)
        print(f"Reklam analizi (TSAM): {len(ad['segments'])} pencere, önbellekten={ad.get('cached')}")

    from agent.graph import run  # ağır import'lar (onnxruntime, mediapipe) burada
    t0 = time.perf_counter()
    final = run(args.frames, session_id, out_dir, mode, _load_clicks(args.clicks), ad)
    total = time.perf_counter() - t0

    print(f"gemini_mode={mode} | iterasyon={final.get('iteration')} | hakem={final['verdicts'].get('source')}"
          f" | öneri={final['recommendations'].get('source')}")
    for t in final.get("timings", []):
        print(f"  {t['node']:15s} {t['ms']:>9.1f} ms", {k: v for k, v in t.items() if k not in ('node', 'ms')} or "")
    print(f"  {'TOPLAM':15s} {total * 1000:>9.1f} ms")
    print("Çıktı:", out_dir)


if __name__ == "__main__":
    main()
