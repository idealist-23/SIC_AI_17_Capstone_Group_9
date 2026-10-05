"""Faz 1 test + süre ölçümü.

Kullanım: python -m agent.tools.test_tools [kare_klasörü] [--repeat N]
Model yükleme süresi ayrı, ısınma (ilk kare) ayrı, sıcak kare süresi ayrı raporlanır.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from agent import config
from agent.tools.common import list_frames


def _bench(name, make_tool, frames, repeat):
    t0 = time.perf_counter()
    tool = make_tool()
    load_s = time.perf_counter() - t0
    result = tool.analyze(frames)  # ilk geçiş: ısınma dahil
    first_ms = result["frames"][0]["timing_ms"]["total"]
    warm = []
    for _ in range(repeat):
        for p in frames:
            warm.append(tool.analyze_frame(p)["timing_ms"]["total"])
    tool.close()
    stats = {
        "tool": name,
        "model_load_s": round(load_s, 2),
        "first_frame_ms": first_ms,
        "warm_mean_ms": round(statistics.mean(warm), 1),
        "warm_median_ms": round(statistics.median(warm), 1),
        "warm_n": len(warm),
    }
    return result, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", nargs="?", default="test_frames")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", default="agent_results/phase1")
    args = ap.parse_args()

    frames = list_frames(args.frames)
    if not frames:
        raise SystemExit(f"Kare bulunamadı: {args.frames}")
    out_dir = config.ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from agent.tools.hsemotion_tool import HSEmotionTool
    from agent.tools.mediapipe_tool import MediaPipeTool

    all_stats = []
    for name, make in [("hsemotion", HSEmotionTool), ("mediapipe", MediaPipeTool)]:
        result, stats = _bench(name, make, frames, args.repeat)
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
        all_stats.append(stats)
        print(f"\n=== {name} ===")
        for f in result["frames"]:
            if name == "hsemotion":
                print(f"  {f['frame']:28s} face={f['face_found']!s:5s} "
                      f"dom={f['dominant_emotion']} V={f['valence']} A={f['arousal']}")
            elif name == "mediapipe":
                print(f"  {f['frame']:28s} face={f['face_found']!s:5s} "
                      f"pose={f.get('head_pose')} look={f.get('looking_at_screen_score')}")
            else:
                top = sorted((f.get("aus") or {}).items(), key=lambda kv: -kv[1])[:4]
                print(f"  {f['frame']:28s} face={f['face_found']!s:5s} pose={f.get('head_pose')} top_aus={top}")
        print("  stats:", stats)
    (out_dir / "timings.json").write_text(json.dumps(all_stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
