"""Faz 4 deneyleri (GEMINI_API_KEY gerekli).

  python -m agent.experiments same    --frame  test_frames/frame_06.jpg [--repeat 3]   # 1 çağrı/tekrar
  python -m agent.experiments shuffle --frames test_frames/            [--repeat 3]   # 3 çağrı/tekrar

same: Aynı kare 5 kez "2,4,6,8,10. saniye" diye gönderilir. Kareler özdeş olduğu için dürüst bir model
      düz bir eğri vermeli. Her tekrarda valence_range raporlanır; belirgin aralık = "hikâye eğrisi" uyduruluyor.
shuffle: Her tekrarda üç çağrı: ileri (A), yine ileri (B), ters (R).
      - baz çizgi (gürültü)  = |A - B| kare başına ortalama   (aynı sıra, iki ayrı çağrı)
      - sıra etkisi          = |A - R| kare başına ortalama   (aynı kare, farklı pozisyon)
      Sıra etkisi ancak gürültüden belirgin büyükse "var" denir (eşikler config'te):
        order_mad > ORDER_EFFECT_RATIO * baseline_mad  VE  order_mad - baseline_mad > ORDER_EFFECT_MIN_DELTA
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from agent import config, gemini_client
from agent.tools.common import list_frames


def _sorted(analysis: list) -> list:
    return sorted(analysis, key=lambda r: r.get("second", 0))


def _series(analysis: list, key: str) -> list:
    return [(r.get("affective_core") or {}).get(key) for r in _sorted(analysis)]


def _aida(analysis: list) -> list:
    return [(r.get("marketing_funnel") or {}).get("aida_stage") for r in _sorted(analysis)]


def _nums(xs) -> list[float]:
    return [float(x) for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]


def _call(frames: list[Path]) -> list:
    raw = gemini_client.generate_json(gemini_client.load_batch_ux_prompt(), images=frames)
    analysis, err = gemini_client.normalize_analysis(raw)
    if err:
        raise ValueError(err)
    return analysis


def _mad(a: list, b: list) -> float | None:
    d = [abs(x - y) for x, y in zip(a, b)
         if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    return statistics.mean(d) if d else None


# ---------------- same ----------------

def same_frame(frame: Path, repeat: int = 3) -> dict:
    reps = []
    for i in range(repeat):
        analysis = _call([frame] * 5)
        val = _series(analysis, "valence_score")
        nums = _nums(val)
        reps.append({
            "rep": i + 1, "analysis": analysis, "valence": val,
            "arousal": _series(analysis, "arousal_score"), "aida": _aida(analysis),
            "valence_range": round(max(nums) - min(nums), 4) if nums else None,
            "valence_std": round(statistics.pstdev(nums), 4) if len(nums) > 1 else None,
            "last_minus_first_valence": round(nums[-1] - nums[0], 4) if len(nums) > 1 else None,
            "distinct_aida": len(set(_aida(analysis))),
        })
    ranges = [r["valence_range"] for r in reps if r["valence_range"] is not None]
    return {
        "experiment": "same_frame", "frame": frame.name, "repeat": repeat, "reps": reps,
        "valence_range_per_rep": [r["valence_range"] for r in reps],
        "valence_range_mean": round(statistics.mean(ranges), 4) if ranges else None,
        "valence_range_max": max(ranges) if ranges else None,
        "distinct_aida_per_rep": [r["distinct_aida"] for r in reps],
        "interpretation": "Özdeş kareler: her tekrarda valence_range ~0 ve tek AIDA aşaması beklenir. "
                          "Tekrarlar boyunca tutarlı biçimde belirgin aralık = sıra/pozisyon kaynaklı uydurma.",
    }


# ---------------- shuffle ----------------

def shuffle(frames: list[Path], repeat: int = 3) -> dict:
    reps = []
    for i in range(repeat):
        fwd = _call(frames)
        fwd2 = _call(frames)
        rev = _call(list(reversed(frames)))
        a, b, r = (_series(x, "valence_score") for x in (fwd, fwd2, rev))
        n = min(len(a), len(b), len(r), len(frames))
        a, b, r = a[:n], b[:n], r[:n]
        r_by_frame = list(reversed(r))  # ters çağrıdaki i. kare, (n-1-i). pozisyonda
        reps.append({
            "rep": i + 1, "forward": fwd, "forward_again": fwd2, "reversed": rev,
            "per_frame": [{"frame": frames[j].name, "forward": a[j], "forward_again": b[j],
                           "reversed": r_by_frame[j]} for j in range(n)],
            "baseline_mad": _mad(a, b),                 # aynı sıra, iki çağrı -> gürültü
            "order_mad_same_frame": _mad(a, r_by_frame),  # aynı kare, farklı pozisyon
            "mad_same_position": _mad(a, r),              # aynı pozisyon, farklı kare
        })
    return {"experiment": "shuffle", "frames": [f.name for f in frames], "repeat": repeat, "reps": reps,
            **summarize_shuffle(reps)}


def summarize_shuffle(reps: list[dict]) -> dict:
    base = _nums(r["baseline_mad"] for r in reps)
    order = _nums(r["order_mad_same_frame"] for r in reps)
    pos = _nums(r["mad_same_position"] for r in reps)
    if not base or not order:
        return {"order_effect": "belirsiz", "reason": "yeterli sayısal valence yok"}
    b, o = statistics.mean(base), statistics.mean(order)
    effect = (o > config.ORDER_EFFECT_RATIO * b) and (o - b > config.ORDER_EFFECT_MIN_DELTA)
    return {
        "baseline_mad_mean": round(b, 4), "baseline_mad_per_rep": [round(x, 4) for x in base],
        "order_mad_mean": round(o, 4), "order_mad_per_rep": [round(x, 4) for x in order],
        "same_position_mad_mean": round(statistics.mean(pos), 4) if pos else None,
        "order_effect": "var" if effect else "yok (gürültüden ayırt edilemiyor)",
        "rule": f"order_mad > {config.ORDER_EFFECT_RATIO} x baseline_mad VE "
                f"order_mad - baseline_mad > {config.ORDER_EFFECT_MIN_DELTA}",
        "interpretation": "Kareye bağlı ölçümde order_mad ~ baseline_mad olmalı. order_mad gürültüden belirgin "
                          "büyükse sonuç saniye etiketine/sıraya bağlıdır. same_position_mad küçükse model "
                          "kareye değil pozisyona bakıyor olabilir.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["same", "shuffle"])
    ap.add_argument("--frame")
    ap.add_argument("--frames")
    ap.add_argument("--repeat", type=int, default=3, help="Tekrar sayısı (varsayılan 3)")
    args = ap.parse_args()
    if not gemini_client.has_key():
        raise SystemExit("GEMINI_API_KEY ortam değişkeni gerekli (Faz 4 deneyleri gerçek Gemini çağrısı yapar)")
    if args.experiment == "same":
        if not args.frame:
            raise SystemExit("--frame gerekli")
        res = same_frame(Path(args.frame), args.repeat)
    else:
        if not args.frames:
            raise SystemExit("--frames gerekli")
        res = shuffle(list_frames(args.frames), args.repeat)
    res["model"] = config.GEMINI_MODEL
    out = config.RESULTS_DIR / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.experiment}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "reps"}, indent=2, ensure_ascii=False))
    print("Kaydedildi:", path)


if __name__ == "__main__":
    main()
