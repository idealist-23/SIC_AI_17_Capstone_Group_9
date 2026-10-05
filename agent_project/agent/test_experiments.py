"""Oturum 6 / madde 7: deney karşılaştırma mantığı, sahte _call ile (ağ yok).

Kullanım: python -m agent.test_experiments
Senaryolar:
  A) "Dürüst" model: valence kareye bağlı + küçük gürültü  -> sıra etkisi YOK
  B) "Hikâye" modeli: valence pozisyona bağlı (kareyi yok sayar) -> sıra etkisi VAR
  C) Sadece yüksek gürültü (kareye bağlı ama çağrılar arası büyük oynama) -> gürültüden ayırt edilemez, YOK
  same: pozisyona bağlı model -> her tekrarda valence_range > 0; kareye bağlı model -> 0
"""
import random
from pathlib import Path

import agent.experiments as ex

FRAMES = [Path(f"f{i}.jpg") for i in range(5)]
FRAME_VAL = {f.name: v for f, v in zip(FRAMES, [0.6, -0.2, 0.3, -0.5, 0.1])}
STORY = [0.5, 0.3, 0.1, -0.2, -0.6]  # "son karede ilgisini kaybetti" eğrisi


def make_call(kind, noise, rng):
    def _call(frames):
        out = []
        for pos, f in enumerate(frames):
            base = FRAME_VAL[f.name] if kind == "frame" else STORY[pos]
            out.append({"second": 2 * (pos + 1),
                        "affective_core": {"valence_score": round(base + rng.uniform(-noise, noise), 3)},
                        "marketing_funnel": {"aida_stage": "ilgi"}})
        return out
    return _call


def main():
    rng = random.Random(0)
    cases = [("A dürüst", "frame", 0.03, "var", False), ("B hikâye", "position", 0.03, "var", True),
             ("C gürültülü", "frame", 0.4, "var", False)]
    for name, kind, noise, _, expect_effect in cases:
        ex._call = make_call(kind, noise, rng)
        r = ex.shuffle(FRAMES, repeat=3)
        got = r["order_effect"] == "var"
        print(f"{name:12s} baseline={r['baseline_mad_mean']:.3f} order={r['order_mad_mean']:.3f} "
              f"same_pos={r['same_position_mad_mean']:.3f} -> {r['order_effect']}")
        assert got == expect_effect, (name, r)
        assert len(r["reps"]) == 3 and len(r["baseline_mad_per_rep"]) == 3

    ex._call = make_call("position", 0.0, rng)
    s = ex.same_frame(Path("x.jpg"), repeat=3)
    print("same (hikâye) valence_range_per_rep:", s["valence_range_per_rep"])
    assert len(s["valence_range_per_rep"]) == 3 and all(v > 1.0 for v in s["valence_range_per_rep"])
    ex._call = lambda frames: [{"second": 2 * (i + 1), "affective_core": {"valence_score": 0.2}}
                               for i in range(len(frames))]
    s = ex.same_frame(Path("x.jpg"), repeat=3)
    print("same (dürüst) valence_range_per_rep:", s["valence_range_per_rep"])
    assert s["valence_range_per_rep"] == [0.0, 0.0, 0.0]
    print("OK: baz çizgi/sıra etkisi kararı ve same tekrarları beklendiği gibi")


if __name__ == "__main__":
    main()
