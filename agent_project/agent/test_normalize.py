"""Oturum 6 / madde 2: Gemini çıktısı normalleştirme testi (ağ yok).

Kullanım: python -m agent.test_normalize
"""
from agent import config, gemini_client, graph
from agent.gemini_client import normalize_analysis


def _item(sec=None):
    d = {"affective_core": {"valence_score": 0.1}}
    if sec is not None:
        d["second"] = sec
    return d


def main():
    # 1) Tek obje -> tek öğeli liste
    out, err = normalize_analysis(_item(4))
    assert err is None and out == [{"affective_core": {"valence_score": 0.1}, "second": 4}], out

    # 2) Sarmalanmış dict -> içteki liste
    out, err = normalize_analysis({"results": [_item(2), _item(4)]})
    assert err is None and [r["second"] for r in out] == [2, 4], out

    # 3) second'sız öğeler -> sırayla DEFAULT_SECONDS; sayı olmayan second da atanır
    out, err = normalize_analysis([_item(), _item(), {"second": "iki", "affective_core": {}}])
    assert err is None and [r["second"] for r in out] == config.DEFAULT_SECONDS[:3], out
    assert all(r["_second_assigned"] for r in out)
    out, err = normalize_analysis([_item("6"), _item(8.0)])  # string/float saniye -> int
    assert [r["second"] for r in out] == [6, 8], out

    # 4) Liste değil / boş -> hata
    for bad in ["metin", 42, None, [], {}, {"x": 1}, ["a", "b"]]:
        out, err = normalize_analysis(bad)
        assert out == [] and err, (bad, out, err)

    # 5) gemini_analyst düğümü: bozuk çıktıda status=error ve agreement'a liste gider
    gemini_client.has_key = lambda: True
    frames = [str(p) for p in sorted((config.ROOT / "test_frames_tmp").iterdir())]
    for raw, status, n in [({"results": [_item(), _item()]}, "ok", 2), ("bozuk", "error", 0),
                           (_item(), "ok", 1)]:
        gemini_client.generate_json = lambda *a, raw=raw, **k: raw
        res = graph.gemini_analyst({"frames": frames, "gemini_mode": "real", "iteration": 0})
        assert res["gemini_meta"]["status"] == status, res["gemini_meta"]
        assert isinstance(res["gemini_analysis"], list) and len(res["gemini_analysis"]) == n, res
    print("OK: tek obje, sarmalanmış dict, second'sız öğe, bozuk çıktı -> hepsi liste; hata durumunda status=error")


if __name__ == "__main__":
    main()
