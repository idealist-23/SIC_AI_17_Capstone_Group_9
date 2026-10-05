"""Oturum 6 / madde 3: LLM hakem/öneri çıktısı şema doğrulaması (ağ yok, sahte generate_json).

Kullanım: python -m agent.test_validation
Her bozuk çıktı için: kural tabanlı yedeğe düşülmeli, fallback_reason yazılmalı, hiçbir istisna
fırlamamalı ve (graph üzerinden) reporter çalışmalı.
"""
import json

from agent import config, gemini_client, judging
from agent.agreement import compute_agreement
from agent.mock import SYNTHETIC_GEMINI

BAD_VERDICTS = [
    "düz metin",                                                  # dict değil
    {"verdicts": "liste değil"},                                  # verdicts liste değil
    {"verdicts": [{"trusted_source": "local", "confidence": 0.5}]},  # second eksik
    {"verdicts": [{"second": 2, "confidence": 0.5}]},             # trusted_source eksik
    {"verdicts": [{"second": 2, "trusted_source": "local"}]},     # confidence eksik
    {"verdicts": [{"second": 2, "trusted_source": "local", "confidence": 1.7}]},   # 0-1 dışı
    {"verdicts": [{"second": 2, "trusted_source": "local", "confidence": "yüksek"}]},
    {"verdicts": [{"second": "iki", "trusted_source": "local", "confidence": 0.5}]},
    {"verdicts": [42]},
]
BAD_RECS = [
    None, {"recommendations": {}},
    {"recommendations": [{"change": "x", "confidence": 0.5}]},
    {"recommendations": [{"second": 4, "confidence": 0.5}]},
    {"recommendations": [{"second": 4, "change": "x"}]},
    {"recommendations": [{"second": 4, "change": "x", "confidence": -0.1}]},
]


def main():
    d = config.ROOT / "agent_results" / "phase1"
    fm = {k: json.loads((d / f"{k}.json").read_text(encoding="utf-8")) for k in ("hsemotion", "mediapipe")}
    agr = compute_agreement(SYNTHETIC_GEMINI, fm)
    gemini_client.has_key = lambda: True

    for bad in BAD_VERDICTS:
        gemini_client.generate_json = lambda *a, bad=bad, **k: bad
        out = judging.judge(agr, fm, SYNTHETIC_GEMINI, use_gemini=True)
        assert out["source"] == "rules" and "fallback_reason" in out, (bad, out.get("source"))
    # Ağ/JSON istisnası da yedeğe düşmeli
    def boom(*a, **k):
        raise ValueError("Gemini JSON parse hatası: test")
    gemini_client.generate_json = boom
    out = judging.judge(agr, fm, SYNTHETIC_GEMINI, use_gemini=True)
    assert out["source"] == "rules" and "başarısız" in out["fallback_reason"], out
    verdicts = out

    # Geçerli çıktı kabul edilmeli, string saniye int'e çevrilmeli
    good = {"verdicts": [{"second": "6", "trusted_source": "local", "confidence": 0.4, "rationale": "x"}],
            "request_reanalysis": False}
    gemini_client.generate_json = lambda *a, **k: json.loads(json.dumps(good))
    ok = judging.judge(agr, fm, SYNTHETIC_GEMINI, use_gemini=True)
    assert ok["source"] == "gemini" and ok["verdicts"][0]["second"] == 6, ok

    for bad in BAD_RECS:
        gemini_client.generate_json = lambda *a, bad=bad, **k: bad
        out = judging.recommend(verdicts, [], None, use_gemini=True)
        assert out["source"] == "rules" and "fallback_reason" in out, (bad, out)

    # rule_recommender eksik alanlı / bozuk kararlarla KeyError fırlatmamalı
    for weird in [None, {}, {"verdicts": None}, {"verdicts": [{}, {"second": 4}, 7,
                                                              {"second": 6, "attention": "no"}]}]:
        judging.rule_recommender(weird, [], {"segments": [{"start": 0}, {"start": 3, "end": 8}]})

    # Madde 4: kalibre edilmemiş dikkat skoru -> "dikkat düşüyor" güveni tavanı geçmemeli
    v = {"verdicts": [{"second": 8, "trusted_source": "local", "confidence": 0.8, "attention": "no"}]}
    assert not config.ATTENTION_CALIBRATED
    r = judging.rule_recommender(v, [], None)["recommendations"][0]
    assert r["confidence"] == config.UNCALIBRATED_ATTENTION_MAX_CONF, r
    assert "(bakma skoru henüz kalibre edilmedi)" in r["reason"], r

    # Uçtan uca: graph + reporter bozuk hakem/öneri çıktısıyla çalışmalı
    from agent.graph import run
    gemini_client.generate_json = lambda *a, **k: {"verdicts": "bozuk", "recommendations": 5}
    final = run("test_frames_tmp", "validation_test", config.RESULTS_DIR / "validation_test", "real")
    assert final["verdicts"]["source"] == "rules" and final["verdicts"].get("fallback_reason")
    assert final["recommendations"]["source"] == "rules" and final["recommendations"].get("fallback_reason")
    assert (config.RESULTS_DIR / "validation_test" / "result.json").exists()
    print(f"OK: {len(BAD_VERDICTS)} bozuk hakem + 1 istisna + {len(BAD_RECS)} bozuk öneri çıktısı -> kural "
          "tabanlı yedek + fallback_reason; geçerli çıktı kabul; graph+reporter uçtan uca çalıştı")
    print("  örnek fallback_reason:", final["verdicts"]["fallback_reason"])


if __name__ == "__main__":
    main()
