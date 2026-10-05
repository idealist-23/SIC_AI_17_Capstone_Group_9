"""Faz 2 testi: agent_results/phase1/*.json + SENTETİK Gemini çıktısı (gerçek API çağrısı değil).

Kullanım: python -m agent.test_agreement
Beklenen (test_frames_tmp sırası: ad_2s, ad_6s, business, business_flip, portrait):
  2 sn: yüz yok + Gemini yüksek ilgi  -> attention conflict
  4 sn: yüz yok, Gemini nötr          -> unverifiable (conflict değil)
  6 sn: valence farkı büyük + AU iddiası (AU ölçümü yok) -> valence + au_claims conflict
  8 sn: bakmıyor (skor<0.5) + Gemini "arzu" -> attention conflict
  10 sn: uyumlu                       -> conflict yok
Ayrıca: Gemini-off (boş analiz) -> tüm kontroller unverifiable, hakem 'uyumlu' demez.
"""
import json

from agent import config
from agent.agreement import compute_agreement
from agent.judging import rule_judge
from agent.mock import SYNTHETIC_GEMINI


def main():
    d = config.ROOT / "agent_results" / "phase1"
    fm = {k: json.loads((d / f"{k}.json").read_text(encoding="utf-8")) for k in ("hsemotion", "mediapipe")}
    ad = {"segments": [{"start": 5, "end": 10, "expected_emotion": "Happiness"}]}
    res = compute_agreement(SYNTHETIC_GEMINI, fm, ad)
    (d / "agreement_synthetic.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for p in res["per_second"]:
        print(p["second"], "conflicts:", p["conflicts"], "unverifiable:", p["unverifiable"],
              "ratio:", p["conflict_ratio"], "insights:", [i["measured"] for i in p["insights"]])
    got = {p["second"]: sorted(p["conflicts"]) for p in res["per_second"]}
    expected = {2: ["attention"], 4: [], 6: ["au_claims", "valence"], 8: ["attention"], 10: []}
    assert got == expected, (got, expected)
    # İddia yoksa "agree" değil "unverifiable" (sahte uyum yok)
    unv = {p["second"]: sorted(p["unverifiable"]) for p in res["per_second"]}
    expected_unv = {2: ["au_claims", "valence"], 4: ["attention", "au_claims", "valence"],
                    6: ["attention"], 8: ["au_claims"], 10: ["au_claims"]}
    assert unv == expected_unv, (unv, expected_unv)
    verd = {v["second"]: v for v in rule_judge(res, fm)["verdicts"]}
    assert verd[10]["rationale"].startswith("Kaynaklar uyumlu"), verd[10]
    print("OK (mock Gemini):", res["summary"])

    # Gemini-off senaryosu: hiçbir kontrol yapılamaz, hakem "uyumlu" DEMEMELİ
    off = compute_agreement([], fm, ad)
    for p in off["per_second"]:
        assert p["conflicts"] == [] and p["conflict_ratio"] is None, p
        assert all(c["status"] == "unverifiable" and c["reason"] == "Gemini analizi yok"
                   for c in p["checks"].values()), p
    for v in rule_judge(off, fm)["verdicts"]:
        assert "uyumlu" not in v["rationale"], v
        assert v["confidence"] <= config.LOCAL_ONLY_MAX_CONF, v
        if v["trusted_source"] != "none":  # yüz bulunan saniyeler
            assert v["trusted_source"] == "local", v
            assert v["rationale"] == "Yalnızca yerel ölçüm; Gemini ile karşılaştırma yapılamadı.", v
    print("OK (Gemini off): tüm kontroller unverifiable, hakem 'uyumlu' demiyor, güven <=",
          config.LOCAL_ONLY_MAX_CONF)


if __name__ == "__main__":
    main()
