"""judge + recommender: Gemini ile (anahtar varsa) ya da kural tabanlı yedekle.
Her iki yol da aynı şemayı döndürür; `source` alanı hangisinin kullanıldığını söyler."""
from __future__ import annotations

import json

from agent import config, gemini_client
from agent.agreement import ad_segment_for, index_by_second

JUDGE_PROMPT = """Sen bir ölçüm hakemisin. Bir reklam izleyicisinin yüz kareleri için iki kaynak var:
(1) Gemini'nin görsel yorumu, (2) yerel yüz modellerinin ölçümleri (HSEmotion: duygu + valence/arousal,
MediaPipe: ekrana bakma skoru, baş pozu ve blendshape tabanlı Action Unit vekilleri).
Aşağıda saniye bazlı UYUM TABLOSU (kural tabanlı çelişki kontrolleri) ve iki kaynağın özetleri var.

Kurallar:
- Ölçüm yoksa (yüz bulunamadı) o saniye için yorum UYDURMA; güveni düşük ver.
- Gemini'nin kanıtsız AU iddialarını ve ölçümle çelişen dikkat iddialarını gerekçede belirt.
- "Son kare" olduğu için ilgi kaybı gibi hikâye kalıplarına karşı dikkatli ol; karar ölçüme dayanmalı.

SADECE şu JSON'u üret:
{"verdicts": [{"second": 2, "valence": -1..1 veya null, "attention": "yes|no|unknown",
  "emotion": "kısa etiket veya null", "trusted_source": "gemini|local|mixed|none",
  "confidence": 0..1, "rationale": "1-2 cümle"}],
 "request_reanalysis": true|false,
 "feedback_to_analyst": "Gemini analistine somut, saniye bazlı düzeltme notu (gerekmezse boş)"}

UYUM TABLOSU:
%s

YEREL ÖLÇÜM ÖZETİ:
%s

GEMINI ANALİZİ:
%s
"""

RECOMMENDER_PROMPT = """Sen bir reklam kurgu danışmanısın. Aşağıda izleyicinin saniye bazlı doğrulanmış
tepkileri (hakem kararları, güven skorlarıyla), tıklamalar ve (varsa) reklamın zaman çizelgesi var.
Reklamın nasıl iyileştirileceğine dair SOMUT düzenleme önerileri üret. Güveni düşük saniyeler için
kesin öneri verme; önerinin güveni dayandığı kararın güvenini aşmasın.

SADECE şu JSON'u üret:
{"recommendations": [{"second": 4, "change": "ne değişmeli", "reason": "neden (hangi ölçüme dayanarak)",
  "confidence": 0..1}], "overall": "1-2 cümle genel değerlendirme"}

KARARLAR:
%s

TIKLAMALAR:
%s

REKLAM ANALİZİ:
%s
"""


def summarize_face_metrics(face_metrics: dict) -> list[dict]:
    """LLM'e verilecek kompakt saniye özeti (ham 478 landmark vb. gönderilmez)."""
    hse = index_by_second(face_metrics.get("hsemotion"))
    mp = index_by_second(face_metrics.get("mediapipe"))
    rows = []
    for sec in sorted(set(hse) | set(mp), key=lambda s: (s is None, s)):
        h, m = hse.get(sec) or {}, mp.get(sec) or {}
        aus = m.get("aus") or {}
        rows.append({
            "second": sec,
            "face_found": bool(h.get("face_found") or m.get("face_found")),
            "face_found_by": [n for n, d in (("hsemotion", h), ("mediapipe", m))
                              if d.get("face_found")],
            "hsemotion": {"emotion": h.get("dominant_emotion"), "valence": h.get("valence"),
                          "arousal": h.get("arousal")},
            "mediapipe": {"looking_at_screen_score": m.get("looking_at_screen_score"),
                          "eyes_closed": (m.get("blink") or {}).get("eyes_closed")},
            "active_aus": sorted([a for a, v in aus.items()
                                         if v is not None and v >= config.AU_PRESENCE_THRESHOLD],
                                        key=lambda a: int(a[2:])),
        })
    return rows


def _compact_agreement(agreement: dict) -> list[dict]:
    return [{"second": p["second"], "conflicts": p["conflicts"], "unverifiable": p["unverifiable"],
             "conflict_ratio": p["conflict_ratio"],
             "valence": {k: p["checks"]["valence"].get(k) for k in ("gemini", "hsemotion", "abs_diff")},
             "attention": {k: p["checks"]["attention"].get(k)
                           for k in ("looking_at_screen_score", "gemini_claims", "reason")},
             "au_unsupported": p["checks"]["au_claims"].get("unsupported"),
             "insights": p["insights"]} for p in agreement["per_second"]]


# ---------------- judge ----------------

def rule_judge(agreement: dict, face_metrics: dict) -> dict:
    """Kural tabanlı yedek hakem: ölçüm varsa yerel kaynağa güven, yoksa 'bilinmiyor'."""
    summary = {r["second"]: r for r in summarize_face_metrics(face_metrics)}
    verdicts = []
    for p in agreement["per_second"]:
        s = summary.get(p["second"], {})
        hv = (s.get("hsemotion") or {}).get("valence")
        look = (s.get("mediapipe") or {}).get("looking_at_screen_score")
        if not s.get("face_found"):
            verdicts.append({"second": p["second"], "valence": None, "attention": "unknown", "emotion": None,
                             "trusted_source": "none", "confidence": 0.1,
                             "rationale": "Yüz ölçülemedi; Gemini yorumu doğrulanamaz."
                                          + (f" (Yalnızca {s['face_found_by']} yüz buldu; tutarsız, yok sayıldı.)"
                                             if s.get("face_found_by") else "")})
            continue
        ratio = p.get("conflict_ratio") or 0.0
        conf = round(max(0.1, 0.9 - 0.6 * ratio - 0.1 * len(p.get("unverifiable", []))), 2)
        checks = p.get("checks") or {}
        agreed = [k for k, c in checks.items() if c.get("status") == "agree"]
        if p.get("conflicts"):
            source = "local"
            rationale = "Çelişkiler: " + ", ".join(p["conflicts"]) + "; yerel ölçüm esas alındı."
        elif agreed:
            # "Uyumlu" yalnızca en az bir kontrol gerçekten karşılaştırılıp tuttuysa
            source = "mixed"
            rationale = "Kaynaklar uyumlu (" + ", ".join(agreed) + ")."
        else:
            # Hiçbir kontrol ölçülemedi (ör. Gemini atlandı): karşılaştırma yok, sadece yerel ölçüm
            source = "local"
            conf = min(conf, config.LOCAL_ONLY_MAX_CONF)
            rationale = "Yalnızca yerel ölçüm; Gemini ile karşılaştırma yapılamadı."
        verdicts.append({
            "second": p["second"], "valence": hv,
            "attention": "unknown" if look is None else
                         ("yes" if look >= config.LOOKING_AT_SCREEN_THRESHOLD else "no"),
            "emotion": (s.get("hsemotion") or {}).get("emotion"),
            "trusted_source": source,
            "confidence": conf,
            "rationale": rationale,
        })
    ratios = [p["conflict_ratio"] for p in agreement["per_second"] if p["conflict_ratio"] is not None]
    mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return {"source": "rules", "verdicts": verdicts,
            "request_reanalysis": mean_ratio > config.REANALYZE_CONFLICT_RATIO,
            "feedback_to_analyst": _feedback_from_agreement(agreement)}


def _feedback_from_agreement(agreement: dict) -> str:
    notes = []
    for p in agreement["per_second"]:
        c = p["checks"]
        if "valence" in p["conflicts"]:
            notes.append(f"{p['second']}. sn: valence {c['valence']['gemini']} dedin, yerel ölçüm "
                         f"{c['valence']['hsemotion']}. Yüzü yeniden değerlendir.")
        if "attention" in p["conflicts"]:
            notes.append(f"{p['second']}. sn: {c['attention'].get('reason')}. İlgi iddiasını kanıtla ya da düşür.")
        if "au_claims" in p["conflicts"]:
            notes.append(f"{p['second']}. sn: {c['au_claims'].get('unsupported')} AU'ları ölçümle desteklenmiyor; "
                         "yalnızca görünür olanları yaz.")
    return "\n".join(notes)


def _valid_conf(c) -> bool:
    return isinstance(c, (int, float)) and not isinstance(c, bool) and 0.0 <= c <= 1.0


def _valid_second(x) -> bool:
    try:
        int(float(x))
        return True
    except (TypeError, ValueError):
        return False


def validate_verdicts(out) -> str | None:
    """LLM hakem çıktısının şema kontrolü. Sorun yoksa None, varsa açıklama döner."""
    if not isinstance(out, dict) or not isinstance(out.get("verdicts"), list):
        return "verdicts liste değil"
    for i, v in enumerate(out["verdicts"]):
        if not isinstance(v, dict):
            return f"verdicts[{i}] dict değil"
        for k in ("second", "trusted_source", "confidence"):
            if k not in v:
                return f"verdicts[{i}] içinde '{k}' eksik"
        if not _valid_second(v["second"]):
            return f"verdicts[{i}].second sayı değil: {v['second']!r}"
        if not _valid_conf(v["confidence"]):
            return f"verdicts[{i}].confidence 0-1 dışında: {v['confidence']!r}"
    return None


def validate_recommendations(out) -> str | None:
    """LLM öneri çıktısının şema kontrolü. Sorun yoksa None, varsa açıklama döner."""
    if not isinstance(out, dict) or not isinstance(out.get("recommendations"), list):
        return "recommendations liste değil"
    for i, r in enumerate(out["recommendations"]):
        if not isinstance(r, dict):
            return f"recommendations[{i}] dict değil"
        for k in ("second", "change", "confidence"):
            if k not in r:
                return f"recommendations[{i}] içinde '{k}' eksik"
        if not _valid_second(r["second"]):
            return f"recommendations[{i}].second sayı değil: {r['second']!r}"
        if not _valid_conf(r["confidence"]):
            return f"recommendations[{i}].confidence 0-1 dışında: {r['confidence']!r}"
    return None


def judge(agreement: dict, face_metrics: dict, gemini_analysis: list, use_gemini: bool) -> dict:
    if use_gemini and gemini_client.has_key():
        prompt = JUDGE_PROMPT % (json.dumps(_compact_agreement(agreement), ensure_ascii=False),
                                 json.dumps(summarize_face_metrics(face_metrics), ensure_ascii=False),
                                 json.dumps(gemini_analysis, ensure_ascii=False))
        try:
            out = gemini_client.generate_json(prompt, temperature=0.2)
            problem = validate_verdicts(out)
        except Exception as e:  # noqa: BLE001 - ağ/JSON hatası da yedeğe düşer
            out, problem = None, f"Gemini hakem çağrısı başarısız: {e!r}"
        if problem is None:
            for v in out["verdicts"]:
                v["second"] = int(float(v["second"]))
            out["source"] = "gemini"
            return out
        fb = rule_judge(agreement, face_metrics)
        fb["fallback_reason"] = f"LLM hakem çıktısı şemaya uymadı: {problem}"
        return fb
    return rule_judge(agreement, face_metrics)


# ---------------- recommender ----------------

def rule_recommender(verdicts: dict, clicks: list, ad_analysis: dict | None) -> dict:
    recs = []
    for v in (verdicts or {}).get("verdicts") or []:
        if not isinstance(v, dict) or v.get("trusted_source") == "none" or v.get("second") is None:
            continue
        sec = v.get("second")
        conf = v.get("confidence") if isinstance(v.get("confidence"), (int, float)) else 0.0
        if v.get("attention") == "no":
            reason, att_conf = "MediaPipe: izleyici ekrana bakmıyor.", conf
            if not config.ATTENTION_CALIBRATED:
                # Bakma skoru sezgisel eşiklerle hesaplanıyor; kalibrasyona kadar güvene tavan
                att_conf = min(conf, config.UNCALIBRATED_ATTENTION_MAX_CONF)
                reason += " (bakma skoru henüz kalibre edilmedi)"
            recs.append({"second": sec, "change": "Bu aralıkta dikkat düşüyor: görsel tempo/kesme ya da hareket ekle.",
                         "reason": reason, "confidence": att_conf})
        val = v.get("valence")
        if isinstance(val, (int, float)) and val < -0.2:
            recs.append({"second": sec, "change": "Olumsuz tepki alan sahneyi/mesajı yumuşat ya da kısalt.",
                         "reason": f"HSEmotion valence {val}.", "confidence": conf})
        seg = ad_segment_for(sec, ad_analysis)
        emo = v.get("emotion")
        if seg and emo and seg.get("expected_emotion") and seg.get("expected_emotion") != emo:
            # İçgörü: reklam modeli (TSAM) bu aralıkta başka bir duygu bekliyor. Tek izleyici + model
            # belirsizliği nedeniyle güven yarıya indirilir.
            recs.append({"second": sec,
                         "change": f"Reklam modeli bu aralıkta izleyicilerde '{seg.get('expected_emotion')}' tepkisi "
                                   f"bekliyor, bu izleyicide '{emo}' ölçüldü: sahnenin amaçlanan duyguyu "
                                   "uyandırıp uyandırmadığını gözden geçir (tek izleyici, düşük güven).",
                         "reason": f"TSAM beklenen={seg.get('expected_emotion')} "
                                   f"({seg.get('start')}-{seg.get('end')} sn), HSEmotion ölçülen={emo}.",
                         "confidence": round(conf * 0.5, 2)})
    return {"source": "rules", "recommendations": recs,
            "overall": "Kural tabanlı öneriler (Gemini kullanılmadı); ölçülen dikkat/valence düşüşlerine ve "
                       "(varsa) reklam modelinin beklediği duygu ile ölçülen duygu farkına dayanır."}


def recommend(verdicts: dict, clicks: list, ad_analysis: dict | None, use_gemini: bool) -> dict:
    if use_gemini and gemini_client.has_key():
        prompt = RECOMMENDER_PROMPT % (json.dumps((verdicts or {}).get("verdicts"), ensure_ascii=False),
                                       json.dumps(clicks, ensure_ascii=False),
                                       json.dumps(ad_analysis, ensure_ascii=False) if ad_analysis else "yok")
        try:
            out = gemini_client.generate_json(prompt, temperature=0.3)
            problem = validate_recommendations(out)
        except Exception as e:  # noqa: BLE001
            out, problem = None, f"Gemini öneri çağrısı başarısız: {e!r}"
        if problem is None:
            for r in out["recommendations"]:
                r["second"] = int(float(r["second"]))
            out["source"] = "gemini"
            return out
        fb = rule_recommender(verdicts, clicks, ad_analysis)
        fb["fallback_reason"] = f"LLM öneri çıktısı şemaya uymadı: {problem}"
        return fb
    return rule_recommender(verdicts, clicks, ad_analysis)
