"""Faz 2: Gemini yorumu ile yerel ölçümler arasındaki uyum (LLM YOK, düz Python).

Girdi:
  gemini_analysis: BATCH_UX_PROMPT şemasındaki liste (her öğede "second").
  face_metrics: {"hsemotion": <hsemotion_tool çıktısı>, "mediapipe": <mediapipe_tool çıktısı>,
                 (mediapipe frames[i]["aus"] = {"AU12": 0.8, ...}: blendshape tabanlı AU vekili)}
  ad_analysis: opsiyonel {"segments": [{"start": 0, "end": 5, "expected_emotion": "Happiness"}]}

Her kontrolün sonucu: "agree" | "conflict" | "unverifiable" (+ neden). Ölçülemeyen durum
çelişki sayılmaz, ama ayrıca raporlanır.
"""
from __future__ import annotations

import re

from agent import config

_AU_RE = re.compile(r"\bAU\s?0?(\d{1,2})\b", re.IGNORECASE)


def index_by_second(tool_output: dict | None) -> dict:
    """Tool çıktısını saniyeye göre indeksle; saniyesiz karelere sırayla DEFAULT_SECONDS ata."""
    if not tool_output:
        return {}
    out = {}
    for i, f in enumerate(tool_output.get("frames", [])):
        sec = f.get("second")
        if sec is None and i < len(config.DEFAULT_SECONDS):
            sec = config.DEFAULT_SECONDS[i]
        out[sec] = f
    return out


def _check_valence(gem: dict, hse: dict | None) -> dict:
    g = (gem.get("affective_core") or {}).get("valence_score")
    if g is None:
        return {"status": "unverifiable", "reason": "Gemini valence yok"}
    if not hse or not hse.get("face_found"):
        return {"status": "unverifiable", "reason": "HSEmotion yüz bulamadı", "gemini": g}
    diff = abs(float(g) - hse["valence"])
    return {
        "status": "conflict" if diff > config.VALENCE_DIFF_THRESHOLD else "agree",
        "gemini": float(g), "hsemotion": hse["valence"], "abs_diff": round(diff, 3),
        "threshold": config.VALENCE_DIFF_THRESHOLD,
    }


def _gemini_high_interest(gem: dict) -> list[str]:
    reasons = []
    intent = (gem.get("commercial_signals") or {}).get("purchase_intent")
    aida = (gem.get("marketing_funnel") or {}).get("aida_stage")
    emo = (gem.get("affective_core") or {}).get("dominant_emotion")
    if intent in config.HIGH_INTEREST_PURCHASE_INTENT:
        reasons.append(f"purchase_intent={intent}")
    if aida in config.HIGH_INTEREST_AIDA:
        reasons.append(f"aida_stage={aida}")
    if emo in config.HIGH_INTEREST_EMOTIONS:
        reasons.append(f"dominant_emotion={emo}")
    return reasons


def _check_attention(gem: dict, mp: dict | None) -> dict:
    claims = _gemini_high_interest(gem)
    if not mp:
        return {"status": "unverifiable", "reason": "MediaPipe çıktısı yok", "gemini_claims": claims}
    if not mp.get("face_found"):
        # Yüz yokken Gemini'nin herhangi bir duygu/ilgi yorumu yapması başlı başına şüpheli.
        return {"status": "conflict" if claims else "unverifiable",
                "reason": "MediaPipe yüz bulamadı" + (", ama Gemini yüksek ilgi diyor" if claims else ""),
                "gemini_claims": claims}
    looking = mp["looking_at_screen"]
    res = {"looking_at_screen": looking, "looking_at_screen_score": mp["looking_at_screen_score"],
           "gemini_claims": claims}
    if not claims:
        # Doğrulanacak bir iddia yokken "uyumlu" demek sahte uyum üretir.
        res.update(status="unverifiable", reason="Doğrulanacak ilgi iddiası yok")
    elif not looking:
        res.update(status="conflict", reason="Ekrana bakmıyor ama Gemini yüksek ilgi/niyet diyor")
    else:
        res.update(status="agree", reason="Gemini yüksek ilgi diyor ve izleyici ekrana bakıyor")
    return res


def _check_au_claims(gem: dict, pf: dict | None) -> dict:
    micro = (gem.get("commercial_signals") or {}).get("micro_expressions") or []
    claimed = sorted({f"AU{int(m)}" for s in micro for m in _AU_RE.findall(str(s))},
                     key=lambda a: int(a[2:]))
    if not claimed:
        return {"status": "unverifiable", "claimed": [], "reason": "AU iddiası yok"}
    if not pf or pf.get("aus") is None:
        return {"status": "conflict", "claimed": claimed, "unsupported": claimed,
                "reason": "Gemini AU iddia ediyor ama yüz/AU ölçümü yok (kanıtsız)"}
    aus = pf["aus"]
    unsupported = [a for a in claimed if aus.get(a, 0.0) < config.AU_PRESENCE_THRESHOLD]
    return {"status": "conflict" if unsupported else "agree", "claimed": claimed,
            "unsupported": unsupported, "measured": {a: aus.get(a) for a in claimed}}


def ad_segment_for(second, ad: dict | None) -> dict | None:
    """Saniyeyi içeren segmentlerden merkezi en yakın olanı (TSAM pencereleri 1 sn adımla çakışıyor)."""
    if not ad:
        return None
    cands = [s for s in ad.get("segments") or [] if isinstance(s, dict)
             and isinstance(s.get("start"), (int, float)) and isinstance(s.get("end"), (int, float))
             and s["start"] < second <= s["end"]]
    return min(cands, key=lambda s: abs((s["start"] + s["end"]) / 2 - second)) if cands else None


def _ad_insight(second, hse: dict | None, ad: dict | None) -> dict | None:
    """Reklamın beklenen duygusu vs ölçülen: çelişki değil, içgörü."""
    if not hse or not hse.get("face_found"):
        return None
    seg = ad_segment_for(second, ad)
    if seg:
        exp, got = seg.get("expected_emotion"), hse["dominant_emotion"]
        if exp and exp != got:
            return {"type": "expected_vs_measured", "expected": exp, "measured": got,
                    "segment": [seg.get("start"), seg.get("end")]}
    return None


def compute_agreement(gemini_analysis: list, face_metrics: dict, ad_analysis: dict | None = None) -> dict:
    hse = index_by_second(face_metrics.get("hsemotion"))
    mp = index_by_second(face_metrics.get("mediapipe"))

    per_second = []
    # Saniye kümesi = Gemini ∪ yüz ölçümleri (Gemini atlandıysa da tablo üretilir)
    gem_by_sec = {int(float(r.get("second", 0))): r for r in gemini_analysis}
    seconds = sorted(s for s in set(gem_by_sec) | set(hse) | set(mp) if s is not None)
    for sec in seconds:
        gem = gem_by_sec.get(sec, {})
        if not gem:
            # Gemini atlandı / bu saniye için veri yok: karşılaştırma yapılamaz
            no_gem = {"status": "unverifiable", "reason": "Gemini analizi yok"}
            checks = {k: dict(no_gem) for k in ("valence", "attention", "au_claims")}
        else:
            checks = {
                "valence": _check_valence(gem, hse.get(sec)),
                "attention": _check_attention(gem, mp.get(sec)),
                "au_claims": _check_au_claims(gem, mp.get(sec)),
            }
        evaluated = [c for c in checks.values() if c["status"] != "unverifiable"]
        conflicts = [k for k, c in checks.items() if c["status"] == "conflict"]
        insight = _ad_insight(sec, hse.get(sec), ad_analysis)
        per_second.append({
            "second": sec,
            "checks": checks,
            "conflicts": conflicts,
            "unverifiable": [k for k, c in checks.items() if c["status"] == "unverifiable"],
            "conflict_ratio": round(len(conflicts) / len(evaluated), 3) if evaluated else None,
            "insights": [insight] if insight else [],
        })

    n_conf = sum(len(p["conflicts"]) for p in per_second)
    return {
        "per_second": per_second,
        "summary": {
            "seconds": len(per_second),
            "total_conflicts": n_conf,
            "seconds_with_conflict": [p["second"] for p in per_second if p["conflicts"]],
            "thresholds": {"valence_diff": config.VALENCE_DIFF_THRESHOLD,
                           "looking_at_screen": config.LOOKING_AT_SCREEN_THRESHOLD,
                           "au_presence": config.AU_PRESENCE_THRESHOLD},
        },
    }
