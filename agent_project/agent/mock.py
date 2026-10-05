"""SENTETİK Gemini çıktısı. Gerçek API çağrısı DEĞİL; yalnızca anahtar yokken
pipeline'ı uçtan uca test etmek için (--mock-gemini). Çıktılarda source="mock" olarak işaretlenir."""


def _gem(sec, emo, val, aida, intent, micro=()):
    return {"second": sec,
            "affective_core": {"dominant_emotion": emo, "valence_score": val, "arousal_score": 0.0},
            "marketing_funnel": {"aida_stage": aida},
            "commercial_signals": {"purchase_intent": intent, "micro_expressions": list(micro)}}


SYNTHETIC_GEMINI = [
    _gem(2, "merak_ilgi", 0.5, "ilgi", "orta"),
    _gem(4, "noetral", 0.0, "farkindalik", "dusuk"),
    _gem(6, "hosnutsuzluk", -0.4, "terk_etme", "dusuk", ["AU4 kaş çatma", "AU15"]),
    _gem(8, "hayranlik", 0.6, "arzu", "yuksek"),
    _gem(10, "hayranlik", 0.8, "ilgi", "orta", ["gülümseme"]),
]


def mock_gemini_analysis(n_frames: int) -> list:
    """Kare sayısı kadar sentetik saniye döndür (fazlası kesilir)."""
    return [dict(r, _source="mock") for r in SYNTHETIC_GEMINI[:n_frames]]
