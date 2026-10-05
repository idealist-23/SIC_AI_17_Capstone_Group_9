"""main_agent.py uçtan uca testi (Flask test client; sunucu/port açılmaz, ağ yok).

Kullanım: python -m agent.test_flask
Akış: reset -> 5 kare upload_frame -> 1 tıklama -> finalize_and_analyze -> JSON/CSV kontrol -> reset
Not: product_frames/ ve product_results/ klasörlerine yazar (main.py ile aynı davranış), sonunda reset temizler.
"""
import base64
import io
import os

import pandas as pd
from PIL import Image

from agent import config

FRAMES = ["ad_2s.jpg", "ad_6s.jpg", "business-person.png", "business-person_flip.jpg", "portrait.jpg"]
ORIG_CSV_COLS = ["second", "dominant_emotion", "valence", "arousal", "aida_stage", "brand_receptivity",
                 "purchase_intent", "friction", "cta_readiness", "scene_verdict", "cro_recommendation",
                 "click_count", "click_coordinates"]


def main():
    import main_agent as m
    c = m.app.test_client()
    assert c.post("/reset").json == {"status": "cleared"}
    for sec, f in zip([2, 4, 6, 8, 10], FRAMES):
        buf = io.BytesIO()
        Image.open(config.ROOT / "test_frames_tmp" / f).convert("RGB").save(buf, "JPEG")
        img = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        assert c.post("/upload_frame", json={"second": sec, "image": img}).status_code == 200
    c.post("/upload_click", json={"video_time": 7.5, "x": 100, "y": 50, "x_norm": .2, "y_norm": .1})

    r = c.post("/finalize_and_analyze")
    d = r.json
    assert r.status_code == 200 and d["status"] == "completed", d
    csv_path = os.path.join(m.RESULTS_FOLDER, "product_ux_results.csv")
    assert os.path.isfile(csv_path), "product_ux_results.csv oluşmadı"
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ORIG_CSV_COLS + ["agent_confidence", "agent_trusted_source"], list(df.columns)
    assert len(df) == len(d["results"]) and int(df.loc[df["second"] == 8, "click_count"].iloc[0]) == 1

    print("status:", r.status_code, d["status"], "| gemini:", (d["agent"]["gemini_meta"] or {}).get("status"))
    print("CSV:", csv_path, df.shape)
    print(df[["second", "dominant_emotion", "valence", "aida_stage", "click_count",
              "agent_confidence", "agent_trusted_source"]].to_string(index=False))
    for x in d["agent"]["recommendations"]["recommendations"]:
        print("  REC", x["second"], x["confidence"], x["reason"][:80])
    assert c.post("/reset").json == {"status": "cleared"}
    assert not os.path.exists(csv_path)
    print("OK: finalize 200, CSV orijinal sütunlar + agent_confidence/agent_trusted_source, reset temizledi")


if __name__ == "__main__":
    main()
