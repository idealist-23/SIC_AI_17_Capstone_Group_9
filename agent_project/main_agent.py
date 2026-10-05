import base64
import glob
import json
import os
import webbrowser
from threading import Timer
import pandas as pd
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from PIL import Image

# ==========================================
# AJAN KATMANI (main.py'nin kopyası; tek fark /finalize_and_analyze)
# Gemini: yeni google-genai SDK, anahtar yalnızca GEMINI_API_KEY ortam değişkeninden.
# Anahtar yoksa graph Gemini'yi atlar, yerel ölçüm + kural tabanlı hakemle çalışır.
# ==========================================
from datetime import datetime
from threading import Lock, Thread

from agent import config as agent_config
from agent import gemini_client
from agent import graph as agent_graph


_ad_lock = Lock()
_ad_analysis = None


def _get_ad_analysis():
    # Reklam tarafı (TSAM, Faz 5): reklam başına bir kez hesaplanır, diskte önbelleklenir.
    # Model/ağırlık yoksa ya da hata olursa None döner; analiz reklam içgörüsü olmadan devam eder.
    global _ad_analysis
    with _ad_lock:
        if _ad_analysis is None:
            try:
                from agent.ad_analysis import analyze_ad
                _ad_analysis = analyze_ad(SAMPLE_VIDEO_PATH)
            except Exception as e:
                print(f"✗ Reklam analizi (TSAM) yapılamadı, içgörüsüz devam: {e}")
                _ad_analysis = False
        return _ad_analysis or None


def _preload_face_models():
    # Yüz modelleri ~10 sn'de yükleniyor; ilk analizi bekletmemek için arka planda önceden yükle.
    try:
        agent_graph._tools()
        print("✓ Yüz ölçüm modelleri yüklendi (HSEmotion-ONNX, MediaPipe).")
    except Exception as e:
        print(f"✗ Yüz modelleri yüklenemedi: {e}")
    if _get_ad_analysis():
        print("✓ Reklam analizi (TSAM) hazır.")

# ==========================================
# GÜVENLİ KLASÖR AYARLARI
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRAMES_FOLDER = os.path.join(BASE_DIR, "product_frames")
RESULTS_FOLDER = os.path.join(BASE_DIR, "product_results")
SAMPLE_VIDEO_PATH = os.path.join(BASE_DIR, "sample_ad.mp4")

CLICKS_JSON_PATH = os.path.join(RESULTS_FOLDER, "click_events.json")
CLICKS_CSV_PATH = os.path.join(RESULTS_FOLDER, "click_events.csv")

os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=BASE_DIR)

session_results = []
click_events = []

# TOPLU (BATCH) ÇIKARIM PROMPT'U
BATCH_UX_PROMPT = """
Sen kıdemli bir Tüketici Nöromarketing Araştırmacısı ve Büyüme Odaklı Pazarlama (Growth & Brand Marketing) Stratejistisisin.
Sana tüketicinin 10 saniyelik bir reklam/ürün videosunu izlerken 2 saniyede bir çekilmiş kamera kareleri sırasıyla verildi.
(Görseller sırasıyla: 2., 4., 6., 8. ve 10. saniyelerdir).

Kullanıcının yüz kaslarını (FACS), göz fiksasyonunu ve zaman içindeki mikro mimik değişim trendini Russell'ın Çevresel Duygu Modeli ile Tüketici Davranışı & Reklam Dönüşüm Hunisi (AIDA) prensiplerini harmanlayarak analiz et.

SADECE ve SADECE her bir kareye karşılık gelen nesneleri içeren, aşağıdaki şemaya uygun bir JSON DİZİSİ (Array) üret:

[
  {
    "second": 2,
    "affective_core": {
      "dominant_emotion": "hayranlik | merak_ilgi | noetral | kafa_karisikligi | suphe | can_sikintisi | hayal_kirikligi | hosnutsuzluk",
      "valence_score": -1.0 ile 1.0 arasında ondalıklı sayı,
      "arousal_score": -1.0 ile 1.0 arasında ondalıklı sayı
    },
    "marketing_funnel": {
      "aida_stage": "farkindalik | ilgi | arzu | eyleme_gecis | terk_etme",
      "brand_receptivity": "pozitif_baglanti | supheli_yaklasim | ilgisiz | tepkisel_reddetme",
      "cta_readiness": "hazir | kararsiz | erken | direncli"
    },
    "commercial_signals": {
      "purchase_intent": "cok_yuksek | yuksek | orta | dusuk | negatif",
      "perceived_value_friction": "Fiyat şüphesi | Güven sorunu | Karmaşık teklif | Yok",
      "micro_expressions": ["mimik1", "mimik2"]
    },
    "marketing_actionable_insight": {
      "scene_verdict": "Sahnenin kullanıcı zihninde yarattığı etki",
      "cro_recommendation": "Dönüşümü artırmak için 1 cümlelik pazarlama aksiyonu"
    }
  }
]
"""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video/<path:filename>")
def serve_video(filename):
    return send_from_directory(BASE_DIR, filename)


@app.route("/reset", methods=["POST"])
def reset():
    global session_results, click_events

    session_results = []
    click_events = []
    for folder in [FRAMES_FOLDER, RESULTS_FOLDER]:
        os.makedirs(folder, exist_ok=True)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except PermissionError:
                    pass
    print("✓ Oturum ve önceki test kareleri temizlendi.")
    return jsonify({"status": "cleared"})


# 2 saniyede bir gelen kareyi kaydeder
@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    try:
        data = request.get_json(force=True)
        second = int(data.get("second", 0))
        image_raw = data.get("image", "")

        if not image_raw or "," not in image_raw:
            return jsonify({"error": "Geçersiz görsel formatı"}), 400

        image_bytes = base64.b64decode(image_raw.split(",")[1])
        frame_path = os.path.join(FRAMES_FOLDER, f"frame_{second:02d}.jpg")
        with open(frame_path, "wb") as f:
            f.write(image_bytes)

        print(f"✓ [{second}. sn] Kare diske yazıldı -> {frame_path}")
        return jsonify({"status": "saved", "second": second})
    except Exception as e:
        print(f"✗ Kare kaydetme hatası: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/upload_click", methods=["POST"])
def upload_click():
    global click_events

    try:
        data = request.get_json(force=True)

        video_time = float(data.get("video_time", 0))
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
        x_norm = float(data.get("x_norm", 0))
        y_norm = float(data.get("y_norm", 0))

        click = {
            "video_time": round(video_time, 3),
            "x": round(x, 1),
            "y": round(y, 1),
            "x_norm": round(x_norm, 4),
            "y_norm": round(y_norm, 4)
        }

        click_events.append(click)

        # JSON olarak kaydet
        with open(CLICKS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(click_events, f, ensure_ascii=False, indent=2)

        # CSV olarak da kaydet
        pd.DataFrame(click_events).to_csv(
            CLICKS_CSV_PATH,
            index=False,
            encoding="utf-8"
        )

        print(
            f"✓ Tıklama kaydedildi | "
            f"Video: {video_time:.2f}s | "
            f"X: {x:.1f} | Y: {y:.1f}"
        )

        return jsonify({
            "status": "saved",
            "click": click
        })

    except Exception as e:
        print(f"✗ Tıklama kaydetme hatası: {e}")
        return jsonify({"error": str(e)}), 500

def merge_clicks_with_emotions(results, clicks):
    """
    Click verilerini emotion sonuçlarıyla zaman aralığına göre eşleştirir.

    0-2 saniye  -> 2. saniye emotion
    2-4 saniye  -> 4. saniye emotion
    4-6 saniye  -> 6. saniye emotion
    6-8 saniye  -> 8. saniye emotion
    8-10 saniye -> 10. saniye emotion
    """

    for result in results:
        emotion_second = float(result.get("second", 0))

        matching_clicks = []

        for click in clicks:
            click_time = float(click.get("video_time", 0))

            if (emotion_second - 2) < click_time <= emotion_second:
                matching_clicks.append(click)

        result["interaction"] = {
            "click_count": len(matching_clicks),
            "clicks": matching_clicks
        }

    return results

def _results_from_local(final):
    """Gemini atlandıysa arayüz boş kalmasın: hakem kararlarından aynı şekilde öğeler üret."""
    items = []
    for v in (final.get("verdicts") or {}).get("verdicts") or []:
        if not isinstance(v, dict) or v.get("second") is None:
            continue
        items.append({
            "second": v.get("second"),
            "source": "local",
            "affective_core": {
                # Ölçülmeyen değerler uydurulmaz (boş kalır); index.html null'u "?? 0" ile gösteriyor.
                # Duygu etiketi arayüzde "Nötr"e düşmesin diye açıkça "olculemedi".
                "dominant_emotion": v.get("emotion") or "olculemedi",
                "valence_score": v.get("valence"),
                "arousal_score": None,
            },
        })
    return items


# Test bittiğinde kareleri AJAN GRAPH'INA gönderir (yerel ölçüm + Gemini + hakem + öneri)
@app.route("/finalize_and_analyze", methods=["POST"])
def finalize_and_analyze():
    global session_results
    session_results = []

    saved_frames = sorted(
        glob.glob(os.path.join(FRAMES_FOLDER, "frame_*.jpg"))
    )

    if not saved_frames:
        return jsonify({"error": "Hiçbir kare yakalanamadı"}), 400

    mode = "real" if gemini_client.has_key() else "off"
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = agent_config.RESULTS_DIR / session_id
    print(f"\n--> {len(saved_frames)} kare ajan graph'ına gönderiliyor (gemini_mode={mode})...")

    try:
        final = agent_graph.run(saved_frames, session_id, out_dir, mode, list(click_events), _get_ad_analysis())
    except Exception as e:
        print(f"✗ Ajan analizi hatası: {e}")
        return jsonify({"error": str(e), "clicks": click_events}), 500

    gemini_results = final.get("gemini_analysis") or []
    if isinstance(gemini_results, dict):
        gemini_results = [gemini_results]
    session_results = [dict(r) for r in gemini_results] or _results_from_local(final)

    # Her saniyeye doğrulanmış hakem kararını ekle
    verdicts = {v.get("second"): v for v in (final.get("verdicts") or {}).get("verdicts") or []
                if isinstance(v, dict)}
    for r in session_results:
        try:
            r["agent_verdict"] = verdicts.get(int(float(r.get("second", 0))))
        except (TypeError, ValueError):
            r["agent_verdict"] = None

    session_results = merge_clicks_with_emotions(session_results, click_events)

    with open(os.path.join(RESULTS_FOLDER, "product_ux_results.json"), "w", encoding="utf-8") as f:
        json.dump(session_results, f, ensure_ascii=False, indent=2, default=str)

    # Orijinal main.py ile aynı düzleştirme + agent_verdict sütunları
    # ==========================================
    # EMOTION + CLICK CSV
    # ==========================================

    flattened = []

    for r in session_results:

        interaction = r.get("interaction", {})
        clicks = interaction.get("clicks", [])

        # Click koordinatlarını tek hücrede sakla
        click_coordinates = "; ".join(
            [
                f"({c.get('x')}, {c.get('y')})"
                for c in clicks
            ]
        )

        flattened.append({
            # Zaman
            "second": r.get("second"),

            # Emotion
            "dominant_emotion": r.get(
                "affective_core", {}
            ).get("dominant_emotion"),

            "valence": r.get(
                "affective_core", {}
            ).get("valence_score"),

            "arousal": r.get(
                "affective_core", {}
            ).get("arousal_score"),

            # Marketing
            "aida_stage": r.get(
                "marketing_funnel", {}
            ).get("aida_stage"),

            "brand_receptivity": r.get(
                "marketing_funnel", {}
            ).get("brand_receptivity"),

            "purchase_intent": r.get(
                "commercial_signals", {}
            ).get("purchase_intent"),

            "friction": r.get(
                "commercial_signals", {}
            ).get("perceived_value_friction"),

            "cta_readiness": r.get(
                "marketing_funnel", {}
            ).get("cta_readiness"),

            # Insight
            "scene_verdict": r.get(
                "marketing_actionable_insight", {}
            ).get("scene_verdict"),

            "cro_recommendation": r.get(
                "marketing_actionable_insight", {}
            ).get("cro_recommendation"),

            # Click / Interaction
            "click_count": interaction.get(
                "click_count", 0
            ),

            "click_coordinates": click_coordinates,

            # Ajan katmanı (ek sütunlar)
            "agent_confidence": (r.get("agent_verdict") or {}).get("confidence"),

            "agent_trusted_source": (r.get("agent_verdict") or {}).get("trusted_source"),
        })

    # CSV dosyasını oluştur
    csv_path = os.path.join(
        RESULTS_FOLDER,
        "product_ux_results.csv"
    )

    pd.DataFrame(flattened).to_csv(
        csv_path,
        index=False,
        encoding="utf-8"
    )

    print(
        f"✓ Raporlar kaydedildi: {csv_path}"
    )

    print(f"✓ Ajan analizi bitti. Rapor: {out_dir}")
    return jsonify({
        "status": "completed",
        "results": session_results,
        "clicks": click_events,
        "agent": {
            "session_id": session_id,
            "gemini_meta": final.get("gemini_meta"),
            "iteration": final.get("iteration"),
            "agreement_summary": (final.get("agreement") or {}).get("summary"),
            "verdicts": final.get("verdicts"),
            "recommendations": final.get("recommendations"),
            "report_dir": str(out_dir),
            "timings": final.get("timings"),
        },
    })

if __name__ == "__main__":
    Thread(target=_preload_face_models, daemon=True).start()
    Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False, port=5000)