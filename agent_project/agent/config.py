"""Tek config dosyası: model yolları ve eşikler (Faz 2 eşikleri de buraya eklenecek)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# --- Model dosyaları (hepsi proje içindeki models/ altında, ~/.cache kullanılmaz) ---
FACE_DETECTOR_MODEL = MODELS_DIR / "blaze_face_short_range.tflite"
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
HSEMOTION_MODEL_NAME = "enet_b0_8_va_mtl"  # 8 duygu + valence + arousal veren model
HSEMOTION_ONNX = MODELS_DIR / f"{HSEMOTION_MODEL_NAME}.onnx"  # .pt'den tek seferlik dönüştürüldü (onnxruntime, torch yok)

# --- Yüz kırpma (HSEmotion girdisi) ---
FACE_DETECTION_MIN_CONF = 0.5
FACE_CROP_MARGIN = 0.10  # bbox her yönde %10 büyütülür

# --- MediaPipe "ekrana bakıyor mu" sezgisi ---
# Bakış = baş açısı + göz (iris) sapması. Skor = exp(-(yaw/σy)^2 - (pitch/σp)^2)
GAZE_SIGMA_YAW_DEG = 25.0
GAZE_SIGMA_PITCH_DEG = 20.0
# Göz blendshape (eyeLookIn/Out/Up/Down, 0-1) -> yaklaşık derece çevrimi
EYE_LOOK_TO_DEG = 30.0
LOOKING_AT_SCREEN_THRESHOLD = 0.5
EYE_CLOSED_BLINK_THRESHOLD = 0.5

# --- Faz 2: uyum hesaplayıcı (agent/agreement.py) ---
VALENCE_DIFF_THRESHOLD = 0.4          # |Gemini valence - HSEmotion valence| > bu ise çelişki
# Gemini'nin "yüksek ilgi/niyet" dediği değerler (dikkat çelişkisi için)
HIGH_INTEREST_PURCHASE_INTENT = {"cok_yuksek", "yuksek"}
HIGH_INTEREST_AIDA = {"ilgi", "arzu", "eyleme_gecis"}
HIGH_INTEREST_EMOTIONS = {"hayranlik", "merak_ilgi"}
AU_PRESENCE_THRESHOLD = 0.3           # MediaPipe blendshape AU vekili >= bu ise AU "var" sayılır
# Kare adında saniye yoksa (frame_XX değilse) sıraya göre atanacak saniyeler
DEFAULT_SECONDS = [2, 4, 6, 8, 10]

# --- Faz 3: graph ---
# GEMINI_MODEL ortam değişkeniyle değiştirilebilir. Orijinal main.py "gemini-3.6-flash" kullanıyor;
# adil kıyas için: GEMINI_MODEL=gemini-3.6-flash
GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"
MAX_ITERATIONS = 2                    # judge -> gemini_analyst geri dönüş sınırı
REANALYZE_CONFLICT_RATIO = 0.5        # ortalama conflict_ratio bunu aşarsa yeniden analiz iste
RESULTS_DIR = ROOT / "agent_results"
LOCAL_ONLY_MAX_CONF = 0.6             # Gemini ile hiçbir kontrol yapılamadıysa hakem güveni tavanı
# Bakma skoru (GAZE_* eşikleri) gerçek webcam kareleriyle kalibre edilince True yapılmalı
ATTENTION_CALIBRATED = False
UNCALIBRATED_ATTENTION_MAX_CONF = 0.5  # kalibrasyon yokken "dikkat düşüyor" önerilerinin güven tavanı

# --- Faz 4: shuffle deneyi (sıra etkisi kararı) ---
ORDER_EFFECT_RATIO = 2.0              # sıra etkisi, gürültü baz çizgisinin en az bu katı olmalı
ORDER_EFFECT_MIN_DELTA = 0.1          # ve baz çizgiden en az bu kadar (valence birimi) büyük olmalı
