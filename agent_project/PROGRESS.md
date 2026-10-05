# PROGRESS

## Oturum 1 (2026-09-29, ~20 dk): Faz 0 + Faz 1 (HSEmotion, MediaPipe)

Kapsam bilerek dar tutuldu: py-feat bu oturumda **kurulmadı** (istek üzerine).
`main (1).py` ve `index.html`'e dokunulmadı. Push yok. Commit de atılmadı (yalnızca repo + dal açıldı).

### Faz 0: Kurulum ✅
- **uv sistemde yoktu.** Global Python'a paket kurmamak için uv, proje içindeki bir araç
  venv'ine kuruldu: `.tools/` (`python -m venv .tools && .tools/Scripts/pip install uv`).
  Kullanım: `.tools/Scripts/uv pip install --python .venv <paket>`.
  (İsterseniz uv'yi resmi kurucuyla global kurup `.tools/`'u silebilirsiniz.)
- `.venv`: `uv venv --python 3.12`. uv, taban yorumlayıcı olarak `anaconda3\python.exe` (3.12.7)
  seçti; venv izole, anaconda'ya paket kurulmadı. 3.12 seçildi, çünkü mediapipe/py-feat için 3.13'ten daha güvenli.
- PyTorch: `--index-url https://download.pytorch.org/whl/cu128`
- Doğrulama: `torch 2.11.0+cu128, CUDA 12.8, cuda.is_available()=True, NVIDIA GeForce RTX 4060 Ti`
  (sürücü 591.86 / CUDA 13.1).
- git: `git init`, dal `agent-layer`. `.gitignore`: `.venv/ .tools/ .env __pycache__ agent_results/ models/`.

| Paket | Sürüm |
|---|---|
| Python | 3.12.7 |
| uv | 0.12.19 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| hsemotion | 0.3.0 |
| timm | 1.0.30 |
| mediapipe | 1.0.1 |
| opencv-python (+contrib) | 5.0.0.93 |
| numpy | 2.5.2 |
| flask / pandas / pillow | 3.1.3 / 3.0.6 / 12.3.0 |

Disk: `.venv` ≈ 4.6 GB (klasör OneDrive altında, senkronizasyonu yavaşlatabilir; bkz. açık konular).

### Model dosyaları (`models/`, hepsi proje içinde)
- `enet_b0_8_va_mtl.pt`: HSEmotion, 8 duygu + valence + arousal (VA veren model).
- `face_landmarker.task`: MediaPipe Face Landmarker (float16, blendshape + transformation matrix).
- `blaze_face_short_range.tflite`: MediaPipe yüz dedektörü (HSEmotion için yüz kırpma).

### Faz 1: Yazılan kod
- `agent/config.py`: model yolları ve eşikler (Faz 2 eşikleri de buraya eklenecek).
- `agent/tools/common.py`: kare listeleme, `frame_{ss}.jpg` adından saniye çıkarma (main.py formatı),
  PIL ile yükleme (cv2.imread, "Masaüstü" gibi ASCII dışı yollarda Windows'ta başarısız).
- `agent/tools/hsemotion_tool.py`: `analyze_frames(frames) -> {"tool","model","device","frames":[...]}`.
  Kare başına: `second, frame, face_found, face_box, dominant_emotion, emotions{8}, valence, arousal, timing_ms`.
- `agent/tools/mediapipe_tool.py`: `analyze_frames(frames)`. Kare başına: `head_pose{pitch,yaw,roll}`,
  `eye_gaze_deg`, `iris_ratio`, `blink{left,right,eyes_closed}`, `looking_at_screen_score (0-1)`,
  `looking_at_screen`, `score_parts`, göz blendshape'leri. Yüz yoksa `face_found=False`,
  değerler `None` ve not: "Yüz bulunamadı: dikkat/bakış ölçülemedi (0 değil, bilinmiyor)".
- `agent/tools/test_tools.py`: `python -m agent.tools.test_tools <klasör> --repeat 5`, JSON'ları
  `agent_results/phase1/` altına yazar.

### Karşılaşılan sorunlar ve çözümler
1. **HSEmotion + torch 2.11:** `torch.load` varsayılanı `weights_only=True`; kütüphane tam model
   pickle'ı yüklüyor. Ayrıca ağırlıkları `~/.hsemotion`'a (proje dışı) indiriyor.
   → `HSEmotionRecognizer` kullanılmadı, aynı ön işleme ve sınıflandırıcı mantığı (hsemotion 0.3.0
   `facial_emotions.py`) kendi sınıfımızda tekrarlandı, model `models/`'tan yükleniyor.
2. **Eski timm pickle'ı + timm 1.0.30:** forward'da `'DepthwiseSeparableConv' has no attribute
   'conv_s2d'` (ardından `aa`). Alan alan yama denendi, bırakıldı.
   → Mimari pickle'daki `pretrained_cfg`'den okunuyor (`tf_efficientnet_b0`, 10 çıkış), güncel
   timm ile yeniden kuruluyor, `state_dict` **strict** yükleniyor (0 eksik / 0 fazla anahtar).
3. Test verisi: `test_frames/` ve `Student-engagement-dataset/` klasörde **yok**. Geçici olarak
   `test_frames_tmp/` oluşturuldu (5 kare):
   - `business-person.png`, `portrait.jpg`: MediaPipe'ın resmi örnek görselleri (storage.googleapis.com/mediapipe-assets)
   - `business-person_flip.jpg`: yatay aynalanmış kopya (baş pozu işaret testi)
   - `ad_2s.jpg`, `ad_6s.jpg`: `sample_ad.mp4`'ten kareler (yüz yok → "yüz bulunamadı" yolu testi)

### Sonuçlar (test_frames_tmp)
| Kare | HSEmotion | V / A | MediaPipe baş pozu (p/y/r) | Ekrana bakma |
|---|---|---|---|---|
| ad_2s / ad_6s | yüz yok | – | yüz yok | None (bilinmiyor) |
| business-person | Contempt | 0.66 / -0.09 | 18.1 / 4.9 / 5.2 | 0.37 |
| business-person_flip | Contempt | 0.51 / -0.05 | 17.8 / -5.2 / -5.2 | 0.32 |
| portrait | Happiness | 0.86 / 0.20 | 3.9 / 1.8 / 0.9 | 0.94 |

Tutarlılık kontrolü: aynalanmış karede yaw ve roll işaret değiştiriyor, pitch aynı kalıyor ✔.
Not: aynalamada HSEmotion valence değeri 0.66'dan 0.51'e düşüyor. Model aynalamaya karşı tam değişmez değil.

### Süre ölçümleri (kare başına, medyan, RTX 4060 Ti)
| Tool | Model yükleme | İlk yüzlü kare (CUDA ısınma) | Sıcak, yüzlü kare | Sıcak, yüzsüz kare | Cihaz |
|---|---|---|---|---|---|
| HSEmotion (dedektör + EfficientNet-B0) | 0.83 s | 275 ms | **32.9 ms** (18 ms tespit dahil) | 9.4 ms | CUDA (dedektör CPU) |
| MediaPipe Face Landmarker | 0.03 s | ~9 ms | **22.1 ms** | 8.3 ms | CPU (XNNPACK) |

Süreler görüntü yüklemeyi de içeriyor (1 MB PNG / JPG çözme ≈ 13 ms/kare). 5 karelik bir oturum için
iki tool toplamda < 0.5 s (ısınma dahil ~1.5 s). Darboğaz değil.

### Açık kalanlar / sonraki oturum
- **Gerçek test verisi gerekli:** `test_frames/` (webcam kareleri) ya da Kaggle setini klasöre koyun.
  Şu anki kareler stok portre, webcam koşullarını temsil etmiyor.
- **"Ekrana bakıyor mu" skoru sezgisel ve kalibre edilmedi.** Baş (σ yaw 25°, σ pitch 20°) ve göz
  blendshape sapması (×30°) ayrı Gaussian'larla çarpılıyor. Göz ve baş açısının işaret konvansiyonu
  doğrulanmadığı için birbirini telafi etmesine izin verilmiyor (muhafazakâr). Ör. business-person
  kamerayı hafif aşağıdan görüyor (pitch 18°), bu yüzden skor 0.37 çıkıyor. Gerçek webcam kareleriyle
  eşikler `agent/config.py`'de ayarlanmalı. Pitch'te kamera yüksekliği ofseti gerekebilir.
- MediaPipe Python Tasks API Windows'ta GPU delegate desteklemiyor, CPU'da çalışıyor (süre yeterli).
- HSEmotion yüz kırpması BlazeFace short-range kullanıyor (~2 m'ye kadar). Webcam için yeterli.
- Faz 1.2 py-feat: sonraki bölümde kuruldu (kullanıcı onayıyla).
- Faz 3–6 başlanmadı (Faz 2 ek süre bölümünde). Tool çıktıları şimdilik liste (`frames: [...]`, her öğede `second`).
  Faz 3'te state'e saniye anahtarlı dict'e çevrilebilir.
- OneDrive: `.venv` (4.6 GB) OneDrive senkronunda. İsterseniz klasörü "Bu cihazda her zaman tut"
  yapın veya `.venv`'i senkron dışı bırakın.
- git: yalnızca `git init` + `agent-layer` dalı. Henüz commit yok.

## Oturum 1, ek 10 dk: Faz 2 (uyum hesaplayıcı) ✅

- `agent/agreement.py`: `compute_agreement(gemini_analysis, face_metrics, ad_analysis=None)`. LLM yok.
  Girdi Gemini şeması `main (1).py`'deki `BATCH_UX_PROMPT` ile aynı (salt okundu, değiştirilmedi).
  Her saniye için 3 kontrol yapılıyor. Her biri `agree | conflict | unverifiable` + gerekçe/değerler döndürüyor:
  1. **valence**: |Gemini `valence_score` − HSEmotion valence| > 0.4 → çelişki.
  2. **attention**: MediaPipe "ekrana bakmıyor" ama Gemini yüksek ilgi/niyet diyor
     (`purchase_intent` ∈ {cok_yuksek, yuksek} veya `aida_stage` ∈ {ilgi, arzu, eyleme_gecis} veya
     duygu ∈ {hayranlik, merak_ilgi}) → çelişki. **Ek kural:** yüz hiç bulunamadığı hâlde Gemini
     yüksek ilgi diyorsa bu da çelişki. Bulunamadıysa ve iddia yoksa `unverifiable` döner.
  3. **au_claims**: `micro_expressions` içinde `AU\d+` geçiyor, py-feat yok ya da AU < 0.5 → "kanıtsız AU" çelişkisi.
     py-feat beklenen biçim: `frames[i]["aus"] = {"AU12": 0.8, ...}` (py-feat tool'u yazılırken buna uyulmalı).
  4. Reklam modeli varsa beklenen ve ölçülen duygu farkı `insights` altında raporlanıyor (çelişki sayılmıyor).
  - Saniye başına `conflict_ratio` = çelişki / (ölçülebilen kontrol sayısı). `unverifiable` paydaya girmiyor.
  - Tüm eşikler `agent/config.py`'de ("Faz 2" bölümü).
  - Kare adında saniye yoksa (`frame_XX` değilse) kareler sırayla 2, 4, 6, 8, 10'a atanıyor (`DEFAULT_SECONDS`).
- `agent/test_agreement.py`: Faz 1 JSON'ları + **sentetik** Gemini çıktısıyla (gerçek API çağrısı değil)
  5 senaryo. Hepsi beklenen sonucu verdi (assert geçti). Sonuç: `agent_results/phase1/agreement_synthetic.json`.

| sn | Senaryo | Sonuç |
|---|---|---|
| 2 | yüz yok + Gemini "ilgi" | attention conflict |
| 4 | yüz yok + Gemini nötr | çelişki yok, 2 kontrol unverifiable |
| 6 | valence −0.4 vs 0.66 + AU4/AU15 iddiası (py-feat yok) | valence + au_claims conflict |
| 8 | bakma skoru 0.32 + Gemini "arzu/yuksek" | attention conflict |
| 10 | uyumlu | çelişki yok |

### Faz 2 açık noktalar
- Gemini duygu etiketleri Türkçe ve farklı bir küme (hayranlik, suphe…), HSEmotion'unkiler ise 8 AffectNet
  sınıfı. Etiket düzeyinde karşılaştırma yapılmıyor, yalnızca valence karşılaştırılıyor. Gerekirse eşleme tablosu eklenebilir.
- Reklam içgörüsü HSEmotion'un İngilizce etiketiyle karşılaştırıyor. TSAM çıktısının etiket kümesi (Faz 5) netleşince uyarlanmalı.
- py-feat yokken her AU iddiası "kanıtsız" sayılıyor (plana uygun). py-feat gelince aynı fonksiyon ölçümle doğrular.

## Oturum 1, gece devamı: py-feat (Faz 1.2) ✅ + Faz 3 hazırlığı

Kullanıcı py-feat kurulumuna izin verdi. Ardından bilgisayar fan sesi yüzünden çalışmayı durdurmamızı istedi.
Faz 1.2 tamamlandı; Faz 3'e yalnızca API doğrulamasıyla başlandı, **kod yazılmadı**.

### Kurulumlar
- `py-feat 2.1.3`: torch'un ezilmemesi için `.tools/torch-constraints.txt` (torch==2.11.0+cu128,
  torchvision==0.26.0+cu128) ile, önce `--dry-run` yapılarak kuruldu. Kurulumdan sonra `torch.cuda.is_available()=True`.
  Beraberinde gelen başlıca paketler: xgboost 3.4.1, scikit-learn 1.9.1, scipy 1.18.1, scikit-image 0.26.0.
- `langgraph 1.2.12` (langchain-core 1.6.5), `google-genai 2.25.0` (Faz 3 için).
- py-feat ağırlıkları `.venv/Lib/site-packages/feat/resources` altına indi (proje içinde). İlk yükleme,
  indirme dahil 67 s sürdü; sonraki yüklemeler ~5 s.

### `agent/tools/pyfeat_tool.py`
- `Detectorv1(emotion_model=None, identity_model=None)`: retinaface + mobilefacenet + **xgb AU** + pose_mlp + L2CS bakış.
  Duygu modeli HSEmotion'da olduğu için, kimlik modeli de gereksiz olduğu için kapalı.
- Kare başına çıktılar:
  - `aus`: `{"AU1":…, "AU4":…}` biçiminde, 0-1 "var" olasılığı. Anahtarlar `agreement.py`'nin beklediği biçimde.
  - `head_pose` ve `gaze`: derece cinsinden (py-feat radyan döndürüyor, tool dereceye çeviriyor).
  - Yüz yoksa: `face_found=False` ve "Yüz bulunamadı" notu.
- `test_tools.py` artık üç tool'u birlikte ölçüyor.

| Tool | Model yükleme | İlk kare | Sıcak kare (medyan) |
|---|---|---|---|
| HSEmotion | 1.1 s | 14 ms (yüzsüz kare) | 30 ms |
| MediaPipe | 0.03 s | 12 ms | 21 ms |
| **py-feat** | 5.0 s | 330 ms | **112 ms** (tek kare/çağrı, GPU) |

### py-feat bulguları / açık konular
- **Baş pozu işaretleri MediaPipe ile uyuşmuyor.** business-person'da MediaPipe pitch +18°, py-feat −11°;
  yaw da ters işaretli. İkisi de aynalamada yaw işaretini doğru çeviriyor. py-feat'in kendi konvansiyonu:
  +pitch yukarı, +yaw kişinin sağı. Gerçek webcam kareleriyle (bilinen bakış yönleriyle) kalibre etmeden
  iki tool'un pozu birleştirilmemeli.
- **xgb AU doyması:** bazı AU'lar tam 1.0 ya da 0.0 çıkıyor (AU7 her yüzde 1.0; AU11 ve AU20 0 ile 1 arasında zıplıyor).
  Yüklemede XGBoost "eski sürümle serileştirilmiş model" uyarısı veriyor.
  `svm` AU modeliyle karşılaştırıldı: svm tamamen ikili (0/1), ama portrede iki model de AU6 + AU12 + AU25'te
  (Duchenne gülümseme) uzlaşıyor. xgb'de kalındı. AU7 tek başına kanıt sayılmamalı.
- Alternatif: py-feat `Detectorv2` (multitask; tek geçişte 24 AU + VA + bakış + poz) denenmedi.

### Faz 3 için doğrulanmış API'ler (kod henüz yok)
- LangGraph 1.2: `StateGraph(state_schema)`, `add_node(name, fn)`, `add_edge(start_key: str | list[str], end)`,
  `add_conditional_edges(source, path_fn, path_map)`, `START`/`END` (`langgraph.graph`).
  Paralellik için `START`'tan `face_measure` ve `gemini_analyst`'e iki kenar, `add_edge([...], "agreement")` ile birleşme.
- google-genai 2.25: `genai.Client(api_key=...)`,
  `client.models.generate_content(model=, contents=[...], config=types.GenerateContentConfig(...))`,
  `types.Part.from_bytes(data=, mime_type=)`. Config alanları: `response_mime_type`, `response_json_schema`,
  `temperature`, `seed`, `system_instruction`, `thinking_config`.
- **`GEMINI_API_KEY` tanımlı değil** (ne oturumda ne kullanıcı düzeyinde). Faz 3'te Gemini düğümleri,
  anahtar yokken "atlandı" diye işaretlenecek ve test için etiketli bir `--mock-gemini` modu olacak.
  Faz 4 deneyleri anahtar olmadan çalıştırılamaz.

### Sabah kaldığımız yer
1. Faz 3: `agent/graph.py` + `agent/run.py` (face_measure ∥ gemini_analyst → agreement → judge → recommender → reporter).
2. `GEMINI_API_KEY` ortam değişkeni tanımlanırsa Faz 4 deneyleri.
3. Gerçek `test_frames/` konursa: bakma skoru ve poz kalibrasyonu.
4. git: repo ve `agent-layer` dalı var, **henüz commit yok**.

## Oturum 2 (2026-09-29, ~08:45–08:55; dosya zaman damgalarına göre): Faz 3 (LangGraph) ✅, Gemini'siz doğrulandı

Durum değişmedi: `GEMINI_API_KEY` yok, `test_frames/` yok, bu yüzden testler `test_frames_tmp/` üzerinde yapıldı.

### Yeni dosyalar
- `agent/gemini_client.py`: google-genai istemcisi. Anahtar yalnızca env'den okunur.
  `BATCH_UX_PROMPT`, `main (1).py` / `main.py` içinden **ast ile salt-okunur** çekiliyor (dosya import
  edilmiyor, kopya da yok; tek kaynak orijinal dosya). JSON yanıtı ```json bloklarından da ayrıştırılıyor.
- `agent/judging.py`: hakem ve öneri prompt'ları. Her biri için **kural tabanlı yedek** var
  (anahtar yoksa ya da LLM hata verirse devreye giriyor). Çıktıdaki `source` alanı hangisinin kullanıldığını söyler:
  `gemini` ya da `rules`. LLM'e ham landmark değil, kompakt saniye özeti gidiyor.
- `agent/graph.py`: `START → {face_measure ∥ gemini_analyst} → agreement → judge → (geri dön | recommender) → reporter`.
  - Hakem geri dönüşü yalnızca gerçek Gemini modunda, `request_reanalysis=True` ve `iteration < 2` ise olur.
  - Yüz tool'ları süreç başına bir kez yüklenip önbellekte tutuluyor (Flask entegrasyonu için hazır).
  - Bir tool hata verirse diğerleri devam ediyor; hata `face_metrics.errors` altına yazılıyor.
  - `timings` alanı paralel dallardan gelen süreleri `operator.add` reducer'ıyla biriktiriyor.
- `agent/reporter.py`: `result.json`, `per_second.csv` (utf-8-sig, Excel uyumlu) ve `chart.png` üretir.
  Grafikte üç panel var: kaynağa göre valence, ekrana bakma, güven + çelişki etiketleri. Mock/atlandı
  durumu grafik lejantında belirtiliyor.
- `agent/run.py`: `python -m agent.run --frames <klasör> [--gemini real|mock|off] [--mock-gemini] [--clicks csv] [--ad-analysis json] [--session-id]`.
  Varsayılan mod: anahtar varsa `real`, yoksa `off`. Çıktı `agent_results/<session_id>/` altına yazılıyor.
- `agent/mock.py`: sentetik Gemini verisi. `test_agreement.py` ile `--mock-gemini` bunu ortak kullanıyor.
- `agent/agreement.py`: saniye kümesi artık Gemini ∪ yüz ölçümleri. Gemini atlansa da tablo üretiliyor.

### Testler
| Test | Sonuç |
|---|---|
| `python -m agent.run --frames test_frames_tmp` (anahtar yok, `off`) | ✅ hatasız, JSON + CSV + grafik |
| `--mock-gemini` | ✅ çelişkiler 2 / 6 / 8. sn'de. py-feat artık devrede olduğu için 6. sn'deki AU4 iddiası gerçek ölçümle "desteklenmiyor" çıktı; AU15 destekleniyor |
| Hakem döngüsü (sahte `generate_json`, ağ yok; test scratchpad'de) | ✅ analist → hakem → analist (geri bildirimli) → hakem → öneri; `iteration=2`'de duruyor, `face_measure` bir kez çalışıyor |
| `python -m agent.test_agreement` | ✅ |

### Düğüm süreleri (5 kare, soğuk başlangıç, mock Gemini)
| Düğüm | Süre |
|---|---|
| face_measure | 11.6 s = **9.7 s model yükleme** + hsemotion 0.30 s + mediapipe 0.09 s + pyfeat 1.45 s |
| gemini_analyst (mock) | ~0 ms (gerçek çağrı ölçülmedi) |
| agreement / judge (kurallar) / recommender (kurallar) | < 1 ms |
| reporter | 0.2–0.3 s (matplotlib) |
| **Toplam** | **~11.8 s** (sıcak süreçte yükleme olmadığı için ~2 s beklenir) |

### Açık kalanlar
1. **Gerçek Gemini ile hiç çalıştırılmadı.** Hakem ve öneri prompt'ları, `generate_json` ve JSON ayrıştırma gerçek
   yanıtla test edilmeli. Anahtar tanımlanınca: `python -m agent.run --frames test_frames_tmp --gemini real`.
2. Faz 4 deneyleri: script hazır (`agent/experiments.py`, `same` ve `shuffle`); karşılaştırma mantığı sahte veriyle doğrulandı. **Gerçek çalıştırma anahtar bekliyor.** Komutlar dosyanın başında.
3. `face_measure` ile `gemini_analyst`'in gerçekten eşzamanlı çalıştığı, gerçek Gemini gecikmesiyle ölçülmeli.
4. Faz 5 (TSAM) başlanmadı. Faz 6 → Oturum 3.
5. git: hâlâ commit yok (`agent-layer` dalı, `.gitignore` hazır).

## Oturum 3 (2026-09-29, ~08:55–09:00): Faz 6 (Flask entegrasyonu) ✅ + yanlış pozitif düzeltmesi

### `main_agent.py`
- `main (1).py`'den **programatik olarak** kopyalandı; yalnızca üç bölüm değişti, orijinal dosyaya dokunulmadı:
  1. Eski `google.generativeai` import'u ve `API_KEY`/`genai.configure` bloğu kaldırıldı (bu SDK `.venv`'de yok).
     Yerine `agent.gemini_client` kullanılıyor (yeni google-genai SDK, anahtar yalnızca env'den).
  2. `/finalize_and_analyze` artık `agent.graph.run(...)` çağırıyor (`gemini_mode`: anahtar varsa `real`, yoksa `off`).
     **Frontend uyumu:** `results` aynı biçimde (`second`, `affective_core.*`), `merge_clicks_with_emotions` aynen
     kullanılıyor. Her öğeye `agent_verdict` (hakem kararı + güven) ekleniyor. Gemini atlandıysa `results`
     yerel ölçümden `source: "local"` etiketiyle üretiliyor, böylece arayüz boş kalmıyor. Yanıta ek olarak
     `agent` alanı geliyor: `gemini_meta`, `iteration`, uyum özeti, kararlar, öneriler, `report_dir`, süreler.
     Rapor dosyaları `agent_results/<zaman damgası>/` altına yazılıyor.
  3. Uygulama açılışında yüz modelleri arka plan thread'inde önceden yükleniyor (~10 s), ilk analiz beklemesin diye.
- Diğer route'lar (`/`, `/video`, `/reset`, `/upload_frame`, `/upload_click`) ve port 5000 aynı.
  **Not:** orijinal dosya `gemini-3.6-flash` modelini kullanıyor; ajan katmanı plana uygun olarak `gemini-2.5-flash` kullanıyor.
- **Test (Flask test client, sunucu açılmadan):** reset → 5 kare `upload_frame` (base64 JPEG) → tıklama (7.5 s) →
  finalize → reset. Sonuç: 200 `completed`; tıklama 8. saniyeye eşlendi; öneriler üretildi; reset klasörleri temizledi ✅.
  Çalıştırmak için: `.venv/Scripts/python main_agent.py`

### Bulunan ve düzeltilen hata: py-feat yanlış pozitif
- Flask testinde reklam karesi (yüz yok) JPEG'e yeniden kodlanınca py-feat/retinaface **0.58 skorla "yüz"** buldu
  (AU11 = 1.0, yaw 35°). Kural tabanlı hakem bunu "uyumlu" sayıp güveni **0.7** verdi, oysa ölçülen hiçbir şey yoktu.
- Düzeltmeler:
  - `config.PYFEAT_MIN_FACE_SCORE = 0.8`: altı yüz sayılmıyor, skor yine raporlanıyor.
  - Hakemde yüz varlığı artık HSEmotion/MediaPipe'a göre belirleniyor. py-feat yalnız kalırsa `face_found_by`
    ile bildiriliyor ve gerekçeye "tutarsız, yok sayıldı" notu düşülüyor.
- Doğrulama: aynı karelerle 2. sn güveni 0.7'den 0.1'e ("yüz ölçülemedi") düştü; yüzlü kareler (skor ≥ 0.9999) etkilenmedi.
  `test_agreement` hâlâ geçiyor ✅.

### Açık kalanlar (güncel)
1. **Gerçek Gemini ile hiç çalıştırılmadı.** `GEMINI_API_KEY` tanımlanınca önce
   `python -m agent.run --frames test_frames_tmp --gemini real`, sonra Faz 4 (`agent/experiments.py`).
2. Gerçek webcam kareleri (`test_frames/`) yok. Bakma skoru, poz işaretleri ve py-feat eşiği bunlarla kalibre edilmeli.
3. ~~py-feat HF Hub isteği~~ → Oturum 4'te çözüldü.
4. `main_agent.py` gerçek tarayıcı ve webcam ile denenmedi (yalnızca test client).
5. Faz 5 (TSAM) başlanmadı.
6. git: commit yok.

## Oturum 4 (2026-09-29, ~09:00–09:05): doğrulamalar + Faz 5 keşfi

### Sıcak süreç süresi ve paralellik (ölçüldü)
| Senaryo | Toplam |
|---|---|
| Aynı süreçte 1. çalıştırma (soğuk, model yükleme dahil) | 11.8 s |
| 2. ve 3. çalıştırma (sıcak; Flask'ta açılıştan sonraki her analiz) | **1.00 s / 1.02 s** |
| Paralellik testi: gemini_analyst'e yapay 2 s gecikme (ağ yok) | 2.20 s (face_measure 0.82 s ∥ gemini 2.00 s) |

→ `face_measure` ile `gemini_analyst` **gerçekten paralel** çalışıyor (sıralı olsaydı ~2.8 s sürerdi).
Test scripti scratchpad'de (`warm_parallel.py`).

### py-feat çevrimdışı yükleme
- `HF_HUB_OFFLINE=1` ile aynı sonuç (AU12 = 0.9342 her iki modda) ve HF Hub isteği/uyarısı yok. Yükleme 10.2 s → 9.5 s.
- `agent/__init__.py`: `feat/resources` altında `*.pth` varsa `HF_HUB_OFFLINE=1` varsayılan olarak ayarlanıyor.
  Bu ayar en başta yapılmalı, çünkü huggingface_hub env'i import anında okuyor ve timm onu py-feat'ten önce import ediyor.
  İlk kurulumda (resources boşken) devreye girmiyor. Tam akışta uyarı artık çıkmıyor ✅.

### Faz 5 (TSAM) keşfi, kurulum YAPILMADI
Kaynak: github.com/gmontana/DecodingViewerEmotions README
- Tahmin: 5 sn'lik video klipten **8 duygu: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise**.
  Bu küme **HSEmotion'un etiketleriyle birebir aynı**, dolayısıyla `agreement._ad_insight` (beklenen vs ölçülen) eşleme gerektirmeden çalışır.
- Girdi: video + **ses** (ikisi birlikte işleniyor).
- Çıkarım: `python predict.py --data config/default.json --model weights --type test --id <id>`
  → `./data/predicted/<id>/`
- Gereksinimler: Python 3.10+, PyTorch 2.5+ (bizde 3.12 / 2.11, uyumlu görünüyor), CUDA GPU, `requirements.txt`,
  **sistem ffmpeg'i**. Global kurulum yasak olduğu için `imageio-ffmpeg` gibi venv içi bir ikili düşünülmeli (araştırılacak).
- Ağırlıklar: HuggingFace `dnamodel/tsam-viewer-emotions` → `backbone_weights.tar` (ResNet50, ImageNet-21K) +
  `tsam_weights.tar`. Boyut belirtilmemiş.
- Lisans: yalnızca akademik araştırma ve ticari olmayan değerlendirme. Proje bu kapsamda (capstone), ama ağırlıklar
  repoya eklenmemeli (`models/` zaten `.gitignore`'da).
- Önerilen plan (sonraki oturum):
  1. Repo `third_party/DecodingViewerEmotions` altına klonlanır, `requirements.txt` önce `--dry-run` ile torch sabitlenerek kurulur.
  2. `sample_ad.mp4` 5 sn'lik kliplere bölünür.
  3. `predict.py` çalıştırılır.
  4. Çıktı `{"segments":[{"start","end","expected_emotion"}]}` biçimine çevrilir ve `agent_results/ad_cache/<video sha1>.json` altında önbelleğe alınır.
  5. `run.py --ad-analysis` ile akışa verilir.

## Oturum 5 (2026-09-29 akşam, ~22:45–23:00): Faz 5 (TSAM reklam analizi) ✅

### Kurulum ve indirilenler
- Repo: `third_party/DecodingViewerEmotions` (shallow clone, commit 8905404, ~1 MB). Kodu **değiştirilmedi**,
  yalnızca import ediliyor. `third_party/`, `product_frames/` ve `product_results/` `.gitignore`'a eklendi.
- Ağırlıklar: HF `dnamodel/tsam-viewer-emotions` → `models/tsam/` (`tsam_weights.tar` 188.6 MB, `backbone_weights.tar` 188.6 MB, `LICENSE`).
- Paketler (`.tools/torch-constraints.txt` ile torch sabit, önce `--dry-run`): `torchaudio 2.11.0+cu128`,
  `soundfile 0.14.0`, `imageio-ffmpeg 0.6.0` (ffmpeg 7.1 ikilisi **.venv içinde**, sistem kurulumu yok).
  Kurulumdan sonra `torch.cuda.is_available()=True`.

### Güvenlik: checkpoint yükleme
- `weights_only=True` varsayılanıyla açılmıyordu. `weights_only=False` (keyfi kod çalıştırabilir) KULLANILMADI.
- Önce pickle içeriği **çalıştırılmadan** `pickletools` ile incelendi. Referanslar yalnızca şunlar:
  `torch._utils._rebuild_tensor_v2`, `torch.FloatStorage/LongStorage`, `collections.OrderedDict`,
  `numpy dtype`, `numpy.core.multiarray.scalar`, `_codecs.encode`.
- Bu referanslar `torch.serialization.safe_globals` ile açıkça izin listesine alındı ve dosya `weights_only=True` ile yüklendi.
  numpy 2'de modül `numpy._core` olduğu için eski ad eşlendi.

### Doğrulanan / düzeltilen bilgiler (belgeler tutarsızdı)
- **Sınıf sayısı 8** (`last_fc` 8×2048). README'deki 8 duygu doğru; HF `config.json` (7 etiket, Neutral yok) yanlış.
  İndeks sırası: `Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise` (`mvlib/mvideo_lib.py`
  `Emotions.list`; `emotion_ids` 1..8 sıralı eşleniyor, `background_size=-1` → arka plan sınıfı yok).
  **HSEmotion ile aynı etiket kümesi**, eşleme gerekmiyor.
- `tsam_weights.tar`: epoch 7, dengeli doğruluk **0.467** (8 sınıf, şans düzeyi 0.125).
- `backbone_weights.tar` README'de "ImageNet-21K ResNet50" diye anlatılıyor, oysa aslında **başka bir TSAM checkpoint'i**
  (epoch 4, dengeli doğruluk 0.422). Kullanılmıyor; omurga `tsam_weights.tar`'ın içinden yükleniyor.
- Model, izleyicilerin 5 sn'lik klipte en çok bildirmesi beklenen duyguyu tahmin ediyor. Reklamın "amaçladığı" duyguyu değil.

### `agent/ad_analysis.py`
- Ön işleme repo'nun kendi scriptiyle aynı: `ffmpeg -vf "scale=-1:256,fps=10" -q:v 0 %06d.jpg`, ses varsayılan wav.
- Model repo sınıflarıyla (`VCM`, `X_input`, `BackBone`) `config/default.json` + `num_class=8` üzerinden kuruluyor.
  Veri `PredictFullVideoDataSet` + `DataLoader` (batch 1, `num_workers=0`) ile, `predict.py` ile aynı yoldan besleniyor.
- `torchaudio.load` yalnızca çalışma süresince soundfile tabanlı bir yükleyiciyle değiştiriliyor (torchaudio 2.11 FFmpeg
  DLL'lerine ihtiyaç duyabiliyor). Aynı dönüş biçimi: `(Tensor[kanal, örnek], sr)`.
- Çıktı: pencere başına 8 sınıf softmax olasılığı + `expected_emotion`. Önbellek: `agent_results/ad_cache/<sha1>.json`.
  Geçici kare/ses klasörü iş bitince siliniyor.
- `sample_ad.mp4` (10 sn, 1280×720, 24 fps, AAC stereo) sonucu, GPU'da ön işleme 0.3 s + model 3.7 s:

| Pencere | Beklenen | İlk 3 olasılık |
|---|---|---|
| 0–5 sn | Surprise | Surprise .62, Contempt .24, Neutral .10 |
| 1–6 sn | Surprise | Surprise .66, Contempt .19, Neutral .10 |
| 2–7 sn | Surprise | Surprise .49, Contempt .18, Disgust .16 |
| 3–8 sn | Disgust | Disgust .41, Surprise .29, Neutral .25 |
| 4–9 sn | Surprise | Surprise .41, Neutral .32, Disgust .16 |

  Doğruluğu bu reklam için doğrulanamaz (etiket yok), yalnızca çalıştığı ve anlamlı biçimde değiştiği gösterildi.

### Entegrasyon
- `agreement.ad_segment_for`: 1 sn adımla çakışan pencerelerden merkezi saniyeye en yakın olanı seçer.
- `judging.rule_recommender`: beklenen ≠ ölçülen olduğunda öneri üretir. Güven hakem güveninin **yarısı**
  (tek izleyici + model belirsizliği). Metin "reklam modeli izleyicilerde X bekliyor" diyor, "sahne X'i hedefliyor" demiyor.
- `run.py --ad-video sample_ad.mp4` (ya da hazır JSON için `--ad-analysis`).
- `main_agent.py`: `sample_ad.mp4` analizi açılışta (arka planda) yapılıp önbellekleniyor ve her finalize'da graph'a veriliyor.
  TSAM yoksa ya da hata verirse analiz içgörüsüz devam ediyor.
- Testler: `run.py --mock-gemini --ad-video` ✅ (6. sn Disgust ≠ Contempt, 8. sn Surprise ≠ Contempt içgörüleri),
  Flask test client ✅, `test_agreement` ✅.

### Faz 5 açık noktalar
- Pencereler `range(T - clip_length)` ile üretiliyor (repo davranışı). 10 sn'lik reklamda son pencere 4–9 sn,
  bu yüzden **10. saniye TSAM kapsamı dışında** kalıyor ve o saniye için içgörü yok.
- Model dengeli doğrulukta 0.47. Çıktı "beklenti" olarak kullanılmalı, kanıt olarak değil (öneri güveninin yarıya indirilmesinin nedeni bu).

## Genel durum (plana göre)
| Faz | Durum |
|---|---|
| 0 Kurulum | ✅ |
| 1 Yüz tool'ları (HSEmotion, py-feat, MediaPipe) | ✅ (kalibrasyon için gerçek webcam karesi gerekiyor) |
| 2 Uyum hesaplayıcı | ✅ |
| 3 LangGraph + CLI | ✅ Gemini'siz ve mock ile doğrulandı. **Gerçek Gemini ile çalıştırılmadı.** |
| 4 Deneyler | ⏸ Script hazır, **`GEMINI_API_KEY` bekliyor** |
| 5 TSAM | ✅ |
| 6 Flask (`main_agent.py`) | ✅ test client ile. Gerçek tarayıcı/webcam ile denenmedi |

"Bitti" kriterleri: `python -m agent.run --frames test_frames/` hatasız çalışıyor ve JSON + grafik üretiyor ✅
(`test_frames_tmp` ile; `test_frames/` henüz yok). Faz 4 sonuçları anahtar gelince eklenecek.
Kalan her iş ya `GEMINI_API_KEY`'e ya da gerçek webcam karelerine bağlı. git: commit yok.

## Oturum 6: review düzeltmeleri (2026-09-29 gece, ~23:30–)

### 0. İlk commit ✅
- `.gitignore` doğrulandı (`git check-ignore`): `.venv`, `.tools`, `models`, `third_party`, `agent_results`,
  `product_frames` ve `product_results` hepsi dışarıda. Staged dosyalarda Gemini anahtarı deseni (`AIza…`) taraması: yok.
- Commit `2f60935` "Faz 0-6: ilk sürüm" (31 dosya). `sample_ad.mp4`, `pokemon.png` ve `test_frames_tmp/` dahil.

### 1. Sahte "uyumlu" kararı ✅
- `agreement.py`:
  - Gemini o saniye için veri vermediyse üç kontrol de `unverifiable` ("Gemini analizi yok").
  - `_check_au_claims`: AU iddiası yoksa `agree` yerine `unverifiable` ("AU iddiası yok").
  - `_check_attention`: yüz bulundu ama Gemini yüksek ilgi demiyorsa `unverifiable` ("Doğrulanacak ilgi iddiası yok").
    `agree` yalnızca "yüksek ilgi + ekrana bakıyor" durumunda dönüyor.
- `rule_judge`:
  - Hiçbir kontrol ölçülemediyse `trusted_source="local"`, gerekçe "Yalnızca yerel ölçüm; Gemini ile karşılaştırma
    yapılamadı." ve güven `min(…, LOCAL_ONLY_MAX_CONF=0.6)`.
  - "Kaynaklar uyumlu (…)" yalnızca en az bir kontrol gerçekten `agree` olduğunda yazılıyor ve hangi kontrollerin tuttuğunu da söylüyor.
- `test_agreement.py`: çelişki beklentileri korunuyor, `unverifiable` beklentileri eklendi, 10. sn'de "uyumlu" kontrolü var.
  **Yeni Gemini-off senaryosu:** tüm kontroller `unverifiable`, `conflict_ratio=None`, hakem hiçbir saniyede "uyumlu"
  demiyor ve güven ≤ 0.6. → `OK (mock Gemini)`, `OK (Gemini off)` ✅
- Yan etki: `unverifiable` kontroller paydaya girmediği için `conflict_ratio` yükseldi (mock verisinde 2 sn 0.5 → 1.0,
  6 sn 0.667 → 1.0, 8 sn 0.333 → 0.5). Gerçek Gemini modunda `REANALYZE_CONFLICT_RATIO=0.5` eşiği artık daha kolay
  aşılır; gerçek veriyle gözden geçirilmeli.

### 2. Gemini çıktısını normalleştirme ✅
- `gemini_client.normalize_analysis(obj) -> (liste, hata)`:
  - Tek obje → `[obj]`.
  - Sarmalanmış dict (ör. `{"results": [...]}`) → içteki ilk dict listesi.
  - `second` eksik ya da sayıya çevrilemiyorsa öğenin konumuna göre `DEFAULT_SECONDS` atanıyor (`_second_assigned: true`).
    String/float saniye `int`'e çevriliyor.
  - Liste değil ya da boşsa hata dönüyor.
- `graph.gemini_analyst`: normalleştirme hatasında `gemini_meta.status="error"` oluyor ve analiz olarak önceki liste
  (yoksa `[]`) kullanılıyor. Böylece `agreement`'a her zaman liste gidiyor.
- `agent/test_normalize.py` (ağ yok): tek obje, sarmalanmış dict, second'sız öğe, string saniye, 7 bozuk girdi ve
  `gemini_analyst` düğümünde sahte `generate_json` ile 3 durum (sarmalanmış → ok/2 öğe, "bozuk" → error/0,
  tek obje → ok/1) test edildi → `OK` ✅

### 3. Hakem ve öneri çıktısının doğrulanması ✅
- `judging.validate_verdicts` / `validate_recommendations`. Şemaya uymama durumları:
  - `verdicts`/`recommendations` liste değil,
  - öğe dict değil,
  - hakem kararında `second`/`trusted_source`/`confidence`, öneride `second`/`change`/`confidence` eksik,
  - `second` sayıya çevrilemiyor,
  - `confidence` sayı değil ya da 0-1 dışında.
- `judge`/`recommend`: şemaya uymazsa ya da çağrı istisna fırlatırsa `rule_judge`/`rule_recommender` kullanılıyor ve
  `fallback_reason` yazılıyor. Geçerli çıktıda `second` `int`'e çevriliyor.
- `v[...]` erişimleri `.get` yapıldı: `rule_recommender` (bozuk/eksik karar ve segment toleransı), `reporter._per_second_rows`,
  `agreement.ad_segment_for`/`_ad_insight`, `main_agent._results_from_local` ve `agent_verdict` eşlemesi.
  `graph.py`'deki except blokları artık yalnızca bu dayanıklı fonksiyonları çağırıyor.
- `agent/test_validation.py` (ağ yok, sahte `generate_json`):
  - 9 bozuk hakem çıktısı, 1 istisna ve 6 bozuk öneri çıktısının hepsi yedeğe düştü ve `fallback_reason` yazıldı.
  - Geçerli çıktı kabul edildi (`"6"` → `6`).
  - `rule_recommender`'a `None`/`{}`/eksik alanlı kararlar verildi, istisna yok.
  - Uçtan uca graph + reporter, hakem ve öneri için bozuk çıktıyla çalıştı. Örnek `fallback_reason`:
    "LLM hakem çıktısı şemaya uymadı: verdicts liste değil" → `OK` ✅
- Regresyon: `test_agreement` ✅, `main_agent.py` derleniyor ✅.

### 4. Kalibre edilmemiş dikkat skoru için güven tavanı ✅
- `config.ATTENTION_CALIBRATED = False` ve `config.UNCALIBRATED_ATTENTION_MAX_CONF = 0.5` eklendi.
- `rule_recommender`: kalibrasyon yoksa "dikkat düşüyor" önerilerinin güveni `min(hakem güveni, 0.5)` ve `reason`'a
  "(bakma skoru henüz kalibre edilmedi)" ekleniyor.
- Test: güven 0.8 olan karar → öneri 0.5 + not. `ATTENTION_CALIBRATED=True` iken 0.8 korunuyor. Kalıcı kontrol
  `test_validation.py`'ye eklendi → `OK` ✅

### 5. `main_agent.py`: `product_ux_results.csv` geri eklendi ✅
- Düzleştirme bloğu `main (1).py`'den **programatik olarak birebir kopyalandı** (94 satır). Tek ek, iki sütun:
  `agent_confidence` ve `agent_trusted_source` (`agent_verdict`'ten, `.get` ile).
- Yerel yedekte (Gemini atlandığında) ölçülmeyen değerler artık uydurulmuyor. Önceden `valence_score=0` ve
  `arousal_score=0` yazılıyordu, şimdi `None` (CSV'de boş). Yüz yoksa duygu etiketi "olculemedi" kalıyor, çünkü
  `index.html` boş etiketi "Nötr" olarak gösterirdi. Sayısal alanlar için `index.html` zaten `?? 0` kullanıyor.
  Gemini'ye özgü sütunlar (aida_stage, purchase_intent, …) boş.
- `agent/test_flask.py` (test client, ağ yok): reset → 5 kare → tıklama → finalize → CSV → reset. Kontroller:
  - finalize 200 döndü,
  - CSV sütunları = orijinal 13 sütun + 2 ek sütun,
  - satır sayısı = sonuç sayısı,
  - 7.5 sn'deki tıklama 8. saniyeye eşlendi,
  - reset CSV'yi sildi → `OK` ✅
- Not: orijinal düzleştirme `r.get("affective_core", {}).get(...)` kullanıyor. Gemini `affective_core: null` döndürürse
  orijinalde olduğu gibi burada da hata verir. "Aynı mantık" istendiği için değiştirilmedi.

### 6. `graph._tools()` kilidi ✅
- `threading.Lock` ile çift kontrollü yükleme: yüklendikten sonra kilitsiz hızlı yol; kilidi bekleyen ikinci çağrı
  ilkinin yüklediği örnekleri kullanıyor.
- `agent/test_tools_lock.py` (sahte, yavaş yüklenen tool'lar; 4 thread `Barrier` ile aynı anda başlıyor): her model
  **1 kez** yüklendi ve 4 çağrı da aynı nesneyi aldı → `OK` ✅.
  Kontrol: kilit geçici olarak `nullcontext` ile devre dışı bırakılınca test **başarısız oldu** (her model 4 kez
  yüklendi). Yani test yarışı gerçekten yakalıyor.

### 7. `experiments.py`: `--repeat` ve gürültü baz çizgisi ✅ (mantık doğrulandı, gerçek çalıştırma anahtar bekliyor)
- İki deneye de `--repeat N` (varsayılan 3) eklendi. Çağrı sayısı: `same` → N, `shuffle` → 3N (varsayılan 9 Gemini çağrısı).
- `same`: her tekrar için `valence_range` (+ std, son−ilk farkı, farklı AIDA sayısı). Özet olarak `valence_range_per_rep`,
  ortalama ve maksimum raporlanıyor.
- `shuffle`: her tekrarda üç çağrı var: ileri (A), yine ileri (B), ters (R).
  - `baseline_mad` = |A−B|: aynı sıra, iki çağrı, yani **gürültü**.
  - `order_mad_same_frame` = |A−R|: aynı kare, farklı pozisyon.
  - `mad_same_position`: aynı pozisyon, farklı kare.
  - Karar kuralı: `order_mad > ORDER_EFFECT_RATIO (2.0) × baseline_mad` **ve** `order_mad − baseline_mad > ORDER_EFFECT_MIN_DELTA (0.1)`
    ise "var"; değilse "yok (gürültüden ayırt edilemiyor)". Eşikler `config.py`'de.
- Model çıktısı artık `normalize_analysis`'ten geçiyor. Sonuç JSON'una `model` alanı ekleniyor.
- `agent/test_experiments.py` (sahte `_call`, ağ yok, seed=0):

| Senaryo | baseline | order | same_pos | Karar |
|---|---|---|---|---|
| A: valence kareye bağlı + küçük gürültü | 0.011 | 0.022 | 0.331 | yok ✅ (oran 2×, ama fark < 0.1) |
| B: valence pozisyona bağlı ("hikâye eğrisi") | 0.019 | 0.632 | 0.012 | **var** ✅ |
| C: kareye bağlı + yüksek gürültü (±0.4) | 0.268 | 0.258 | 0.330 | yok ✅ |

  `same`: hikâye modeli `valence_range_per_rep = [1.1, 1.1, 1.1]`, dürüst model `[0.0, 0.0, 0.0]` → `OK` ✅
- Gerçek çalıştırma: `python -m agent.experiments shuffle --frames test_frames_tmp --repeat 2` →
  "GEMINI_API_KEY ortam değişkeni gerekli" (beklenen, exit 1).

### 8. `GEMINI_MODEL` ortam değişkeninden okunuyor ✅
- `config.GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"`. Test: varsayılan `gemini-2.5-flash`,
  `GEMINI_MODEL=gemini-3.6-flash` ile `gemini-3.6-flash` ✅
- **Adil kıyas notu:** orijinal `main (1).py` (eski `google.generativeai` SDK'sı) `gemini-3.6-flash` modelini kullanıyor.
  Ajan katmanı plana uygun olarak `gemini-2.5-flash` ile çalışıyor. Orijinal sistemle aynı modelle kıyas için:
  `GEMINI_MODEL=gemini-3.6-flash python -m agent.run --frames … --gemini real` (deneylerde de aynı env geçerli;
  kullanılan model sonuç JSON'unda `model` alanına yazılıyor).

### 9. PROGRESS tarihleri düzeltildi ✅
- "2026-09-30" geçen her başlık düzeltildi (Oturum 2–6). Tüm çalışma 2026-09-29 tarihinde yapıldı.
- Saatler `agent_results/*` ve `models/*` dosyalarının zaman damgalarından alındı:
  - Oturum 1: 01:09 (nvidia-smi çıktısı) – 01:31 (`phase1/`).
  - Oturum 2–4: `mock_test/` 08:51 … `offline_check/` 09:03.
  - Oturum 5: `models/tsam` 22:52, `ad_cache/` 22:56, `20260929_225836/` 22:58.
  - Oturum 6: 23:3x.
- **Uyuşmazlık notu:** review'da Oturum 2–5'in 29'u akşam/gece yapıldığı yazıyordu. Zaman damgaları ise Oturum 2–4'ün
  29'u sabahı (~08:50–09:05), Oturum 5'in 29'u akşamı yapıldığını gösteriyor. Başlıklarda zaman damgaları esas alındı.

### Son çalıştırmalar (Oturum 6 sonu)

**1) `python -m agent.test_agreement`** ✅
```
2 conflicts: ['attention'] unverifiable: ['valence', 'au_claims'] ratio: 1.0
4 conflicts: [] unverifiable: ['valence', 'attention', 'au_claims'] ratio: None
6 conflicts: ['valence', 'au_claims'] unverifiable: ['attention'] ratio: 1.0 insights: ['Contempt']
8 conflicts: ['attention'] unverifiable: ['au_claims'] ratio: 0.5 insights: ['Contempt']
10 conflicts: [] unverifiable: ['au_claims'] ratio: 0.0
OK (mock Gemini): total_conflicts 4, seconds_with_conflict [2, 6, 8]
OK (Gemini off): tüm kontroller unverifiable, hakem 'uyumlu' demiyor, güven <= 0.6
```

**2) `python -m agent.run --frames test_frames_tmp`** (Gemini off, `agent_results/final_off`) ✅
Artık "uyumlu" denmiyor. Önce/sonra karşılaştırması:

| sn | ÖNCE (`20260929_225836`) | SONRA (`final_off`) |
|---|---|---|
| 2, 4 | none 0.1 "Yüz ölçülemedi" | none 0.1 "Yüz ölçülemedi" |
| 6, 8, 10 | **mixed 0.8 "Kaynaklar uyumlu."** | **local 0.6 "Yalnızca yerel ölçüm; Gemini ile karşılaştırma yapılamadı."** |

Süre: toplam 11.8 s (model yükleme 9.9 s; hsemotion 0.36 / mediapipe 0.10 / pyfeat 1.18 s).

**3) `python -m agent.run --frames test_frames_tmp --mock-gemini --ad-video sample_ad.mp4`** (`final_mock_ad`) ✅
- TSAM: 5 pencere, önbellekten. Toplam 9.4 s.
- Kararlar:
  - 2, 4 sn: none 0.1.
  - 6 sn: local 0.2 (valence + au_claims çelişkisi).
  - 8 sn: local 0.5 (attention çelişkisi).
  - 10 sn: **mixed 0.8 "Kaynaklar uyumlu (valence, attention)."** Yani "uyumlu" yalnızca gerçekten karşılaştırılan
    kontrollerde yazılıyor.
- Öneriler: 6 ve 8. sn "dikkat düşüyor" (0.2 / 0.5, "(bakma skoru henüz kalibre edilmedi)");
  TSAM içgörüleri 6 sn Disgust≠Contempt (0.1), 8 sn Surprise≠Contempt (0.25).

**4) Flask test client (`python -m agent.test_flask`)** ✅
- finalize 200 `completed` (gemini: skipped), `product_results/product_ux_results.csv` (5×15) oluştu.
- 6/8/10 sn `agent_trusted_source=local`, `agent_confidence=0.6`; yüzsüz saniyelerde valence boş; tıklama 8. sn'de.
- Öneriler dikkat için 0.5 tavanında. Reset CSV'yi sildi.

**Regresyon:** `test_normalize` ✅, `test_validation` ✅, `test_tools_lock` ✅, `test_experiments` ✅

### Oturum 6'da eklenen test dosyaları
`agent/test_normalize.py`, `agent/test_validation.py`, `agent/test_flask.py`, `agent/test_tools_lock.py`,
`agent/test_experiments.py`. Hepsi ağsız ve `python -m agent.<ad>` ile çalışıyor.

### Açık kalanlar (değişmedi)
1. Gerçek Gemini ile uçtan uca çalıştırma ve Faz 4 deneyleri: `GEMINI_API_KEY` bekliyor.
2. Gerçek webcam kareleri (`test_frames/`) ile bakma skoru kalibrasyonu; ardından `ATTENTION_CALIBRATED=True`.
3. `REANALYZE_CONFLICT_RATIO` gerçek veriyle gözden geçirilmeli (madde 1'in yan etkisi, bkz. yukarı).
4. `main_agent.py` gerçek tarayıcı ve webcam ile denenmedi.

## Mac kurulumu (2026-09-30, ~00:05–00:35)

Apple Silicon (arm64, macOS 26 / Darwin 25.6), GPU yok, her şey CPU. Kod mantığına dokunulmadı; aşağıdaki iki sorun
yalnızca `.venv` içinde çözüldü. Global kurulum yapılmadı (brew vb. yok).

### Kurulum
- `uv 0.11.18` sistemde vardı (`~/.local/bin/uv`). `uv venv --python 3.12 .venv` → CPython 3.12.13 (uv'nin kendi yorumlayıcısı).
- torch standart PyPI'dan (cu128 index yok). En güncel 2.14.0 geldi; Windows ile eşlik için `torch==2.11.0`,
  `torchvision==0.26.0`'a indirildi ve `.tools/torch-constraints.txt` ile sabitlendi.
- Paketler: `uv pip install --python .venv -c .tools/torch-constraints.txt hsemotion timm mediapipe opencv-python py-feat
  langgraph google-genai flask pandas pillow matplotlib soundfile imageio-ffmpeg`.
- Süre: venv + torch 15 s, diğer paketler ~27 s. `.venv` ≈ 1.1 GB (Windows'ta CUDA ile 4.6 GB).
- `torch.cuda.is_available()=False` → kod zaten `cpu`'ya düşüyor (hsemotion/pyfeat/ad_analysis). MPS kullanılmıyor.

| Paket | Sürüm |
|---|---|
| Python | 3.12.13 |
| torch / torchvision | 2.11.0 / 0.26.0 (CPU) |
| hsemotion / timm | 0.3.0 / 1.0.30 |
| mediapipe | **1.0.0** (1.0.1 değil, bkz. sorun 2) |
| py-feat / xgboost | 2.1.3 / 3.4.1 |
| numpy | 2.5.3 |
| langgraph / google-genai | 1.2.12 / 2.25.0 |
| flask / pandas / pillow / matplotlib | 3.1.3 / 3.0.6 / 12.3.0 / 3.11.2 |
| soundfile / imageio-ffmpeg | 0.14.0 / 0.6.0 |

### Mac'e özgü sorunlar
1. **xgboost `libomp.dylib` bulamıyor** (py-feat import'u düşüyor). Dylib'in tek rpath'i `/opt/homebrew/opt/libomp/lib`.
   `brew install libomp` yerine torch'un paketlediği libomp kullanıldı:
   `install_name_tool -add_rpath @loader_path/../../torch/lib .venv/.../xgboost/lib/libxgboost.dylib` + `codesign --force -s -`.
   Böylece süreçte tek libomp kopyası var. **xgboost yeniden kurulursa bu adım tekrarlanmalı.**
2. **mediapipe 1.0.1 macOS'ta FaceDetector'da çöküyor:** `graph_service.h:139 Check failed: service_ Service is unavailable`
   (`TensorsToDetectionsCalculator` → `DrishtiMetalHelper`, CPU delegate'e rağmen Metal servisi istiyor; süreç abort).
   Ayrı venv'de izole testle 1.0.0, 0.10.35, 0.10.32, 0.10.21 çalışıyor, yalnızca 1.0.1 çöküyor → `mediapipe==1.0.0`.
3. **Port 5000 macOS'ta AirPlay Receiver'da** (ControlCenter, `*:5000`). Flask `127.0.0.1:5000`'e daha spesifik bağlandığı için
   çakışma olmadı; ancak tarayıcıda `localhost:5000` IPv6'ya (::1) çözülürse AirPlay'e (403 AirTunes) gidebilir →
   `http://127.0.0.1:5000` kullanılmalı (`main_agent.py` zaten bunu açıyor).
4. Reklam analizi `agent_results/ad_cache/89db…925f.json`'dan geliyor (`sample_ad.mp4` SHA1 eşleşiyor); TSAM çalışmadı, beklendiği gibi.

### Testler
**1) `python -m agent.run --frames test_frames_tmp`** (Gemini off) ✅
- İlk çalıştırma (py-feat ağırlıkları HF'den indi): toplam 41.5 s (model yükleme 35.7 s; hsemotion 0.56 / mediapipe 0.08 / pyfeat 4.50 s).
- İkinci (ısınmış): toplam **16.1 s** (model yükleme 12.3 s; hsemotion 0.56 / mediapipe 0.07 / pyfeat 3.03 s). Windows/CUDA: 11.8 s.
- Çıktı Windows `final_off` ile aynı: yüz 6/8/10 sn, Contempt/Contempt/Happiness, valence 0.66/0.5137/0.8606
  (Windows 0.6601), bakma skorları 4. ondalıkta farklı (0.3682 vs 0.3678).

**2) Regresyon:** `test_agreement`, `test_normalize`, `test_validation`, `test_experiments`, `test_tools_lock`, `test_flask` ✅

**3) `python main_agent.py` + tarayıcı + webcam (MacBook kamerası)** ✅
- Sunucu açıldı, model ön yüklemesi arka planda ~11 s. Kamera izni verildi, "Testi Başlat" ile bir tur yapıldı.
- 5 kare (2 sn aralık) yüklendi → `finalize_and_analyze` 200. Çıktı: `agent_results/20260930_003052`.
- Sonuç: 5/5 saniyede yüz bulundu, HSEmotion hepsi Neutral, valence −0.24…0.04, bakma skoru 0.12–0.49 (kalibre değil,
  hepsi "no"); güven 0.6, kaynak local. Gemini skipped (anahtar yok).
- Süre (modeller sıcak): face_measure **2.19 s** (hsemotion 0.32 / mediapipe 0.05 / pyfeat 1.83 s), diğer düğümler ~0.
- Böylece "`main_agent.py` gerçek tarayıcı ve webcam ile denenmedi" açık maddesi Mac'te kapandı.

## Hafif sürüm (2026-10-06 gece): PyTorch ve py-feat kaldırıldı

Amaç: grubun projesi gibi "kur ve çalıştır" olsun. Ajan mimarisi (LangGraph, paralel dallar, uyum, hakem, geri besleme
döngüsü, öneri, rapor) aynen kaldı; yalnızca yüz ölçümünün altındaki ağır bağımlılıklar değişti.

### Değişiklikler
- **HSEmotion → ONNX:** `enet_b0_8_va_mtl.pt` tek seferlik `torch.onnx.export` (opset 17, dinamik batch) ile
  `models/enet_b0_8_va_mtl.onnx`'e çevrildi. Rastgele 4'lü batch'te PyTorch ile en büyük fark **1.2e-6**.
  `hsemotion_tool.py` artık onnxruntime (CPU) ve numpy/PIL ön işleme kullanıyor. Test karelerinde duygu, valence,
  arousal ve 8 olasılık PyTorch sürümüyle 4. ondalığa kadar **aynı**. Model yükleme 1.1 s → 0.22 s, kare ~22 ms.
- **py-feat kaldırıldı.** AU kontrolü için MediaPipe Face Landmarker'ın zaten ürettiği 52 blendshape'ten FACS AU vekilleri
  türetiliyor (`mediapipe_tool.AU_FROM_BLENDSHAPES`; ör. mouthSmile → AU12, browDown → AU4, cheekSquint → AU6).
  Ölçek farklı olduğu için `AU_PRESENCE_THRESHOLD` 0.5 → 0.3. AU verisi artık `face_metrics["mediapipe"].frames[i].aus`.
- **TSAM:** `ad_analysis.py` torch'u yalnızca önbellek yoksa (modeli gerçekten çalıştırırken) import ediyor. Hafif
  kurulumda örnek reklamın önbellek JSON'u kullanılıyor; önbellek yoksa main_agent hatayı yakalayıp içgörüsüz devam ediyor.
- `agent/__init__.py`'deki py-feat/HF çevrimdışı ayarı, `pyfeat_tool.py` ve `.pt` model dosyası kaldırıldı.
- `requirements.txt`: flask, pandas, pillow, numpy, matplotlib, google-genai, langgraph, mediapipe==1.0.0, onnxruntime.

### Ölçümler (temiz Linux venv, Python 3.12, CPU)
| | Önceki (torch + py-feat) | Hafif |
|---|---|---|
| Kurulum boyutu | ~1.5 GB (Mac) / 4.6 GB (Windows CUDA) + 330 MB py-feat ağırlığı | **~640 MB** (çoğu mediapipe'ın OpenCV'si), indirme yok |
| `agent.run --mock-gemini` toplam | 12.3 s (model yükleme 5–10 s, py-feat 1–7 s) | **0.9 s** (yükleme 0.4 s) |
| Mac özel yamaları | libomp rpath, mediapipe sürüm sabitleme | yalnızca mediapipe==1.0.0 |

Testler: `test_tools`, `test_agreement` (beklenen çelişkiler aynı: 2/6/8 sn), `test_validation`, `test_flask`,
`test_normalize`, `test_experiments`, `test_tools_lock` ✅

### Kaybedilen / dikkat
- Blendshape AU'ları eğitimli bir AU dedektörü değil, yaklaşık. Portrede (gülümseyen yüz) browDown 0.83 çıktı,
  yani AU4 yanlış pozitif verebilir. AU12 (gülümseme) tutarlı. Gerçek veriyle eşik gözden geçirilmeli.
- py-feat'in ikinci baş pozu ve L2CS bakış tahmini yok; dikkat skoru zaten yalnızca MediaPipe'tan geliyordu.
