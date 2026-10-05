# SIC Capstone: Ajan tabanlı reklam analizi (çalışma planı)

## Bağlam
Bu klasördeki uygulama (main.py + index.html) bir reklam izleyicisinin webcam karelerini
(2., 4., 6., 8., 10. saniye) ve video üzerindeki tıklamalarını topluyor. Ardından kareleri tek
bir Gemini çağrısına gönderip saniye bazlı duygu, valence/arousal, AIDA aşaması ve pazarlama
önerisi üretiyor. Temel sistem ekip arkadaşlarımın (İrem, Meryem) işi. Onu BOZMADAN üstüne
yeni bir katman kuruyoruz.

Sorun: Tüm analiz tek bir LLM tahminine dayanıyor ve doğrulanmıyor. İlk testte Gemini
"son karede ilgisini kaybetti" dedi. Bu yüzden değil, "son kare" olduğu için yazılmış gibi
duruyor.

Hedef: LangGraph ile çok kaynaklı bir analiz. Yerel yüz modelleri ölçüm yapar, Gemini
yorumlar, bir hakem ikisini karşılaştırır, en sonda reklamın nasıl iyileştirileceğine dair
öneri üretilir.

## Çalışma kuralları (önemli)
- Sadece bu klasörde çalış. Klasör dışındaki hiçbir dosyaya dokunma.
- `main.py` ve `index.html`'i DEĞİŞTİRME. Yeni kodu ayrı dosyalara/modüllere yaz (ör. `agent/`).
  Flask entegrasyonu gerekirse `main.py`'nin bir kopyası olan `main_agent.py`'de yap.
- Her şey `.venv` sanal ortamına kurulacak (uv kullan). Global Python'a paket kurma.
- API anahtarını hiçbir dosyaya yazma. `GEMINI_API_KEY` ortam değişkeninden okunur.
  `.env` kullanırsan `.gitignore`'a ekle.
- git: yerel repo başlat / `agent-layer` dalında çalış. Asla push yapma.
- İlerlemeyi `PROGRESS.md`'ye yaz: ne yapıldı, ne denendi, ne takıldı, ölçülen süreler.
  Bir adımda takılırsan nedenini PROGRESS.md'ye yaz ve sonraki adıma geç. Aynı hatada
  döngüye girme.
- Kütüphane API'lerini tahmin etme. Kurduktan sonra gerçek fonksiyon imzalarını/README'lerini
  kontrol et. Sürüm bilgilerini PROGRESS.md'ye not et.
- Makine: Windows, NVIDIA RTX 4060 Ti. GPU'yu kullan (CUDA'lı PyTorch). Kurulumu doğrula.

## Test verisi
- `test_frames/` klasöründe örnek webcam kareleri olacak (ben koyacağım). Yoksa
  `Student-engagement-dataset/` (Kaggle, yüz fotoğrafları) içinden birkaç kare kullan.
- Kaggle seti yalnızca test içindir. Az sayıda kişiden art arda kesilmiş kareler içerir,
  eğitim için KULLANMA.

## Görevler (sırayla)

### Faz 0: Kurulum
1. uv ile `.venv`, CUDA'lı PyTorch, mevcut bağımlılıklar (flask, pandas, pillow).
2. `python -c "import torch; print(torch.cuda.is_available())"` → True olmalı.

### Faz 1: Yüz ölçüm tool'ları (`agent/tools/`)
Her biri: girdi = kare yolu (veya kare listesi), çıktı = saniye bazlı JSON-uyumlu dict.
1. **HSEmotion**: 8 duygu olasılığı + valence + arousal (valence/arousal veren modeli seç).
2. **py-feat**: Action Unit yoğunlukları/varlığı (AU1, AU2, AU4, AU6, AU7, AU12, AU15, AU25, AU26 …),
   baş pozu.
3. **MediaPipe Face Landmarker**: baş yönü (yaw/pitch), göz/iris konumu → "ekrana bakıyor mu"
   (0-1 skor), göz kırpma blendshape'leri. Yüz bulunamazsa bunu açıkça raporla.
4. Her tool için küçük bir test scripti ve kare başına süre ölçümü (PROGRESS.md'ye).

### Faz 2: Uyum hesaplayıcı (`agent/agreement.py`, LLM YOK, düz Python)
Her saniye için:
- valence farkı (Gemini vs HSEmotion) → |fark| > 0.4 ise çelişki
- dikkat çelişkisi: MediaPipe "ekrana bakmıyor" ama Gemini yüksek ilgi/niyet diyor
- kanıtsız AU iddiası: Gemini `micro_expressions` içinde AU adı geçiyor ama py-feat ölçmüyor
- (reklam modeli varsa) beklenen duygu vs ölçülen duygu farkı → içgörü (çelişki değil)
Eşikler tek bir config dosyasında dursun.

### Faz 3: LangGraph (`agent/graph.py`)
State: session_id, frames, clicks, ad_analysis, face_metrics, gemini_analysis,
agreement, verdicts, iteration, recommendations.

Düğümler:
1. `face_measure`: Faz 1 tool'ları (sabit düğüm, LLM tool-calling değil)
2. `gemini_analyst`: mevcut BATCH_UX_PROMPT mantığı. Yeni `google-genai` SDK,
   model `gemini-2.5-flash`. Hakemden geri bildirim gelirse prompt'a eklenir.
3. `agreement`: Faz 2
4. `judge`: LLM (Gemini). Uyum tablosunu okur, her saniye için karar + 0-1 güven +
   gerekçe verir (structured JSON). Çelişki yüksekse ve iteration < 2 ise
   `gemini_analyst`'a gerekçeli geri bildirimle döner.
5. `recommender`: kararlar + (varsa) reklam zaman çizelgesi → somut düzenleme önerileri
   (hangi saniye, ne değişmeli, neden, güven).
6. `reporter`: JSON + CSV + saniye bazlı grafik (valence kaynaklara göre, dikkat, güven).

`face_measure` ile `gemini_analyst` paralel çalışabilir.
CLI: `python -m agent.run --frames test_frames/` → sonuç `agent_results/` altına.

### Faz 4: Ölçüm ve doğrulama deneyleri
1. **Aynı kare testi**: aynı kareyi 5 kez "2,4,6,8,10. saniye" diye Gemini'ye gönder.
   Gemini bir "hikâye eğrisi" uyduruyor mu? Sonucu kaydet.
2. **Karıştırma testi**: kareleri ters sırada gönder, sonuçları karşılaştır.
3. Her düğümün süresini ölç → PROGRESS.md'de tablo.

### Faz 5 (bonus): Reklam tarafı
Warwick "DecodingViewerEmotions" (github.com/gmontana/DecodingViewerEmotions) TSAM modeli.
Ağırlıklar HuggingFace'te. Lisans akademik/ticari olmayan. `sample_ad.mp4`'ü 5 sn'lik
kliplere bölüp beklenen duyguyu çıkar, `ad_analysis` olarak state'e ver. Sonuç reklam
başına önbelleğe alınsın.

### Faz 6 (bonus): Flask entegrasyonu
`main_agent.py`: `/finalize_and_analyze` tek Gemini çağrısı yerine graph'ı çağırsın.
Orijinal `main.py` aynen kalsın.

## Bitti sayılması için
- `python -m agent.run --frames test_frames/` hatasız çalışıyor, sonuç JSON + grafik üretiyor.
- PROGRESS.md: yapılanlar, süre tablosu, Faz 4 deney sonuçları, açık sorunlar.
