"""Gemini erişimi (google-genai SDK). Anahtar yalnızca GEMINI_API_KEY ortam değişkeninden okunur."""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

from agent import config


class GeminiUnavailable(RuntimeError):
    pass


def has_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


_client = None


def _get_client():
    global _client
    if not has_key():
        raise GeminiUnavailable("GEMINI_API_KEY ortam değişkeni tanımlı değil")
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def load_batch_ux_prompt() -> str:
    """BATCH_UX_PROMPT'u orijinal Flask dosyasından salt-okunur çeker (dosya import EDİLMEZ,
    çünkü import klasör oluşturur/Flask app kurar). Tek kaynak orijinal dosyada kalır."""
    for name in ("main.py", "main (1).py"):
        p = config.ROOT / name
        if p.exists():
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "BATCH_UX_PROMPT" for t in node.targets):
                    return ast.literal_eval(node.value)
    raise FileNotFoundError("BATCH_UX_PROMPT bulunamadı (main.py / 'main (1).py')")


def _parse_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def generate_json(prompt: str, images: list[Path] | None = None, temperature: float | None = None):
    """Metin + (opsiyonel) görseller -> JSON. Hata olursa ham metinle birlikte ValueError."""
    from google.genai import types

    parts = []
    for img in images or []:
        mime = "image/png" if img.suffix.lower() == ".png" else "image/jpeg"
        parts.append(types.Part.from_bytes(data=img.read_bytes(), mime_type=mime))
    parts.append(prompt)
    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=temperature)
    resp = _get_client().models.generate_content(model=config.GEMINI_MODEL, contents=parts, config=cfg)
    try:
        return _parse_json(resp.text)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Gemini JSON parse hatası: {e}; ham yanıt: {resp.text[:500]}") from e


def normalize_analysis(obj) -> tuple[list, str | None]:
    """Gemini analiz çıktısını her zaman bir listeye çevir. Dönüş: (liste, hata_mesajı | None).

    - list  -> dict öğeleri alınır
    - dict  -> tek saniye objesiyse ("second"/"affective_core" varsa) [obj]; değilse içindeki ilk
               dict-listesi (ör. {"results": [...]}) kullanılır
    - "second" alanı olmayan öğelere sırayla (listedeki konumuna göre) DEFAULT_SECONDS atanır
    - liste çıkarılamazsa ya da boşsa hata döner
    """
    items = None
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        if "second" in obj or "affective_core" in obj:
            items = [obj]
        else:
            items = next((v for v in obj.values()
                          if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), None)
    if items is None:
        return [], f"Gemini çıktısı liste değil ({type(obj).__name__})"
    out = []
    for i, it in enumerate(x for x in items if isinstance(x, dict)):
        it = dict(it)
        try:
            it["second"] = int(float(it["second"]))  # agreement/reporter sayısal saniye bekliyor
        except (KeyError, TypeError, ValueError):
            it["second"] = None
        if it["second"] is None:
            if i >= len(config.DEFAULT_SECONDS):
                continue  # atanacak saniye yok
            it["second"] = config.DEFAULT_SECONDS[i]
            it["_second_assigned"] = True
        out.append(it)
    if not out:
        return [], "Gemini çıktısı boş"
    return out, None
