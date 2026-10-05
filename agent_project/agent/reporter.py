"""reporter: state -> result.json + per_second.csv + chart.png (saniye bazlı)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from agent.agreement import index_by_second  # noqa: E402


def _per_second_rows(state: dict) -> list[dict]:
    fm = state.get("face_metrics") or {}
    hse, mp = index_by_second(fm.get("hsemotion")), index_by_second(fm.get("mediapipe"))
    gem = {int(float(r.get("second", 0))): r for r in (state.get("gemini_analysis") or [])}
    ver = {v.get("second"): v for v in (state.get("verdicts") or {}).get("verdicts") or []
           if isinstance(v, dict)}
    agr = {p["second"]: p for p in (state.get("agreement") or {}).get("per_second", [])}
    clicks = state.get("clicks") or []
    rows = []
    for sec in sorted(s for s in set(hse) | set(mp) | set(gem) | set(ver) if s is not None):
        h, m, g, v, a = hse.get(sec, {}), mp.get(sec, {}), gem.get(sec, {}), ver.get(sec, {}), agr.get(sec, {})
        rows.append({
            "second": sec,
            "face_found": h.get("face_found"),
            "valence_gemini": (g.get("affective_core") or {}).get("valence_score"),
            "valence_hsemotion": h.get("valence"),
            "valence_verdict": v.get("valence"),
            "emotion_gemini": (g.get("affective_core") or {}).get("dominant_emotion"),
            "emotion_hsemotion": h.get("dominant_emotion"),
            "looking_at_screen_score": m.get("looking_at_screen_score"),
            "attention_verdict": v.get("attention"),
            "confidence": v.get("confidence"),
            "trusted_source": v.get("trusted_source"),
            "conflicts": ";".join(a.get("conflicts", [])),
            "clicks": sum(1 for c in clicks if sec - 2 < float(c.get("video_time", -1)) <= sec),
        })
    return rows


def _chart(df: pd.DataFrame, path: Path, title: str, gemini_label: str = "Gemini"):
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax = axes[0]
    for col, label, style in [("valence_gemini", gemini_label, "o--"), ("valence_hsemotion", "HSEmotion", "s-"),
                              ("valence_verdict", "Hakem kararı", "^:")]:
        ax.plot(df["second"], pd.to_numeric(df[col], errors="coerce"), style, label=label)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Valence")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)

    ax = axes[1]
    ax.bar(df["second"], pd.to_numeric(df["looking_at_screen_score"], errors="coerce").fillna(0), width=0.8)
    for s, f in zip(df["second"], df["face_found"]):
        if not f:
            ax.text(s, 0.05, "yüz\nyok", ha="center", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Ekrana bakma")

    ax = axes[2]
    ax.bar(df["second"], pd.to_numeric(df["confidence"], errors="coerce").fillna(0), width=0.8)
    for s, c in zip(df["second"], df["conflicts"]):
        if c:
            ax.text(s, 0.02, c.replace(";", "\n"), ha="center", fontsize=6, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Güven")
    ax.set_xlabel("Saniye")
    ax.set_xticks(list(df["second"]))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(state: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _per_second_rows(state)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_second.csv", index=False, encoding="utf-8-sig")
    title = f"Oturum {state.get('session_id')} | hakem: {(state.get('verdicts') or {}).get('source')}"
    status = (state.get("gemini_meta") or {}).get("status")
    label = {"mock": "Gemini (SENTETİK)", "skipped": "Gemini (atlandı)"}.get(status, "Gemini")
    if rows:
        _chart(df, out_dir / "chart.png", title, label)
    result = {k: state.get(k) for k in ("session_id", "frames", "clicks", "ad_analysis", "face_metrics",
                                         "gemini_analysis", "gemini_meta", "agreement", "verdicts",
                                         "iteration", "recommendations", "timings")}
    result["per_second"] = rows
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    return {"out_dir": str(out_dir), "files": ["result.json", "per_second.csv", "chart.png"]}
