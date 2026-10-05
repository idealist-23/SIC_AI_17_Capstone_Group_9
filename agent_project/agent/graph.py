"""Faz 3: LangGraph akışı.

START ─┬─> face_measure ───┐
       └─> gemini_analyst ─┴─> agreement -> judge ─┬─> recommender -> reporter -> END
                 ^                                  │
                 └────── (çelişki yüksek, iteration < MAX, gerçek Gemini) ─┘

gemini_mode: "real" (GEMINI_API_KEY gerekli) | "mock" (sentetik, etiketli) | "off" (Gemini atlanır).
"""
from __future__ import annotations

import operator
import threading
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent import config, gemini_client, judging
from agent.agreement import compute_agreement
from agent.mock import mock_gemini_analysis
from agent.reporter import write_report
from agent.tools.common import list_frames


class AgentState(TypedDict, total=False):
    session_id: str
    frames: list[str]
    clicks: list[dict]
    ad_analysis: dict | None
    gemini_mode: str
    out_dir: str
    face_metrics: dict
    gemini_analysis: list
    gemini_meta: dict
    agreement: dict
    verdicts: dict
    feedback: str
    iteration: int
    recommendations: dict
    report: dict
    # Paralel dallar aynı anahtara yazabildiği için liste birleştirme reducer'ı
    timings: Annotated[list, operator.add]


# Yüz tool'ları pahalı yükleniyor (ilk yükleme birkaç sn); süreç boyunca bir kez yükle.
_TOOLS: dict[str, Any] = {}
# Flask açılışındaki önceden yükleme thread'i ile ilk istek aynı anda gelirse modeller iki kez yüklenmesin
_TOOLS_LOCK = threading.Lock()


def _tools():
    if _TOOLS:  # hızlı yol: yüklendikten sonra kilide gerek yok
        return _TOOLS
    with _TOOLS_LOCK:
        if not _TOOLS:  # kilidi bekleyen ikinci çağrı, ilkinin yüklediğini kullanır
            from agent.tools.hsemotion_tool import HSEmotionTool
            from agent.tools.mediapipe_tool import MediaPipeTool
            _TOOLS.update(hsemotion=HSEmotionTool(), mediapipe=MediaPipeTool())
    return _TOOLS


def _timed(name, t0, **extra):
    return [{"node": name, "ms": round((time.perf_counter() - t0) * 1000, 1), **extra}]


# ---------------- düğümler ----------------

def face_measure(state: AgentState) -> dict:
    t0 = time.perf_counter()
    frames = [Path(f) for f in state["frames"]]
    tools = _tools()
    load_ms = round((time.perf_counter() - t0) * 1000, 1)  # ilk çağrıda model yükleme, sonra ~0
    metrics, errors, per_tool = {}, {}, {}
    for name, tool in tools.items():
        t1 = time.perf_counter()
        try:
            metrics[name] = tool.analyze(frames)
        except Exception as e:  # noqa: BLE001 - bir tool düşerse diğerleri devam etsin
            errors[name] = repr(e)
        per_tool[name] = round((time.perf_counter() - t1) * 1000, 1)
    if errors:
        metrics["errors"] = errors
    return {"face_metrics": metrics,
            "timings": _timed("face_measure", t0, model_load_ms=load_ms, per_tool_ms=per_tool)}


def gemini_analyst(state: AgentState) -> dict:
    t0 = time.perf_counter()
    it = state.get("iteration", 0) + 1
    mode = state.get("gemini_mode", "off")
    frames = [Path(f) for f in state["frames"]]
    if mode == "mock":
        return {"gemini_analysis": mock_gemini_analysis(len(frames)), "iteration": it,
                "gemini_meta": {"status": "mock", "note": "SENTETİK veri, gerçek Gemini çağrısı değil"},
                "timings": _timed("gemini_analyst", t0, iteration=it)}
    if mode != "real" or not gemini_client.has_key():
        return {"gemini_analysis": [], "iteration": it,
                "gemini_meta": {"status": "skipped", "reason": "gemini_mode=off veya GEMINI_API_KEY yok"},
                "timings": _timed("gemini_analyst", t0, iteration=it)}
    prompt = gemini_client.load_batch_ux_prompt()
    if state.get("feedback"):
        prompt += ("\n\nHAKEM GERİ BİLDİRİMİ (önceki analizin yerel yüz ölçümleriyle çelişti; "
                   "bunu dikkate alarak yeniden analiz et):\n" + state["feedback"])
    try:
        raw = gemini_client.generate_json(prompt, images=frames)
        analysis, err = gemini_client.normalize_analysis(raw)
        if err:
            analysis, meta = (state.get("gemini_analysis") or []), {"status": "error", "error": err}
        else:
            meta = {"status": "ok", "model": config.GEMINI_MODEL, "with_feedback": bool(state.get("feedback"))}
    except Exception as e:  # noqa: BLE001
        analysis, meta = (state.get("gemini_analysis") or []), {"status": "error", "error": repr(e)}
    return {"gemini_analysis": analysis, "gemini_meta": meta, "iteration": it,
            "timings": _timed("gemini_analyst", t0, iteration=it)}


def agreement(state: AgentState) -> dict:
    t0 = time.perf_counter()
    res = compute_agreement(state.get("gemini_analysis") or [], state["face_metrics"], state.get("ad_analysis"))
    return {"agreement": res, "timings": _timed("agreement", t0)}


def judge(state: AgentState) -> dict:
    t0 = time.perf_counter()
    use_gemini = state.get("gemini_mode") == "real"
    try:
        v = judging.judge(state["agreement"], state["face_metrics"], state.get("gemini_analysis") or [], use_gemini)
    except Exception as e:  # noqa: BLE001 - LLM hakem düşerse kural tabanlı yedeğe geç
        v = judging.rule_judge(state["agreement"], state["face_metrics"])
        v["fallback_reason"] = repr(e)
    return {"verdicts": v, "feedback": v.get("feedback_to_analyst", ""),
            "timings": _timed("judge", t0, source=v.get("source"))}


def route_after_judge(state: AgentState) -> str:
    v = state.get("verdicts") or {}
    if (state.get("gemini_mode") == "real" and gemini_client.has_key()
            and v.get("request_reanalysis") and state.get("iteration", 0) < config.MAX_ITERATIONS):
        return "gemini_analyst"
    return "recommender"


def recommender(state: AgentState) -> dict:
    t0 = time.perf_counter()
    use_gemini = state.get("gemini_mode") == "real"
    try:
        r = judging.recommend(state["verdicts"], state.get("clicks") or [], state.get("ad_analysis"), use_gemini)
    except Exception as e:  # noqa: BLE001
        r = judging.rule_recommender(state["verdicts"], state.get("clicks") or [], state.get("ad_analysis"))
        r["fallback_reason"] = repr(e)
    return {"recommendations": r, "timings": _timed("recommender", t0, source=r.get("source"))}


def reporter(state: AgentState) -> dict:
    t0 = time.perf_counter()
    timings = list(state.get("timings") or []) + _timed("reporter", t0)
    rep = write_report({**state, "timings": timings}, Path(state["out_dir"]))
    return {"report": rep, "timings": _timed("reporter", t0)}


def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [("face_measure", face_measure), ("gemini_analyst", gemini_analyst),
                     ("agreement", agreement), ("judge", judge), ("recommender", recommender),
                     ("reporter", reporter)]:
        g.add_node(name, fn)
    g.add_edge(START, "face_measure")
    g.add_edge(START, "gemini_analyst")
    # Ayrı kenarlar: ilk turda ikisi aynı adımda bittiği için agreement bir kez çalışır;
    # hakem geri döndüğünde yalnızca gemini_analyst -> agreement tetiklenir.
    g.add_edge("face_measure", "agreement")
    g.add_edge("gemini_analyst", "agreement")
    g.add_edge("agreement", "judge")
    g.add_conditional_edges("judge", route_after_judge, ["gemini_analyst", "recommender"])
    g.add_edge("recommender", "reporter")
    g.add_edge("reporter", END)
    return g.compile()


def run(frames, session_id: str, out_dir: Path, gemini_mode: str = "real",
        clicks: list | None = None, ad_analysis: dict | None = None) -> dict:
    paths = [str(p) for p in list_frames(frames)]
    state: AgentState = {"session_id": session_id, "frames": paths, "clicks": clicks or [],
                         "ad_analysis": ad_analysis, "gemini_mode": gemini_mode, "out_dir": str(out_dir),
                         "iteration": 0, "timings": []}
    return build_graph().invoke(state)
