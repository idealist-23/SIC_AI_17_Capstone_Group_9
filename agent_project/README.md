# Agent Layer: Multi-source verification for the Ad Emotion Analyzer

This folder adds a **LangGraph agent layer** on top of the existing app in `main_project/`.
`main_project/` is **not modified**; `main.py` and `index.html` here are unchanged copies
(the agent reads `BATCH_UX_PROMPT` from `main.py` read-only). Same UI, same CSV columns, plus two confidence columns.

It stays as light as the original: **no PyTorch, no GPU, no model downloads**. Install takes seconds,
a full analysis of one session takes about one second on a laptop CPU.

## Why
The original pipeline sends five webcam frames to a single Gemini call and trusts the answer. Nothing checks it.
In early tests Gemini wrote "the viewer lost interest" for the last frame, apparently because it *was* the last frame.
The agent layer measures the face locally, lets Gemini interpret, and a judge compares the two before any
recommendation is made.

```
            ┌─ face_measure (HSEmotion-ONNX + MediaPipe) ─┐
START ──────┤                                             ├─ agreement ─ judge ─┬─ recommender ─ reporter
            └─ gemini_analyst (original BATCH_UX_PROMPT) ─┘                     │
                         ▲                                                      │
                         └───── re-analysis with judge feedback (max 2 rounds) ─┘
```

| Node | What it does |
|---|---|
| `face_measure` | **HSEmotion** (`enet_b0_8_va_mtl`, run with onnxruntime): 8 emotions + valence/arousal · **MediaPipe Face Landmarker**: head pose + iris → "looking at screen" score, blinks, and Action Unit estimates from facial blendshapes (e.g. smile → AU12, brow down → AU4) |
| `gemini_analyst` | Original prompt via the new `google-genai` SDK; on re-analysis it receives the judge's per-second feedback |
| `agreement` | Plain Python, no LLM. Per second: valence conflict (Gemini vs HSEmotion), attention conflict (Gemini says "high interest" but the viewer is not looking), AU claims without evidence |
| `judge` | Per-second verdict + 0–1 confidence + reason (Gemini, schema-validated, with rule-based fallback). Requests re-analysis when conflicts are high |
| `recommender` | Concrete edit suggestions per second. Confidence never exceeds the verdict it is based on |
| `reporter` | `result.json`, `per_second.csv`, `chart.png` under `agent_results/<session>/` |

`face_measure` and `gemini_analyst` run in parallel. Ad side: the expected viewer emotion per scene from the
TSAM model (*DecodingViewerEmotions*, Univ. of Warwick) is precomputed for `sample_ad.mp4` and shipped as a small
JSON in `agent_results/ad_cache/`, so the heavy video model is not needed at runtime.

## Setup (Python 3.10–3.12)
```bash
cd agent_project
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Gemini (optional; without a key the app still runs and reports local measurements only):
```bash
export GEMINI_API_KEY=...              # Windows: set GEMINI_API_KEY=...   (never commit the key)
export GEMINI_MODEL=gemini-2.5-flash   # default
```

## Run
Web app (same UI as the original, port 5000):
```bash
python main_agent.py      # http://127.0.0.1:5000  (on macOS use 127.0.0.1; port 5000 is shared with AirPlay)
```
CLI on the bundled test frames:
```bash
python -m agent.run --frames test_frames_tmp --mock-gemini --ad-video sample_ad.mp4   # labelled synthetic Gemini
python -m agent.run --frames test_frames_tmp --gemini real                            # with API key
```
Tests (no network needed):
```bash
python -m agent.tools.test_tools test_frames_tmp   # run first: creates inputs for the next two
python -m agent.test_agreement
python -m agent.test_validation
python -m agent.test_flask
python -m agent.test_normalize
python -m agent.test_experiments
python -m agent.test_tools_lock
```

## Honest limitations
- Not yet run end-to-end with a real Gemini key; `--mock-gemini` uses clearly labelled synthetic output.
- Single viewer, no ground truth: the judge can show that two sources disagree, not which one is right.
- The "looking at screen" score is a heuristic and not calibrated; attention recommendations are capped at 0.5 confidence.
- Blendshape-based AUs are approximations, not a trained AU detector (e.g. brow-down can fire on a smiling face).
- TSAM balanced accuracy is ~0.47 (8 classes); its output is treated as an expectation, not evidence.

Development log (Turkish): `PROGRESS.md`. Model licenses: HSEmotion (Apache-2.0), MediaPipe models (Apache-2.0).
TSAM weights are academic/non-commercial and are not included.
