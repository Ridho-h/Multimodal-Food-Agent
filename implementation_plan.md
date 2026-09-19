# Multimodal Food Agent — Implementation Plan

A production-quality multi-agent system that identifies food from a photo, retrieves
nutrition data, and answers dietary questions — combining a custom CV classifier with
LLM-powered reasoning through the Model Context Protocol (MCP).

---

## Improvements over the Original Plan

> [!IMPORTANT]
> These improvements are recommended additions / substitutions to the 7-step plan.

| # | Original Approach | Improved Approach | Why |
|---|---|---|---|
| **1** | Wrap a custom food classifier model | Use **Google Gemini 2.0 Flash** (multimodal) as the *primary* vision tool, with an optional EfficientNet-B4 fine-tuned on Food-101 as a *secondary* classifier | Gemini's native vision eliminates model loading complexity, handles mixed dishes, non-English foods, and gives calibrated confidence. The custom PyTorch model remains as a portfolio CV artifact — Gemini falls back to it if needed |
| **2** | Single nutrition API | **USDA FoodData Central** (primary, free, no hard limits) + **Open Food Facts** (fallback, fully open, 3M+ products) | Single API failure no longer blocks the pipeline |
| **3** | Simple orchestrator | **Google ADK** (Agent Development Kit) orchestrator with typed tool calls | ADK manages retries, tool schema validation, streaming, and conversation memory natively |
| **4** | Confidence check as a note | Confidence-gated dual-path: if Gemini confidence < 0.65 → confirm with EfficientNet; if both low → show honest uncertainty card in the UI | Two independent signals make low-confidence handling more robust |
| **5** | Manual eval with 15-20 photos | Structured eval harness with `pytest` + a CSV of expected labels; logs accuracy, latency, nutrition hit-rate per food category | Reproducible, runnable with one command |
| **6** | Streamlit OR Gradio | **Gradio** for the demo UI (native multimodal `ChatInterface`, HuggingFace Spaces deploy) **+** a separate **FastAPI** REST endpoint for programmatic access | Broader audience: both human users and API consumers |
| **7** | Plain README | README + auto-generated architecture SVG via `diagrams` library + Mermaid diagram embedded in the doc | Visual assets make GitHub repos stand out |

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│              Gradio UI                  │
│   Image upload + chat follow-ups        │
└──────────────┬──────────────────────────┘
               │ HTTP (REST or WebSocket)
               ▼
┌─────────────────────────────────────────┐
│           FastAPI Gateway               │
│  /analyze  /chat  /health               │
└──────────────┬──────────────────────────┘
               │ ADK agent call
               ▼
┌─────────────────────────────────────────────────────┐
│              Orchestrator Agent (ADK)               │
│  - Manages tool call sequence                       │
│  - Confidence gating logic                          │
│  - Multi-turn conversation memory                   │
│  - Dietary reasoning (Gemini 2.0 Flash)             │
└────────┬──────────────────────┬─────────────────────┘
         │                      │
         ▼                      ▼
┌─────────────────┐   ┌──────────────────────┐
│  Vision MCP     │   │  Nutrition MCP        │
│  Server         │   │  Server               │
│  ─────────────  │   │  ──────────────────   │
│  classify_food  │   │  get_nutrition        │
│  (image_b64)    │   │  (food_name)          │
│                 │   │                       │
│  Gemini Vision  │   │  USDA FoodData API    │
│  + EfficientNet │   │  Open Food Facts API  │
│  fallback       │   │  (fallback)           │
└─────────────────┘   └──────────────────────┘
```

---

## Project Structure

```
Multimodal Food Agent/
├── mcp_servers/
│   ├── vision_server/
│   │   ├── server.py          # FastMCP server exposing classify_food()
│   │   ├── classifier.py      # EfficientNet-B4 inference wrapper
│   │   ├── gemini_vision.py   # Gemini 2.0 Flash vision calls
│   │   └── models/            # .pt checkpoint (downloaded on first run)
│   └── nutrition_server/
│       ├── server.py          # FastMCP server exposing get_nutrition()
│       ├── usda.py            # USDA FoodData Central client
│       └── openfoodfacts.py   # Open Food Facts fallback client
├── orchestrator/
│   ├── agent.py               # ADK agent definition + tool registration
│   ├── prompts.py             # System prompts, confidence thresholds
│   └── memory.py              # Conversation session state
├── api/
│   └── main.py                # FastAPI gateway (REST endpoints)
├── ui/
│   └── app.py                 # Gradio ChatInterface UI
├── eval/
│   ├── test_pipeline.py       # pytest end-to-end eval
│   ├── food_samples/          # 20 diverse test images
│   └── expected_labels.csv    # Ground truth for eval
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Proposed Changes (Detailed)

### Component 1 — Vision MCP Server

#### [NEW] `mcp_servers/vision_server/server.py`
FastMCP server. Exposes one tool:
```python
classify_food(image_b64: str) -> ClassificationResult
```
Returns: `{ label, confidence, top5, method, source }`.
- `method`: `"gemini"` | `"efficientnet"` | `"gemini+efficientnet"`
- `source`: which model answered (for transparency in UI)

#### [NEW] `mcp_servers/vision_server/gemini_vision.py`
Uses `google-generativeai` SDK. Sends the image to `gemini-2.0-flash` with a structured
JSON output prompt requesting top-5 food labels + per-label confidence.

#### [NEW] `mcp_servers/vision_server/classifier.py`
EfficientNet-B4 fine-tuned on Food-101 (101-class).
- Downloads weights from Hugging Face Hub on first run (`nateraw/food`)
- Returns softmax probabilities for top-5 classes

**Confidence gating logic (inside server.py):**
```
if gemini_confidence >= 0.65  →  return gemini result
elif efficientnet_confidence >= 0.65  →  return efficientnet result
else  →  return both with honest uncertainty flag
```

---

### Component 2 — Nutrition MCP Server

#### [NEW] `mcp_servers/nutrition_server/server.py`
FastMCP server. Exposes one tool:
```python
get_nutrition(food_name: str, serving_size_g: int = 100) -> NutritionResult
```
Returns: `{ calories, protein_g, carbs_g, fat_g, fiber_g, sodium_mg, data_source }`.

#### [NEW] `mcp_servers/nutrition_server/usda.py`
Calls `api.nal.usda.gov/fdc/v1/foods/search`. Returns best-match food's nutrients.
Caches results in-memory with `functools.lru_cache` to avoid repeated API hits.

#### [NEW] `mcp_servers/nutrition_server/openfoodfacts.py`
Calls `world.openfoodfacts.org/cgi/search.pl`. Used as fallback when USDA returns no
results (common for non-US or ethnic food names).

---

### Component 3 — Orchestrator Agent (ADK)

#### [NEW] `orchestrator/agent.py`
Google ADK `LlmAgent` that:
1. Receives `(image_b64, user_question)` as input
2. Calls `classify_food` tool → gets classification result
3. Checks confidence; if uncertain, includes uncertainty preamble
4. Calls `get_nutrition` tool with the top label
5. Synthesizes both results + answers `user_question` using Gemini reasoning
6. Returns structured response: `{ answer, classification, nutrition, uncertainty }`

**ADK tool registration:**
```python
from google.adk.tools.mcp_tool import McpToolset

vision_tools = McpToolset(connection_params=StdioServerParameters(
    command="python", args=["mcp_servers/vision_server/server.py"]
))
nutrition_tools = McpToolset(connection_params=StdioServerParameters(
    command="python", args=["mcp_servers/nutrition_server/server.py"]
))
```

#### [NEW] `orchestrator/prompts.py`
System prompt that instructs the LLM to:
- Be honest about classification uncertainty
- Always ground dietary advice in the nutrition numbers (not hallucinate)
- Disclaim it's not a medical tool

---

### Component 4 — FastAPI Gateway

#### [NEW] `api/main.py`
Two endpoints:
- `POST /analyze` — accepts image file + question, runs full agent pipeline, returns JSON
- `POST /chat` — follow-up questions within a session (uses session memory)
- `GET /health` — liveness probe for Docker/k8s

---

### Component 5 — Gradio UI

#### [NEW] `ui/app.py`
`gr.ChatInterface` with `multimodal=True`:
- Left panel: image upload with preview
- Right panel: chat history
- Below image: **Nutrition Card** (rendered as a table after first analysis)
- Shows confidence badge: 🟢 High / 🟡 Moderate / 🔴 Low

---

### Component 6 — Eval Harness

#### [NEW] `eval/test_pipeline.py`
Runs 20 food images through the pipeline, compares predicted label to ground truth, and
prints a summary table:
```
Food              Expected        Got             Conf   Nutrition OK
apple_pie         apple pie       apple pie       0.92   ✅
bibimbap          bibimbap        mixed rice      0.51   ✅  ← low-conf flagged
...
```

---

### Component 7 — Docker + Deployment

#### [NEW] `Dockerfile`
Multi-stage build:
- Stage 1: install dependencies, download EfficientNet weights
- Stage 2: run FastAPI + Gradio on port 7860

#### [NEW] `docker-compose.yml`
Starts 3 services:
- `vision-mcp` — vision MCP server (stdio via subprocess)
- `nutrition-mcp` — nutrition MCP server
- `app` — FastAPI + Gradio gateway

**HuggingFace Spaces deploy:** `Dockerfile` is already HF Spaces compatible (port 7860).
User can `git push` the repo to a new HF Space.

---

## Open Questions

> [!IMPORTANT]
> **API Keys Required** — Before starting, please confirm:
> 1. Do you have a **Google Gemini API key**? (free tier: 15 req/min) — Get at [aistudio.google.com](https://aistudio.google.com)
> 2. Do you have a **USDA FoodData Central API key**? (free, instant signup) — Get at [fdc.nal.usda.gov/api-key-signup.html](https://fdc.nal.usda.gov/api-key-signup.html)
> 3. Open Food Facts requires **no key** (public API) ✅

> [!NOTE]
> **EfficientNet Model Strategy**: The plan uses a *pre-trained HuggingFace checkpoint* (`nateraw/food`) fine-tuned on Food-101 so you don't need to train from scratch. If you already have your own `.pt` file, just point `classifier.py` to it.

> [!NOTE]
> **Deployment Target**: The Dockerfile targets HuggingFace Spaces (port 7860, free tier). If you prefer a different platform (Railway, Render, Fly.io, GCP Cloud Run), the setup is similar but `docker-compose.yml` would change slightly. Let me know your preference.

---

## Verification Plan

### Automated Tests
```bash
# Unit test MCP tools in isolation
pytest eval/test_pipeline.py -v

# Check MCP server starts correctly
python mcp_servers/vision_server/server.py --test
python mcp_servers/nutrition_server/server.py --test
```

### Manual Verification
1. Start the app: `docker-compose up` or `python ui/app.py`
2. Upload a photo of pizza → confirm classification + nutrition card appears
3. Ask "is this good for a low-carb diet?" → verify grounded answer with carb numbers
4. Upload an ambiguous photo (mixed dish) → confirm uncertainty card appears
5. Check the `/health` endpoint responds 200

---

## Technology Stack Summary

| Layer | Technology |
|---|---|
| Vision (primary) | Google Gemini 2.0 Flash (multimodal) |
| Vision (fallback/CV portfolio) | EfficientNet-B4 fine-tuned on Food-101 |
| Nutrition API (primary) | USDA FoodData Central (free) |
| Nutrition API (fallback) | Open Food Facts (free, no key) |
| MCP Framework | FastMCP (Python) |
| Agent Orchestrator | Google ADK (`google-adk`) |
| UI | Gradio `ChatInterface` |
| API Gateway | FastAPI |
| Containerization | Docker + docker-compose |
| Deployment | HuggingFace Spaces (or Cloud Run) |
| Eval | pytest + CSV fixtures |
