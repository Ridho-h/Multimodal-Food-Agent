"""
Gemini Vision Classifier for Vision MCP Server.
Leverages Google Gemini multimodal capabilities to classify foods from images.
"""
import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Preferred Gemini vision models in priority order
PREFERRED_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]


def _clean_json_response(raw_text: str) -> str:
    """Extract and clean raw JSON content from markdown code fences."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def classify_image_gemini(image_b64: str) -> Dict[str, Any]:
    """
    Classify a food item from a base64 encoded image string using Google Gemini Vision.

    Args:
        image_b64: Base64-encoded image string, optionally prefixed with
                   data URL (e.g., 'data:image/jpeg;base64,...').

    Returns:
        Dict with schema:
            {
                "label": str,
                "confidence": float,
                "top5": [{"label": str, "confidence": float}],
                "notes": str
            }
    """
    if not image_b64 or not isinstance(image_b64, str):
        return {
            "label": "unknown_food",
            "confidence": 0.0,
            "top5": [],
            "notes": "No image data provided.",
        }

    # 1. Decode base64 image gracefully
    try:
        cleaned_b64 = image_b64.strip()
        if "," in cleaned_b64:
            cleaned_b64 = cleaned_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(cleaned_b64)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        logger.warning("Failed to decode base64 image: %s", e)
        return {
            "label": "unknown_food",
            "confidence": 0.0,
            "top5": [],
            "notes": f"Failed to decode image: {e}",
        }

    # 2. Check API key
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        return {
            "label": "unknown_food",
            "confidence": 0.0,
            "top5": [],
            "notes": "GEMINI_API_KEY is not configured or missing.",
        }

    # 3. Call Gemini
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)

        prompt = (
            "Analyze this food image carefully and identify the primary dish or food item.\n"
            "Return a strictly valid JSON object with the following schema:\n"
            "{\n"
            '  "label": "<primary food name in lowercase, e.g. \'pizza\', \'apple pie\', \'caesar salad\'>",\n'
            '  "confidence": <float between 0.0 and 1.0 indicating confidence in primary label>,\n'
            '  "top5": [\n'
            '    {"label": "<candidate food 1>", "confidence": <float between 0.0 and 1.0>},\n'
            '    {"label": "<candidate food 2>", "confidence": <float between 0.0 and 1.0>},\n'
            '    {"label": "<candidate food 3>", "confidence": <float between 0.0 and 1.0>},\n'
            '    {"label": "<candidate food 4>", "confidence": <float between 0.0 and 1.0>},\n'
            '    {"label": "<candidate food 5>", "confidence": <float between 0.0 and 1.0>}\n'
            "  ],\n"
            '  "notes": "<brief description of observed ingredients, cuisine, or explanation if not food>"\n'
            "}\n"
            "If the image does not depict food, set label to 'non-food', confidence to 0.0, and describe in notes.\n"
            "Return ONLY valid JSON."
        )

        last_error = None
        for model_name in PREFERRED_MODELS:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={"response_mime_type": "application/json"}
                )
                response = model.generate_content([prompt, pil_image])
                raw_text = response.text if hasattr(response, "text") else ""
                clean_text = _clean_json_response(raw_text)
                data = json.loads(clean_text)

                label = str(data.get("label", "unknown_food")).strip().lower()
                conf = float(data.get("confidence", 0.0))
                top5_raw = data.get("top5", [])
                top5: List[Dict[str, Any]] = []

                for item in top5_raw:
                    if isinstance(item, dict) and "label" in item:
                        top5.append({
                            "label": str(item["label"]).strip().lower(),
                            "confidence": round(float(item.get("confidence", 0.0)), 4)
                        })

                if not top5 and label != "unknown_food":
                    top5 = [{"label": label, "confidence": round(conf, 4)}]

                notes = str(data.get("notes", "")).strip()

                return {
                    "label": label,
                    "confidence": round(conf, 4),
                    "top5": top5[:5],
                    "notes": notes,
                }
            except Exception as err:
                last_error = err
                logger.debug("Model %s failed: %s", model_name, err)
                continue

        return {
            "label": "unknown_food",
            "confidence": 0.0,
            "top5": [],
            "notes": f"Gemini vision call failed: {last_error}",
        }

    except Exception as e:
        logger.error("Gemini vision initialization error: %s", e)
        return {
            "label": "unknown_food",
            "confidence": 0.0,
            "top5": [],
            "notes": f"Gemini vision error: {str(e)}",
        }
