"""
Vision MCP Server.
Exposes classify_food tool with dual-model confidence gating:
1. Gemini Vision (primary)
2. EfficientNet-B4 (secondary/local)
"""
import argparse
import base64
import json
import logging
import os
import sys
from io import BytesIO
from typing import Any, Dict
from PIL import Image

# Support both MCP v2 and v1
try:
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

# Ensure imports resolve whether run as module or standalone script
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .gemini_vision import classify_image_gemini
    from .classifier import classify_image_local
except ImportError:
    from gemini_vision import classify_image_gemini
    from classifier import classify_image_local

logger = logging.getLogger("vision_server")

# Instantiate MCP Server
mcp = FastMCP("vision-server")


@mcp.tool()
def classify_food(image_b64: str) -> Dict[str, Any]:
    """
    Classify a food item from a base64 encoded image using confidence gating.

    Evaluation logic:
    - If gemini_confidence >= 0.65 -> return Gemini result (method: "gemini", source: "gemini-2.0-flash", uncertain: False)
    - Elif local_confidence >= 0.65 -> return local classifier result (method: "efficientnet", source: "efficientnet-b4", uncertain: False)
    - Else -> return best result with uncertain: True flag and both predictions included in details.

    Args:
        image_b64: Base64-encoded image string.

    Returns:
        dict: {
            "label": str,
            "confidence": float,
            "top5": [{"label": str, "confidence": float}],
            "method": str,
            "source": str,
            "uncertain": bool,
            "notes": str,
            "details": dict
        }
    """
    gemini_res = classify_image_gemini(image_b64)
    gemini_conf = float(gemini_res.get("confidence", 0.0))

    # Path 1: Gemini has high confidence
    if gemini_conf >= 0.65:
        return {
            "label": gemini_res.get("label", "unknown_food"),
            "confidence": gemini_conf,
            "top5": gemini_res.get("top5", []),
            "method": "gemini",
            "source": "gemini-2.0-flash",
            "uncertain": False,
            "notes": gemini_res.get("notes", ""),
            "details": {
                "gemini": gemini_res
            }
        }

    # Path 2: Gemini confidence < 0.65, check local classifier
    local_res = classify_image_local(image_b64)
    local_conf = float(local_res.get("confidence", 0.0))

    if local_conf >= 0.65:
        return {
            "label": local_res.get("label", "unknown_food"),
            "confidence": local_conf,
            "top5": local_res.get("top5", []),
            "method": "efficientnet",
            "source": "efficientnet-b4",
            "uncertain": False,
            "notes": gemini_res.get("notes", "") or "Classified by local EfficientNet-B4 model.",
            "details": {
                "gemini": gemini_res,
                "efficientnet": local_res
            }
        }

    # Path 3: Both models < 0.65 -> return best prediction with uncertain: True
    if gemini_conf >= local_conf:
        best = gemini_res
        best_method = "gemini"
        best_source = "gemini-2.0-flash"
    else:
        best = local_res
        best_method = "efficientnet"
        best_source = "efficientnet-b4"

    return {
        "label": best.get("label", "unknown_food"),
        "confidence": float(best.get("confidence", 0.0)),
        "top5": best.get("top5", []),
        "method": best_method,
        "source": best_source,
        "uncertain": True,
        "notes": "Low classification confidence. Both models scored below confidence threshold (0.65).",
        "details": {
            "gemini": gemini_res,
            "efficientnet": local_res
        }
    }


def _create_test_image_b64() -> str:
    """Generate a small dummy JPEG image for testing."""
    img = Image.new("RGB", (100, 100), color=(210, 70, 30))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision MCP Server")
    parser.add_argument("--test", action="store_true", help="Test classify_food with a dummy image")
    args = parser.parse_args()

    if args.test:
        print("[Vision Server] Running self-test with dummy image...")
        test_b64 = _create_test_image_b64()
        res = classify_food(test_b64)
        print("[Vision Server] Test Result:")
        print(json.dumps(res, indent=2))
        print("[Vision Server] Self-test passed.")
    else:
        mcp.run()
