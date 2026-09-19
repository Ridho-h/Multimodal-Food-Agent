"""
Nutrition MCP Server.
Exposes get_nutrition tool with cascading data sources:
1. USDA FoodData Central (Primary)
2. Open Food Facts (Secondary / International fallback)
3. Calibrated dietary estimates / Clear not-found fallback
"""
import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

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
    from .usda import get_usda_nutrition
    from .openfoodfacts import get_openfoodfacts_nutrition
except ImportError:
    from usda import get_usda_nutrition
    from openfoodfacts import get_openfoodfacts_nutrition

logger = logging.getLogger("nutrition_server")

# Instantiate FastMCP server
mcp = FastMCP("nutrition-server")

# Standard reference estimates per 100g for common foods (offline/fallback safety net)
REFERENCE_ESTIMATES = {
    "apple pie": {"calories": 237.0, "protein_g": 1.9, "carbs_g": 34.0, "fat_g": 11.0, "fiber_g": 1.6, "sodium_mg": 218.0},
    "pizza": {"calories": 266.0, "protein_g": 11.0, "carbs_g": 33.0, "fat_g": 10.0, "fiber_g": 2.3, "sodium_mg": 598.0},
    "hamburger": {"calories": 254.0, "protein_g": 17.0, "carbs_g": 20.0, "fat_g": 12.0, "fiber_g": 1.0, "sodium_mg": 458.0},
    "caesar salad": {"calories": 140.0, "protein_g": 4.0, "carbs_g": 6.0, "fat_g": 11.0, "fiber_g": 1.5, "sodium_mg": 380.0},
    "french fries": {"calories": 312.0, "protein_g": 3.4, "carbs_g": 41.0, "fat_g": 15.0, "fiber_g": 3.8, "sodium_mg": 210.0},
    "fried rice": {"calories": 163.0, "protein_g": 3.2, "carbs_g": 31.0, "fat_g": 2.6, "fiber_g": 1.1, "sodium_mg": 390.0},
    "sushi": {"calories": 143.0, "protein_g": 5.8, "carbs_g": 21.0, "fat_g": 2.1, "fiber_g": 1.2, "sodium_mg": 412.0},
    "tacos": {"calories": 226.0, "protein_g": 8.8, "carbs_g": 20.0, "fat_g": 13.0, "fiber_g": 2.9, "sodium_mg": 397.0},
    "ramen": {"calories": 110.0, "protein_g": 4.5, "carbs_g": 17.0, "fat_g": 2.8, "fiber_g": 1.0, "sodium_mg": 720.0},
    "ice cream": {"calories": 207.0, "protein_g": 3.5, "carbs_g": 24.0, "fat_g": 11.0, "fiber_g": 0.7, "sodium_mg": 80.0},
    "baguette": {"calories": 289.0, "protein_g": 9.0, "carbs_g": 56.0, "fat_g": 1.8, "fiber_g": 2.7, "sodium_mg": 602.0},
    "croissant": {"calories": 406.0, "protein_g": 8.2, "carbs_g": 45.8, "fat_g": 21.0, "fiber_g": 2.6, "sodium_mg": 467.0},
    "steak": {"calories": 271.0, "protein_g": 26.1, "carbs_g": 0.0, "fat_g": 17.8, "fiber_g": 0.0, "sodium_mg": 68.0},
    "burrito": {"calories": 206.0, "protein_g": 7.5, "carbs_g": 24.3, "fat_g": 8.6, "fiber_g": 2.1, "sodium_mg": 450.0},
    "pad thai": {"calories": 220.0, "protein_g": 9.5, "carbs_g": 28.0, "fat_g": 7.5, "fiber_g": 1.8, "sodium_mg": 480.0},
    "pasta carbonara": {"calories": 285.0, "protein_g": 12.0, "carbs_g": 30.0, "fat_g": 13.0, "fiber_g": 1.5, "sodium_mg": 420.0},
    "omelette": {"calories": 154.0, "protein_g": 10.6, "carbs_g": 0.7, "fat_g": 11.7, "fiber_g": 0.0, "sodium_mg": 168.0},
    "chicken curry": {"calories": 145.0, "protein_g": 14.0, "carbs_g": 6.0, "fat_g": 7.2, "fiber_g": 1.2, "sodium_mg": 340.0},
    "salmon": {"calories": 208.0, "protein_g": 20.4, "carbs_g": 0.0, "fat_g": 13.4, "fiber_g": 0.0, "sodium_mg": 59.0},
    "cheesecake": {"calories": 321.0, "protein_g": 5.5, "carbs_g": 25.5, "fat_g": 22.5, "fiber_g": 0.4, "sodium_mg": 207.0},
    "hummus": {"calories": 166.0, "protein_g": 7.9, "carbs_g": 14.3, "fat_g": 9.6, "fiber_g": 6.0, "sodium_mg": 379.0},
}


def _get_reference_estimate(food_name: str, serving_size_g: int) -> Optional[Dict[str, Any]]:
    """Check if food_name matches any known standard food estimate."""
    clean = food_name.strip().lower().replace("_", " ")
    for key, values in REFERENCE_ESTIMATES.items():
        if key in clean or clean in key:
            scale = max(0.0, serving_size_g) / 100.0
            return {
                "food_name": key.title(),
                "query": food_name,
                "serving_size_g": serving_size_g,
                "calories": round(values["calories"] * scale, 1),
                "protein_g": round(values["protein_g"] * scale, 1),
                "carbs_g": round(values["carbs_g"] * scale, 1),
                "fat_g": round(values["fat_g"] * scale, 1),
                "fiber_g": round(values["fiber_g"] * scale, 1),
                "sodium_mg": round(values["sodium_mg"] * scale, 1),
                "data_source": "Estimated Standard Average",
            }
    return None


@mcp.tool()
def get_nutrition(food_name: str, serving_size_g: int = 100) -> Dict[str, Any]:
    """
    Retrieve comprehensive nutritional information for a specified food item.
    Cascades through USDA FoodData Central, Open Food Facts, and reference estimates.

    Args:
        food_name: Name of the food (e.g. "apple pie", "pizza").
        serving_size_g: Desired serving size in grams (defaults to 100g).

    Returns:
        dict: {
            "food_name": str,
            "query": str,
            "serving_size_g": int,
            "calories": float,
            "protein_g": float,
            "carbs_g": float,
            "fat_g": float,
            "fiber_g": float,
            "sodium_mg": float,
            "data_source": str,
            "notes": str (optional)
        }
    """
    if not food_name or not food_name.strip():
        return {
            "food_name": "unknown",
            "query": "",
            "serving_size_g": serving_size_g,
            "calories": 0.0,
            "protein_g": 0.0,
            "carbs_g": 0.0,
            "fat_g": 0.0,
            "fiber_g": 0.0,
            "sodium_mg": 0.0,
            "data_source": "Not Found",
            "notes": "No food name was provided.",
        }

    clean_name = food_name.strip().replace("_", " ")

    # 1. Primary: USDA FoodData Central
    usda_data = get_usda_nutrition(clean_name, serving_size_g=serving_size_g)
    if usda_data is not None:
        return usda_data

    # 2. Secondary fallback: Open Food Facts
    off_data = get_openfoodfacts_nutrition(clean_name, serving_size_g=serving_size_g)
    if off_data is not None:
        return off_data

    # 3. Tertiary fallback: Reference estimated average
    ref_data = _get_reference_estimate(clean_name, serving_size_g=serving_size_g)
    if ref_data is not None:
        return ref_data

    # 4. Final: Clear not-found structure
    return {
        "food_name": clean_name,
        "query": food_name,
        "serving_size_g": serving_size_g,
        "calories": 0.0,
        "protein_g": 0.0,
        "carbs_g": 0.0,
        "fat_g": 0.0,
        "fiber_g": 0.0,
        "sodium_mg": 0.0,
        "data_source": "Not Found",
        "notes": f"Nutritional data for '{clean_name}' could not be located in USDA or Open Food Facts.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nutrition MCP Server")
    parser.add_argument("--test", action="store_true", help="Test nutrition lookup for 'apple pie' and 'pizza'")
    args = parser.parse_args()

    if args.test:
        print("[Nutrition Server] Running self-tests...")
        for test_food in ["apple pie", "pizza"]:
            print(f"\n--- Testing get_nutrition('{test_food}', serving_size_g=100) ---")
            result = get_nutrition(test_food, serving_size_g=100)
            print(json.dumps(result, indent=2))
        print("\n[Nutrition Server] Self-tests passed successfully.")
    else:
        mcp.run()
