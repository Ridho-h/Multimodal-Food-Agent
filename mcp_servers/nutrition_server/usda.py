"""
USDA FoodData Central API client.
Queries api.nal.usda.gov to retrieve nutritional breakdown for food items.
Includes LRU caching and robust fallback handling.
"""
import copy
import functools
import logging
import os
from typing import Any, Dict, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"


def _extract_nutrients(food_nutrients: list) -> Dict[str, float]:
    """Extract standard macro/micronutrients from USDA foodNutrients list."""
    calories: Optional[float] = None
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0
    sodium_mg: float = 0.0

    for item in food_nutrients:
        name = (item.get("nutrientName") or "").lower()
        unit = (item.get("unitName") or "").lower()
        val = item.get("value")
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue

        if "energy" in name:
            if unit == "kcal":
                calories = val
            elif unit == "kj" and calories is None:
                calories = val / 4.184
        elif name.startswith("protein"):
            protein_g = val
        elif "carbohydrate" in name:
            carbs_g = val
        elif "total lipid" in name or name == "fat":
            fat_g = val
        elif "fiber" in name:
            fiber_g = val
        elif "sodium" in name:
            if unit == "g":
                sodium_mg = val * 1000.0
            else:
                sodium_mg = val

    if calories is None:
        # Standard Atwater approximation: 4 * protein + 4 * carbs + 9 * fat
        calories = (protein_g * 4.0) + (carbs_g * 4.0) + (fat_g * 9.0)

    return {
        "calories": calories,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
        "fiber_g": fiber_g,
        "sodium_mg": sodium_mg,
    }


@functools.lru_cache(maxsize=256)
def _fetch_usda_cached(food_name: str, serving_size_g: int) -> Optional[Dict[str, Any]]:
    """Internal cached helper for USDA queries."""
    api_key = os.getenv("USDA_API_KEY", "").strip() or "DEMO_KEY"

    params = {
        "query": food_name,
        "pageSize": 5,
        "api_key": api_key,
    }

    try:
        response = requests.get(USDA_API_URL, params=params, timeout=8)
        if response.status_code != 200:
            logger.warning("USDA API returned status %d: %s", response.status_code, response.text[:200])
            return None

        data = response.json()
        foods = data.get("foods", [])
        if not foods:
            logger.info("No USDA food records found for '%s'", food_name)
            return None

        # Select top matched food
        top_food = foods[0]
        description = top_food.get("description", food_name)
        nutrients_100g = _extract_nutrients(top_food.get("foodNutrients", []))

        # USDA values are reported per 100g -> scale to serving_size_g
        scale = max(0.0, serving_size_g) / 100.0

        return {
            "food_name": description,
            "query": food_name,
            "serving_size_g": serving_size_g,
            "calories": round(nutrients_100g["calories"] * scale, 1),
            "protein_g": round(nutrients_100g["protein_g"] * scale, 1),
            "carbs_g": round(nutrients_100g["carbs_g"] * scale, 1),
            "fat_g": round(nutrients_100g["fat_g"] * scale, 1),
            "fiber_g": round(nutrients_100g["fiber_g"] * scale, 1),
            "sodium_mg": round(nutrients_100g["sodium_mg"] * scale, 1),
            "data_source": "USDA FoodData Central",
        }

    except Exception as e:
        logger.warning("Error fetching USDA nutrition for '%s': %s", food_name, e)
        return None


def get_usda_nutrition(food_name: str, serving_size_g: int = 100) -> Optional[Dict[str, Any]]:
    """
    Retrieve nutritional facts for a given food name from USDA FoodData Central.

    Args:
        food_name: Common name of the food item (e.g. "apple pie", "pizza").
        serving_size_g: Serving size in grams (defaults to 100g).

    Returns:
        Dict with keys (calories, protein_g, carbs_g, fat_g, fiber_g, sodium_mg, data_source),
        or None if not found or query fails.
    """
    if not food_name or not food_name.strip():
        return None

    clean_name = food_name.strip().lower()
    res = _fetch_usda_cached(clean_name, serving_size_g)
    return copy.deepcopy(res) if res is not None else None
