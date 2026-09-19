"""
Open Food Facts API client.
Provides fallback nutrition search when USDA has no match or is unavailable.
"""
import copy
import functools
import json
import logging
from typing import Any, Dict, Optional
import requests

logger = logging.getLogger(__name__)

OFF_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
USER_AGENT = "MultimodalFoodAgent - Python - Version 1.0 (contact@example.com)"


def _parse_off_nutrients(nutriments: dict) -> Dict[str, float]:
    """Extract standard macro/micronutrients from Open Food Facts nutriments dict per 100g."""
    # Calories (kcal)
    calories = nutriments.get("energy-kcal_100g")
    if calories is None:
        calories = nutriments.get("energy-kcal")
    if calories is None and nutriments.get("energy_100g"):
        # If energy is in kJ
        try:
            calories = float(nutriments["energy_100g"]) / 4.184
        except (ValueError, TypeError):
            calories = None

    def _to_float(val: Any) -> float:
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    protein_g = _to_float(nutriments.get("proteins_100g", nutriments.get("proteins")))
    carbs_g = _to_float(nutriments.get("carbohydrates_100g", nutriments.get("carbohydrates")))
    fat_g = _to_float(nutriments.get("fat_100g", nutriments.get("fat")))
    fiber_g = _to_float(nutriments.get("fiber_100g", nutriments.get("fiber")))

    # Sodium (OFF usually reports sodium in grams)
    sodium_val = _to_float(nutriments.get("sodium_100g", nutriments.get("sodium")))
    unit = str(nutriments.get("sodium_unit", "g")).lower()
    if unit == "g":
        sodium_mg = sodium_val * 1000.0
    else:
        sodium_mg = sodium_val

    if calories is None:
        calories = (protein_g * 4.0) + (carbs_g * 4.0) + (fat_g * 9.0)
    else:
        calories = _to_float(calories)

    return {
        "calories": calories,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
        "fiber_g": fiber_g,
        "sodium_mg": sodium_mg,
    }


@functools.lru_cache(maxsize=256)
def _fetch_openfoodfacts_cached(food_name: str, serving_size_g: int) -> Optional[Dict[str, Any]]:
    """Internal cached helper for Open Food Facts queries."""
    params = {
        "search_terms": food_name,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 5,
        "fields": "product_name,generic_name,nutriments",
    }
    headers = {
        "User-Agent": USER_AGENT
    }

    try:
        response = requests.get(OFF_SEARCH_URL, params=params, headers=headers, timeout=8)
        if response.status_code != 200:
            logger.debug("Open Food Facts status %d for '%s'", response.status_code, food_name)
            return None

        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            return None

        products = data.get("products", [])
        if not products:
            return None

        # Find first product with non-empty nutriments
        target_product = None
        for p in products:
            nutriments = p.get("nutriments", {})
            if nutriments and any(
                k in nutriments for k in ["energy-kcal_100g", "energy_100g", "proteins_100g", "fat_100g"]
            ):
                target_product = p
                break

        if not target_product:
            target_product = products[0]

        product_name = (
            target_product.get("product_name")
            or target_product.get("generic_name")
            or food_name
        )
        nutrients_100g = _parse_off_nutrients(target_product.get("nutriments", {}))
        scale = max(0.0, serving_size_g) / 100.0

        return {
            "food_name": product_name,
            "query": food_name,
            "serving_size_g": serving_size_g,
            "calories": round(nutrients_100g["calories"] * scale, 1),
            "protein_g": round(nutrients_100g["protein_g"] * scale, 1),
            "carbs_g": round(nutrients_100g["carbs_g"] * scale, 1),
            "fat_g": round(nutrients_100g["fat_g"] * scale, 1),
            "fiber_g": round(nutrients_100g["fiber_g"] * scale, 1),
            "sodium_mg": round(nutrients_100g["sodium_mg"] * scale, 1),
            "data_source": "Open Food Facts",
        }

    except Exception as e:
        logger.debug("Error querying Open Food Facts for '%s': %s", food_name, e)
        return None


def get_openfoodfacts_nutrition(food_name: str, serving_size_g: int = 100) -> Optional[Dict[str, Any]]:
    """
    Retrieve nutritional facts for a food item from Open Food Facts API.

    Args:
        food_name: Food name or product query.
        serving_size_g: Serving size in grams (defaults to 100g).

    Returns:
        Dict with keys (calories, protein_g, carbs_g, fat_g, fiber_g, sodium_mg, data_source),
        or None if not found or query fails.
    """
    if not food_name or not food_name.strip():
        return None

    clean_name = food_name.strip().lower()
    res = _fetch_openfoodfacts_cached(clean_name, serving_size_g)
    return copy.deepcopy(res) if res is not None else None
