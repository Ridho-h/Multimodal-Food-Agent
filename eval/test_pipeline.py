"""
Comprehensive PyTest Test Suite for Multimodal Food Agent.

Covers:
- Test 1: Vision server classification and fallback handling.
- Test 2: Nutrition server USDA lookup ("apple pie").
- Test 3: Nutrition server Open Food Facts fallback ("baguette").
- Test 4: Confidence gating logic (validating uncertainty flag when confidence is low).
- Test 5: Orchestrator analyze() workflow.
- Test 6: Orchestrator chat() multi-turn session memory.
- Test 7: FastAPI endpoints (/health, /analyze, /chat, /session).
"""
import asyncio
import base64
import csv
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from api.main import app
from mcp_servers.nutrition_server.openfoodfacts import get_openfoodfacts_nutrition
from mcp_servers.nutrition_server.server import get_nutrition
from mcp_servers.nutrition_server.usda import get_usda_nutrition
from mcp_servers.vision_server.server import _create_test_image_b64, classify_food
from orchestrator.agent import FoodOrchestrator
from orchestrator.memory import SessionMemory


# --- Fixtures ---

@pytest.fixture(scope="session")
def test_image_b64() -> str:
    """Generates a reliable dummy test image in base64 format."""
    return _create_test_image_b64()


@pytest.fixture(scope="session")
def pizza_sample_b64() -> str:
    """Loads generated pizza sample image or creates one if absent."""
    sample_path = WORKSPACE_DIR / "eval" / "food_samples" / "pizza.jpg"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return _create_test_image_b64()


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def isolated_orchestrator() -> FoodOrchestrator:
    """Creates an isolated FoodOrchestrator instance with fresh memory."""
    return FoodOrchestrator(memory=SessionMemory())


# --- Test 1: Vision Server Classification and Fallback ---

def test_1_vision_classification_and_fallback(test_image_b64: str):
    """
    Test 1: Vision server classification and fallback handling.
    Verifies that classify_food accepts base64 images, executes confidence-gated
    classification, returns expected schema, and gracefully handles invalid input.
    """
    # 1. Valid test image
    result = classify_food(test_image_b64)
    assert isinstance(result, dict), "Result must be a dictionary"
    assert "label" in result, "Missing 'label' in classification"
    assert "confidence" in result, "Missing 'confidence' in classification"
    assert "top5" in result, "Missing 'top5' in classification"
    assert "method" in result, "Missing 'method' in classification"
    assert "source" in result, "Missing 'source' in classification"
    assert "uncertain" in result, "Missing 'uncertain' in classification"
    assert isinstance(result["top5"], list), "'top5' must be a list"
    assert 0.0 <= result["confidence"] <= 1.0, "Confidence must be bounded [0.0, 1.0]"

    # 2. Corrupt / invalid base64 input
    invalid_result = classify_food("invalid_base64_string_xyz_123")
    assert isinstance(invalid_result, dict)
    assert invalid_result.get("uncertain") is True or invalid_result.get("label") == "unknown_food"
    assert invalid_result.get("confidence") == 0.0


# --- Test 2: Nutrition Server USDA Lookup ---

def test_2_nutrition_usda_lookup():
    """
    Test 2: Nutrition server USDA lookup ('apple pie').
    Verifies retrieval of USDA nutrient data or graceful reference fallback,
    ensuring all core macro fields are populated and calorie counts are realistic.
    """
    food_name = "apple pie"
    result = get_nutrition(food_name, serving_size_g=100)

    assert isinstance(result, dict), "Result must be a dictionary"
    assert result["serving_size_g"] == 100
    assert "calories" in result
    assert "protein_g" in result
    assert "carbs_g" in result
    assert "fat_g" in result
    assert "fiber_g" in result
    assert "sodium_mg" in result
    assert "data_source" in result

    # Apple pie is calorie-dense (standard range 200 - 450 kcal per 100g)
    assert result["calories"] >= 150.0, f"Unexpected calories for apple pie: {result['calories']}"
    assert result["carbs_g"] >= 20.0, f"Unexpected carbs for apple pie: {result['carbs_g']}"
    assert result["data_source"] in [
        "USDA FoodData Central",
        "Open Food Facts",
        "Estimated Standard Average",
    ]


# --- Test 3: Nutrition Server Open Food Facts Fallback ---

def test_3_nutrition_openfoodfacts_fallback():
    """
    Test 3: Nutrition server Open Food Facts fallback ('baguette').
    Validates Open Food Facts parser, direct client fallback, and cascaded lookup
    confirming correct nutrient scaling and carb/protein balance for bread.
    """
    from mcp_servers.nutrition_server.openfoodfacts import _parse_off_nutrients

    # 1. Test unit parsing logic for Open Food Facts nutriments dict
    mock_nutriments = {
        "energy-kcal_100g": 289.0,
        "proteins_100g": 9.0,
        "carbohydrates_100g": 56.0,
        "fat_100g": 1.8,
        "fiber_100g": 2.7,
        "sodium_100g": 0.602,
        "sodium_unit": "g",
    }
    parsed = _parse_off_nutrients(mock_nutriments)
    assert parsed["calories"] == 289.0
    assert parsed["protein_g"] == 9.0
    assert parsed["carbs_g"] == 56.0
    assert parsed["fat_g"] == 1.8
    assert parsed["sodium_mg"] == 602.0

    # 2. Test live or cached Open Food Facts client (handles offline/503 gracefully)
    food_name = "baguette"
    off_result = get_openfoodfacts_nutrition(food_name, serving_size_g=100)
    if off_result is not None:
        assert isinstance(off_result, dict)
        assert off_result["calories"] > 0.0
        assert off_result["carbs_g"] > 0.0
        assert "Open Food Facts" in off_result["data_source"]

    # 3. Test cascaded lookup via main get_nutrition tool
    cascaded_result = get_nutrition(food_name, serving_size_g=100)
    assert cascaded_result["calories"] > 100.0, "Baguette must have > 100 kcal per 100g"
    assert cascaded_result["carbs_g"] > 20.0, "Baguette must have substantial carbohydrates"
    assert cascaded_result["data_source"] != "Not Found"


# --- Test 4: Confidence Gating Logic ---

def test_4_confidence_gating_logic(test_image_b64: str):
    """
    Test 4: Confidence gating logic.
    Validates that low-confidence predictions flag 'uncertain: True',
    and verifies that badge derivation correctly categorizes High, Moderate, and Low confidence.
    """
    # 1. Badge derivation rules
    assert FoodOrchestrator.derive_confidence_badge(0.95) == "High (🟢)"
    assert FoodOrchestrator.derive_confidence_badge(0.75) == "High (🟢)"
    assert FoodOrchestrator.derive_confidence_badge(0.74) == "Moderate (🟡)"
    assert FoodOrchestrator.derive_confidence_badge(0.50) == "Moderate (🟡)"
    assert FoodOrchestrator.derive_confidence_badge(0.49) == "Low (🔴)"
    assert FoodOrchestrator.derive_confidence_badge(0.10) == "Low (🔴)"

    # 2. Low-confidence dummy image gating
    dummy_classification = classify_food(test_image_b64)
    # Dummy non-food images should be low confidence and marked uncertain
    if dummy_classification["confidence"] < 0.65:
        assert dummy_classification["uncertain"] is True
        assert "notes" in dummy_classification
        assert "uncertain" in dummy_classification["notes"].lower() or "confidence" in dummy_classification["notes"].lower()


# --- Test 5: Orchestrator Analyze Workflow ---

def test_5_orchestrator_analyze_workflow(
    isolated_orchestrator: FoodOrchestrator,
    pizza_sample_b64: str,
):
    """
    Test 5: Orchestrator analyze() workflow.
    Validates complete end-to-end flow: image -> classification -> nutrition lookup -> dietary reasoning.
    """
    question = "Is this dish suitable for a high-protein diet?"
    result = asyncio.run(
        isolated_orchestrator.analyze(
            image_b64=pizza_sample_b64,
            question=question,
        )
    )

    assert isinstance(result, dict)
    assert "session_id" in result
    assert "answer" in result
    assert "classification" in result
    assert "nutrition" in result
    assert "confidence_badge" in result
    assert "uncertain" in result

    assert len(result["session_id"]) > 0
    assert len(result["answer"]) > 50, "Answer must provide substantial dietary feedback"
    # Mandatory disclaimer must be present
    assert "disclaimer" in result["answer"].lower() or "educational" in result["answer"].lower()

    # Verify session persisted in memory
    saved_session = isolated_orchestrator.memory.get_session(result["session_id"])
    assert saved_session is not None
    assert len(saved_session.messages) >= 2  # user + assistant


# --- Test 6: Orchestrator Chat Multi-Turn Session Memory ---

def test_6_orchestrator_chat_session_memory(
    isolated_orchestrator: FoodOrchestrator,
    pizza_sample_b64: str,
):
    """
    Test 6: Orchestrator chat() multi-turn session memory.
    Validates that follow-up questions retain contextual memory of the previously analyzed dish.
    """
    # Turn 1: Analyze food
    init_res = asyncio.run(
        isolated_orchestrator.analyze(
            image_b64=pizza_sample_b64,
            question="What food is this?",
        )
    )
    session_id = init_res["session_id"]
    detected_food = init_res["classification"].get("label")

    # Turn 2: Follow-up question relying on memory
    followup_res = asyncio.run(
        isolated_orchestrator.chat(
            question="What is the sodium content and is it heart-healthy?",
            session_id=session_id,
        )
    )

    assert followup_res["session_id"] == session_id
    assert "answer" in followup_res
    assert len(followup_res["answer"]) > 30

    # Turn 3: Second follow-up query
    turn3_res = asyncio.run(
        isolated_orchestrator.chat(
            question="How could I adjust this meal to lower the carbs?",
            session_id=session_id,
        )
    )
    assert turn3_res["session_id"] == session_id
    assert len(turn3_res["answer"]) > 30

    # Validate memory history contains all turns
    session = isolated_orchestrator.memory.get_session(session_id)
    assert session is not None
    # 3 user questions + 3 assistant responses = 6 messages
    assert len(session.messages) == 6


# --- Test 7: FastAPI Endpoints ---

def test_7_fastapi_endpoints(client: TestClient, test_image_b64: str):
    """
    Test 7: FastAPI REST endpoints.
    Verifies /health, /analyze, /chat, and /session/{session_id}.
    """
    # 1. GET /health
    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    health_json = health_resp.json()
    assert health_json["status"] in ["healthy", "ok"]
    assert "api" in health_json
    assert "vision" in health_json
    assert "nutrition" in health_json

    # 2. POST /analyze (JSON payload)
    analyze_payload = {
        "image_b64": test_image_b64,
        "question": "Can I eat this for weight loss?",
    }
    analyze_resp = client.post("/analyze", json=analyze_payload)
    assert analyze_resp.status_code == 200
    analyze_json = analyze_resp.json()
    assert "session_id" in analyze_json
    assert "answer" in analyze_json
    assert "classification" in analyze_json
    assert "nutrition" in analyze_json
    assert "confidence_badge" in analyze_json
    assert "uncertain" in analyze_json

    session_id = analyze_json["session_id"]

    # 3. POST /chat (Follow-up)
    chat_payload = {
        "session_id": session_id,
        "question": "What is the recommended serving size?",
    }
    chat_resp = client.post("/chat", json=chat_payload)
    assert chat_resp.status_code == 200
    chat_json = chat_resp.json()
    assert chat_json["session_id"] == session_id
    assert "answer" in chat_json

    # 4. GET /session/{session_id}
    session_resp = client.get(f"/session/{session_id}")
    assert session_resp.status_code == 200
    session_json = session_resp.json()
    assert session_json["session_id"] == session_id
    assert len(session_json["messages"]) >= 4  # 2 user + 2 assistant
