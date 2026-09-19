"""
Food Orchestrator Agent.

Orchestrates the multimodal food analysis pipeline:
1. Calls Vision MCP server (Gemini Vision / EfficientNet-B4 confidence-gated classifier).
2. Calls Nutrition MCP server (USDA FoodData Central / Open Food Facts / reference estimates).
3. Synthesizes dietary reasoning using Google Gemini models, with intelligent rule-based
   fallback if the Gemini API is offline or unconfigured.
4. Manages multi-turn conversation memory.
"""
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from mcp_servers.nutrition_server.server import get_nutrition
from mcp_servers.vision_server.server import classify_food
from orchestrator.memory import SessionMemory, default_session_memory
from orchestrator.prompts import (
    MEDICAL_DISCLAIMER,
    SYSTEM_PROMPT,
    build_rule_based_analysis,
    build_rule_based_chat_answer,
    format_analysis_prompt,
    format_chat_prompt,
)

load_dotenv()
logger = logging.getLogger("orchestrator.agent")

PREFERRED_GEMINI_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]


class FoodOrchestrator:
    """
    Multimodal Food Intelligence Orchestrator.
    Combines computer vision, nutritional knowledge retrieval, and LLM reasoning.
    """

    def __init__(
        self,
        memory: Optional[SessionMemory] = None,
        gemini_api_key: Optional[str] = None,
    ) -> None:
        self.memory = memory or default_session_memory
        self.api_key = (
            gemini_api_key
            or os.getenv("GEMINI_API_KEY", "")
        ).strip()

        if self.api_key == "your_gemini_api_key_here":
            self.api_key = ""

        self._genai_configured = False
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._genai_configured = True
                logger.info("FoodOrchestrator configured with Gemini API key.")
            except Exception as e:
                logger.warning("Could not initialize google.generativeai: %s", e)

    @staticmethod
    def derive_confidence_badge(confidence: float) -> str:
        """
        Derives human-readable confidence badge:
        - >= 0.75 -> 'High (🟢)'
        - >= 0.50 -> 'Moderate (🟡)'
        - < 0.50  -> 'Low (🔴)'
        """
        if confidence >= 0.75:
            return "High (🟢)"
        elif confidence >= 0.50:
            return "Moderate (🟡)"
        else:
            return "Low (🔴)"

    def _call_gemini_sync(self, prompt: str) -> Optional[str]:
        """Synchronously invoke Gemini API through priority list of models."""
        if not self._genai_configured or not self.api_key:
            return None

        try:
            import google.generativeai as genai
        except ImportError:
            return None

        rate_limited = False
        for model_name in PREFERRED_GEMINI_MODELS:
            try:
                try:
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=SYSTEM_PROMPT,
                    )
                    resp = model.generate_content(prompt)
                except (TypeError, ValueError):
                    # Fallback if system_instruction parameter is unsupported in model/version
                    model = genai.GenerativeModel(model_name=model_name)
                    combined_prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{prompt}"
                    resp = model.generate_content(combined_prompt)

                if resp and hasattr(resp, "text") and resp.text:
                    return resp.text.strip()
            except Exception as err:
                err_str = str(err)
                if "ResourceExhausted" in type(err).__name__ or "429" in err_str or "quota" in err_str.lower():
                    rate_limited = True
                    logger.warning("Gemini model %s reached rate limit (429). Attempting next model in pool...", model_name)
                else:
                    logger.debug("Gemini model %s failed in orchestrator: %s", model_name, err)
                continue

        if rate_limited:
            logger.warning("All Gemini models reached rate limits. Serving verified nutritional data.")
        else:
            logger.warning("All preferred Gemini models failed.")
        return None

    async def _generate_llm_analysis(
        self,
        food_name: str,
        confidence: float,
        nutrition: Dict[str, Any],
        question: Optional[str] = None,
        uncertain: bool = False,
        top5: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generates dietary analysis via Gemini with rule-based fallback."""
        if self._genai_configured:
            prompt = format_analysis_prompt(
                food_name=food_name,
                confidence=confidence,
                nutrition=nutrition,
                question=question,
                uncertain=uncertain,
                top5=top5,
            )
            gemini_response = await asyncio.to_thread(self._call_gemini_sync, prompt)
            if gemini_response:
                # Ensure disclaimer is included if model forgot it
                if "Disclaimer" not in gemini_response and "disclaimer" not in gemini_response:
                    gemini_response = f"{gemini_response}\n\n{MEDICAL_DISCLAIMER}"
                return gemini_response

        # Fallback to rich rule-based analysis
        return build_rule_based_analysis(
            food_name=food_name,
            confidence=confidence,
            nutrition=nutrition,
            question=question,
            uncertain=uncertain,
            top5=top5,
        )

    async def _generate_llm_chat(
        self,
        question: str,
        food_name: Optional[str],
        nutrition: Optional[Dict[str, Any]],
        history: List[Dict[str, str]],
    ) -> str:
        """Generates chat answer via Gemini with rule-based fallback."""
        if self._genai_configured:
            prompt = format_chat_prompt(
                question=question,
                food_name=food_name,
                nutrition=nutrition,
                history=history,
            )
            gemini_response = await asyncio.to_thread(self._call_gemini_sync, prompt)
            if gemini_response:
                return gemini_response

        # Fallback
        return build_rule_based_chat_answer(
            question=question,
            food_name=food_name,
            nutrition=nutrition,
        )

    async def analyze(
        self,
        image_b64: str,
        question: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a food image:
        1. Classify food using Vision MCP server (Gemini Vision + local fallback).
        2. Query nutrition facts from Nutrition MCP server (USDA + Open Food Facts).
        3. Derive confidence badge and uncertainty status.
        4. Synthesize dietary reasoning and answer any specific user questions.
        5. Persist the turn in session memory.

        Args:
            image_b64: Base64-encoded image.
            question: Optional user dietary inquiry.
            session_id: Optional session identifier.

        Returns:
            Dict: {
                "session_id": str,
                "answer": str,
                "classification": dict,
                "nutrition": dict,
                "confidence_badge": str,
                "uncertain": bool
            }
        """
        session = self.memory.get_or_create_session(session_id)
        current_session_id = session.session_id

        # 1. Vision classification
        try:
            classification = await asyncio.to_thread(classify_food, image_b64)
        except Exception as e:
            logger.error("classify_food error: %s", e)
            classification = {
                "label": "unknown_food",
                "confidence": 0.0,
                "top5": [],
                "method": "error",
                "source": "error_handler",
                "uncertain": True,
                "notes": f"Classification error: {e}",
                "details": {},
            }

        food_label = str(classification.get("label", "unknown_food")).strip()
        confidence = float(classification.get("confidence", 0.0))
        top5 = classification.get("top5", [])
        uncertain = bool(classification.get("uncertain", False) or confidence < 0.65)

        # 2. Nutrition retrieval
        try:
            nutrition = await asyncio.to_thread(get_nutrition, food_label)
        except Exception as e:
            logger.error("get_nutrition error: %s", e)
            nutrition = {
                "food_name": food_label,
                "query": food_label,
                "serving_size_g": 100,
                "calories": 0.0,
                "protein_g": 0.0,
                "carbs_g": 0.0,
                "fat_g": 0.0,
                "fiber_g": 0.0,
                "sodium_mg": 0.0,
                "data_source": "Not Found",
                "notes": f"Nutrition lookup error: {e}",
            }

        # 3. Derive confidence badge
        confidence_badge = self.derive_confidence_badge(confidence)

        # 4. Synthesize dietary reasoning
        answer = await self._generate_llm_analysis(
            food_name=food_label,
            confidence=confidence,
            nutrition=nutrition,
            question=question,
            uncertain=uncertain,
            top5=top5,
        )

        # 5. Record to session history
        user_prompt_summary = question if (question and question.strip()) else f"Analyze food image ({food_label})"
        self.memory.add_message(current_session_id, role="user", content=user_prompt_summary)
        self.memory.add_message(current_session_id, role="assistant", content=answer)
        self.memory.update_session_data(
            current_session_id,
            classification=classification,
            nutrition=nutrition,
            image_b64=image_b64,
        )

        return {
            "session_id": current_session_id,
            "answer": answer,
            "classification": classification,
            "nutrition": nutrition,
            "confidence_badge": confidence_badge,
            "uncertain": uncertain,
        }

    async def chat(self, question: str, session_id: str) -> Dict[str, Any]:
        """
        Handle conversational follow-up questions within a session.

        Args:
            question: Follow-up user question.
            session_id: Active session identifier.

        Returns:
            Dict: {
                "session_id": str,
                "answer": str,
                "classification": dict,
                "nutrition": dict,
                "confidence_badge": str
            }
        """
        session = self.memory.get_or_create_session(session_id)
        current_session_id = session.session_id

        last_classification = session.last_classification or {}
        last_nutrition = session.last_nutrition or {}
        food_name = last_classification.get("label") or last_nutrition.get("food_name")
        confidence = float(last_classification.get("confidence", 0.0))
        confidence_badge = self.derive_confidence_badge(confidence)

        history = self.memory.get_history(current_session_id)

        answer = await self._generate_llm_chat(
            question=question,
            food_name=food_name,
            nutrition=last_nutrition,
            history=history,
        )

        # Record messages
        self.memory.add_message(current_session_id, role="user", content=question)
        self.memory.add_message(current_session_id, role="assistant", content=answer)

        return {
            "session_id": current_session_id,
            "answer": answer,
            "classification": last_classification,
            "nutrition": last_nutrition,
            "confidence_badge": confidence_badge,
        }
