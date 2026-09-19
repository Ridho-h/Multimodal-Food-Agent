"""
Prompts and Dietary Reasoning Guidance for Multimodal Food Agent.

Defines:
1. System prompt grounding LLM advice strictly in verified nutritional numbers.
2. Calibration for common dietary profiles (Keto, High Protein, Deficit, Diabetic, Low Sodium).
3. Honest uncertainty preambles when vision confidence is low.
4. Mandatory educational/medical disclaimer.
5. High-quality rule-based fallback generators for offline / missing API key operation.
"""
from typing import Any, Dict, List, Optional

# Mandatory educational and medical disclaimer
MEDICAL_DISCLAIMER = (
    "⚠️ **Disclaimer**: *This analysis and dietary feedback are provided solely for educational "
    "and informational purposes and do not constitute medical advice or a personalized nutritional "
    "prescription. Always consult a licensed healthcare provider or registered dietitian before making "
    "significant changes to your diet, especially if you have underlying medical conditions.*"
)

# Uncertainty preamble for low-confidence classifications
UNCERTAINTY_PREAMBLE_TEMPLATE = (
    "> ⚠️ **Identification Notice**: The vision model flagged uncertainty for this image "
    "(confidence: {confidence_pct:.1f}%). The food appears to be **{food_name}**"
    "{alternatives_clause}. If this differs from your actual dish, please interpret the "
    "nutritional data accordingly."
)

# Comprehensive System Prompt for Gemini LLM
SYSTEM_PROMPT = """You are an expert AI Food & Nutrition Intelligence Agent.
Your role is to analyze food items identified from images, explain their nutritional profiles, and provide objective, helpful dietary advice tailored to user questions or general wellness goals.

CORE PRINCIPLES & CONSTRAINTS:
1. STRICT NUTRITIONAL GROUNDING:
   - Ground ALL dietary assessments strictly in the verified nutritional data provided to you (calories, protein, carbs, fat, fiber, sodium, serving size).
   - NEVER invent, extrapolate, or hallucinate nutrition numbers. If a nutrient metric is missing or zero, state that clearly rather than guessing.
   - Always state the reference serving size (e.g. 100g) associated with the numbers.

2. CALIBRATION FOR COMMON DIETARY GOALS:
   - Low Carb / Keto:
     * Focus on Net Carbohydrates (Total Carbs minus Dietary Fiber).
     * Evaluate healthy fat content and overall keto macro compatibility (typically <20-50g net carbs/day).
   - High Protein / Muscle Building:
     * Emphasize total protein grams and protein-to-calorie ratio (e.g., >10g protein per 100 kcal is high protein density).
     * Mention satiety and muscle recovery benefits.
   - Calorie Deficit / Weight Loss:
     * Evaluate caloric density (kcal per 100g).
     * Highlight fiber and protein for fullness, and flag high-calorie liquid or hidden fat traps.
   - Diabetic Friendly / Blood Sugar Management:
     * Examine total carbohydrates, glycemic load indicators, and dietary fiber buffering.
     * Note whether the item contains fast-digesting simple carbs or slow-digesting complex carbs/fiber.
   - Heart Healthy / Low Sodium:
     * Review sodium levels: Low sodium (< 140 mg per serving), Moderate (140-400 mg), High (> 400 mg).
     * Consider saturated fat vs unsaturated fat content and dietary fiber for cholesterol management.

3. HONEST UNCERTAINTY HANDLING:
   - If the classification result has 'uncertain: True' or confidence < 0.65, acknowledge this uncertainty up front.
   - Clarify that if the actual food is different, the nutritional metrics will differ.

4. USER INQUIRY FOCUSED:
   - If the user provides a specific question (e.g. "Can I eat this for dinner on a cut?"), prioritize answering that question directly and concisely before presenting broader observations.

5. MANDATORY DISCLAIMER:
   - Always conclude your analysis with the official educational and medical disclaimer.
"""


def format_uncertainty_preamble(
    food_name: str,
    confidence: float,
    top5: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate an honest uncertainty preamble when confidence is under 0.65 or uncertain is True."""
    confidence_pct = max(0.0, min(1.0, confidence)) * 100.0
    alternatives = []
    if top5:
        for item in top5:
            lbl = item.get("label", "")
            if lbl and lbl.lower() != food_name.lower() and lbl not in alternatives:
                alternatives.append(lbl)

    alternatives_clause = ""
    if alternatives:
        alternatives_clause = f", though alternative possibilities include: {', '.join(alternatives[:3])}"

    return UNCERTAINTY_PREAMBLE_TEMPLATE.format(
        confidence_pct=confidence_pct,
        food_name=food_name.title(),
        alternatives_clause=alternatives_clause,
    )


def format_analysis_prompt(
    food_name: str,
    confidence: float,
    nutrition: Dict[str, Any],
    question: Optional[str] = None,
    uncertain: bool = False,
    top5: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the user prompt instructing Gemini to analyze the food and nutrition context."""
    serving_size = nutrition.get("serving_size_g", 100)
    calories = nutrition.get("calories", 0.0)
    protein = nutrition.get("protein_g", 0.0)
    carbs = nutrition.get("carbs_g", 0.0)
    fat = nutrition.get("fat_g", 0.0)
    fiber = nutrition.get("fiber_g", 0.0)
    sodium = nutrition.get("sodium_mg", 0.0)
    source = nutrition.get("data_source", "Database")

    # Net carbs calculation
    net_carbs = max(0.0, carbs - fiber)

    prompt = (
        f"FOOD IDENTIFICATION:\n"
        f"- Identified Food: {food_name}\n"
        f"- Classifier Confidence: {confidence:.2f} ({confidence * 100:.1f}%)\n"
        f"- Classification Uncertain: {'Yes' if uncertain else 'No'}\n\n"
        f"VERIFIED NUTRITIONAL DATA (per {serving_size}g serving, Source: {source}):\n"
        f"- Calories: {calories} kcal\n"
        f"- Protein: {protein} g\n"
        f"- Total Carbohydrates: {carbs} g\n"
        f"- Dietary Fiber: {fiber} g\n"
        f"- Net Carbohydrates: {net_carbs:.1f} g\n"
        f"- Total Fat: {fat} g\n"
        f"- Sodium: {sodium} mg\n\n"
    )

    if uncertain or confidence < 0.65:
        prompt += (
            "NOTICE: The vision model expressed uncertainty. Include the uncertainty warning "
            "prominently at the top of your answer.\n\n"
        )

    if question and question.strip():
        prompt += (
            f"USER QUESTION:\n"
            f'"{question.strip()}"\n\n'
            f"INSTRUCTION:\n"
            f"Answer the user's question directly based on these verified nutritional metrics, "
            f"then provide a concise nutritional breakdown with dietary goal context (Low Carb/Keto, "
            f"High Protein, Calorie Deficit, Diabetic Friendly, Heart Healthy/Sodium). Conclude with the disclaimer."
        )
    else:
        prompt += (
            "INSTRUCTION:\n"
            "Provide a comprehensive, structured nutritional breakdown for this food. "
            "Evaluate its suitability across key dietary goals: Low Carb / Keto, High Protein, "
            "Calorie Deficit / Weight Loss, Diabetic Friendly, and Heart Healthy / Low Sodium. "
            "Conclude with the disclaimer."
        )

    return prompt


def format_chat_prompt(
    question: str,
    food_name: Optional[str],
    nutrition: Optional[Dict[str, Any]],
    history: List[Dict[str, str]],
) -> str:
    """Build a conversational follow-up prompt for Gemini including chat history."""
    context = ""
    if food_name and nutrition:
        serving = nutrition.get("serving_size_g", 100)
        context = (
            f"CURRENT FOOD CONTEXT:\n"
            f"- Food Item: {food_name}\n"
            f"- Serving Size: {serving}g\n"
            f"- Calories: {nutrition.get('calories', 0.0)} kcal\n"
            f"- Protein: {nutrition.get('protein_g', 0.0)} g\n"
            f"- Carbs: {nutrition.get('carbs_g', 0.0)} g (Fiber: {nutrition.get('fiber_g', 0.0)} g)\n"
            f"- Fat: {nutrition.get('fat_g', 0.0)} g\n"
            f"- Sodium: {nutrition.get('sodium_mg', 0.0)} mg\n"
            f"- Source: {nutrition.get('data_source', 'Database')}\n\n"
        )

    history_str = ""
    if history:
        history_str = "CONVERSATION HISTORY:\n"
        for msg in history[-6:]:  # Keep recent context window
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_str += f"{role}: {msg.get('content', '')}\n"
        history_str += "\n"

    prompt = (
        f"{context}"
        f"{history_str}"
        f"NEW USER QUESTION:\n"
        f'"{question.strip()}"\n\n'
        f"INSTRUCTION:\n"
        f"Answer the user's question accurately. Refer back to the nutritional metrics and conversation "
        f"history where relevant. Never invent new nutritional values. Conclude with the medical disclaimer if giving dietary advice."
    )
    return prompt


def build_rule_based_analysis(
    food_name: str,
    confidence: float,
    nutrition: Dict[str, Any],
    question: Optional[str] = None,
    uncertain: bool = False,
    top5: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Robust rule-based nutritional summary and dietary explanation.
    Used as an intelligent fallback when Gemini API key is missing or calls fail.
    """
    food_display = food_name.replace("_", " ").title()
    serving = nutrition.get("serving_size_g", 100)
    calories = float(nutrition.get("calories", 0.0))
    protein = float(nutrition.get("protein_g", 0.0))
    carbs = float(nutrition.get("carbs_g", 0.0))
    fat = float(nutrition.get("fat_g", 0.0))
    fiber = float(nutrition.get("fiber_g", 0.0))
    sodium = float(nutrition.get("sodium_mg", 0.0))
    source = nutrition.get("data_source", "Nutritional Database")
    net_carbs = max(0.0, carbs - fiber)

    sections = []

    # 1. Uncertainty preamble
    if uncertain or confidence < 0.65:
        sections.append(format_uncertainty_preamble(food_display, confidence, top5))

    # 2. Main Title & Macro Overview
    header = (
        f"### 🍽️ Nutritional Breakdown: **{food_display}**\n"
        f"*Reference Serving Size: **{serving}g** | Data Source: {source}*\n\n"
        f"| Metric | Amount | Dietary Context |\n"
        f"| :--- | :--- | :--- |\n"
        f"| **Calories** | `{calories} kcal` | Energy density |\n"
        f"| **Protein** | `{protein} g` | Muscle recovery & satiety |\n"
        f"| **Total Carbs** | `{carbs} g` | Energy source |\n"
        f"| **Fiber** | `{fiber} g` | Digestive & glycemic buffer |\n"
        f"| **Net Carbs** | `{net_carbs:.1f} g` | Carbs minus fiber |\n"
        f"| **Total Fat** | `{fat} g` | Essential fatty acids & satiety |\n"
        f"| **Sodium** | `{sodium} mg` | Fluid balance & cardiovascular |"
    )
    sections.append(header)

    # 3. Direct Question Answer (if user provided a specific query)
    q_lower = (question or "").lower()
    if question and question.strip():
        q_answer = []
        q_answer.append(f"#### 💬 Direct Response to: *\"{question.strip()}\"*")

        if any(term in q_lower for term in ["keto", "low carb", "carbs", "atkins"]):
            if net_carbs <= 5.0:
                q_answer.append(f"- **Low Carb / Keto Verdict**: **Highly Compatible**. With only **{net_carbs:.1f}g net carbs** per {serving}g, this food fits easily within a standard 20-50g daily keto carbohydrate allowance.")
            elif net_carbs <= 15.0:
                q_answer.append(f"- **Low Carb / Keto Verdict**: **Moderate / Portion-Dependent**. Contains **{net_carbs:.1f}g net carbs** per {serving}g. Consume in mindful portions to stay in ketosis.")
            else:
                q_answer.append(f"- **Low Carb / Keto Verdict**: **Not Ideal for Strict Keto**. At **{net_carbs:.1f}g net carbs** per {serving}g, a single serving accounts for a substantial portion of typical keto limits.")

        elif any(term in q_lower for term in ["protein", "muscle", "bulk", "gym", "workout"]):
            if protein >= 15.0:
                q_answer.append(f"- **Protein Profile**: **Excellent High-Protein Source**. Delivering **{protein}g protein** per {serving}g, it is well-suited for muscle synthesis and satiety.")
            elif protein >= 7.0:
                q_answer.append(f"- **Protein Profile**: **Moderate Protein**. Provides **{protein}g protein** per {serving}g. Good as a complementary protein component in a meal.")
            else:
                q_answer.append(f"- **Protein Profile**: **Low Protein Content**. Contains **{protein}g protein** per {serving}g. Consider pairing with a lean protein source (e.g., eggs, tofu, poultry) if targeting higher protein goals.")

        elif any(term in q_lower for term in ["weight loss", "deficit", "fat loss", "cut", "diet"]):
            if calories <= 120.0:
                q_answer.append(f"- **Calorie Deficit Verdict**: **Great for Weight Loss**. Low caloric density (**{calories} kcal** per {serving}g) allows for satisfying meal volume without high calorie intake.")
            elif calories <= 250.0:
                q_answer.append(f"- **Calorie Deficit Verdict**: **Moderate Caloric Density**. With **{calories} kcal** per {serving}g, it can be seamlessly incorporated into a calorie deficit with portion control.")
            else:
                q_answer.append(f"- **Calorie Deficit Verdict**: **Calorie-Dense**. At **{calories} kcal** per {serving}g, monitor portion sizes carefully to prevent accidentally exceeding your daily energy budget.")

        elif any(term in q_lower for term in ["diabetic", "diabetes", "sugar", "blood sugar", "glucose"]):
            if carbs <= 10.0 or fiber >= 4.0:
                q_answer.append(f"- **Blood Sugar Impact**: **Favorable / Low Glycemic Load**. Moderate carbs ({carbs}g) combined with fiber ({fiber}g) helps promote a steady glucose response.")
            else:
                q_answer.append(f"- **Blood Sugar Impact**: **Mindful Consumption Recommended**. Contains **{carbs}g carbohydrates** with **{fiber}g fiber**. Pairing with healthy fats or proteins can blunt rapid glycemic spikes.")

        elif any(term in q_lower for term in ["sodium", "heart", "blood pressure", "salt", "hypertension"]):
            if sodium <= 140.0:
                q_answer.append(f"- **Heart & Sodium Verdict**: **Low Sodium**. At **{sodium} mg**, it meets the standard criterion for low sodium foods (<140 mg/serving).")
            elif sodium <= 400.0:
                q_answer.append(f"- **Heart & Sodium Verdict**: **Moderate Sodium**. Contains **{sodium} mg** per {serving}g serving. Keep total daily sodium under 2,300 mg (or 1,500 mg for hypertension).")
            else:
                q_answer.append(f"- **Heart & Sodium Verdict**: **Elevated Sodium**. Contains **{sodium} mg**, which is a significant portion of the recommended daily limit.")

        else:
            q_answer.append(
                f"Based on the nutritional facts for {food_display}, a {serving}g serving provides "
                f"**{calories} kcal**, **{protein}g protein**, **{carbs}g carbs**, and **{fat}g fat**. "
                f"Check the dietary goal evaluations below to see how this fits your specific lifestyle."
            )

        sections.append("\n".join(q_answer))

    # 4. Calibrated Dietary Goal Evaluations
    goals_card = ["#### 🎯 Dietary Goal Suitability:"]

    # Low Carb / Keto
    if net_carbs <= 5.0:
        goals_card.append(f"- **🥑 Low Carb / Keto**: ✅ **Excellent** ({net_carbs:.1f}g net carbs per {serving}g)")
    elif net_carbs <= 15.0:
        goals_card.append(f"- **🥑 Low Carb / Keto**: ⚠️ **Moderate** ({net_carbs:.1f}g net carbs; watch portion sizing)")
    else:
        goals_card.append(f"- **🥑 Low Carb / Keto**: ❌ **Challenging** ({net_carbs:.1f}g net carbs may exceed strict daily allowances)")

    # High Protein
    if protein >= 15.0:
        goals_card.append(f"- **💪 High Protein**: ✅ **High Protein Density** ({protein}g per {serving}g)")
    elif protein >= 7.0:
        goals_card.append(f"- **💪 High Protein**: ⚠️ **Moderate Protein** ({protein}g per {serving}g)")
    else:
        goals_card.append(f"- **💪 High Protein**: ℹ️ **Low Protein** ({protein}g per {serving}g; pair with protein-rich sides)")

    # Calorie Deficit
    if calories <= 120.0:
        goals_card.append(f"- **⚖️ Calorie Deficit**: ✅ **Volume Friendly** ({calories} kcal per {serving}g)")
    elif calories <= 260.0:
        goals_card.append(f"- **⚖️ Calorie Deficit**: ⚠️ **Moderate Caloric Density** ({calories} kcal per {serving}g)")
    else:
        goals_card.append(f"- **⚖️ Calorie Deficit**: ⚠️ **Calorie Dense** ({calories} kcal per {serving}g; portion control advised)")

    # Diabetic Friendly
    if carbs <= 12.0 or (fiber / max(1.0, carbs)) >= 0.25:
        goals_card.append(f"- **🩸 Diabetic Friendly**: ✅ **Steady Glycemic Profile** ({fiber}g fiber buffers {carbs}g total carbs)")
    else:
        goals_card.append(f"- **🩸 Diabetic Friendly**: ⚠️ **Monitor Carbohydrates** ({carbs}g carbs with {fiber}g fiber)")

    # Heart Healthy / Low Sodium
    if sodium <= 140.0:
        goals_card.append(f"- **❤️ Heart Healthy / Low Sodium**: ✅ **Low Sodium** ({sodium} mg sodium per {serving}g)")
    elif sodium <= 400.0:
        goals_card.append(f"- **❤️ Heart Healthy / Low Sodium**: ⚠️ **Moderate Sodium** ({sodium} mg sodium)")
    else:
        goals_card.append(f"- **❤️ Heart Healthy / Low Sodium**: ⚠️ **High Sodium** ({sodium} mg sodium; balance with low-sodium meals)")

    sections.append("\n".join(goals_card))

    # 5. Medical Disclaimer
    sections.append(MEDICAL_DISCLAIMER)

    return "\n\n".join(sections)


def build_rule_based_chat_answer(
    question: str,
    food_name: Optional[str],
    nutrition: Optional[Dict[str, Any]],
) -> str:
    """Rule-based conversational answer when Gemini LLM is offline or rate-limited during chat follow-ups."""
    if not food_name or not nutrition:
        return (
            "⚠️ **Notice:** The AI conversational model is currently offline or rate-limited. "
            "Please upload a food image first to begin an analysis, or wait ~30 seconds for the quota window to refresh.\n\n"
            f"{MEDICAL_DISCLAIMER}"
        )

    notice = (
        "> ⚠️ **Rate Limit Fallback:** The Gemini API free-tier request limit was reached. "
        "Showing rule-based dietary guidance based on the verified nutritional facts below:\n\n"
    )
    return notice + build_rule_based_analysis(
        food_name=food_name,
        confidence=1.0,
        nutrition=nutrition,
        question=question,
        uncertain=False,
    )
