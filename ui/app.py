"""
Multimodal Food Agent - Gradio User Interface.

Provides an interactive web interface:
- Food image upload + optional dietary inquiry
- Confidence & Status Card with calibrated visual badges (🟢 High, 🟡 Moderate, 🔴 Low / Honest Uncertainty)
- Nutrition Card with macros (Calories, Protein, Carbs, Fat, Fiber, Sodium, Serving Size, Data Source)
- Top 5 classification candidates with probability distribution
- Multi-turn dietary assistant chat with session memory
- Quick test examples for 1-click evaluation
"""
import asyncio
import base64
import io
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

# Ensure workspace root is in sys.path
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from orchestrator.agent import FoodOrchestrator
from orchestrator.memory import SessionMemory

# Initialize shared food orchestrator
orchestrator = FoodOrchestrator()


def pil_to_base64_jpeg(pil_img: Image.Image, quality: int = 90) -> str:
    """Converts a PIL Image to a JPEG base64 data string."""
    buffered = io.BytesIO()
    rgb_img = pil_img.convert("RGB")
    rgb_img.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def render_confidence_card(
    classification: Optional[Dict[str, Any]] = None,
    confidence_badge: Optional[str] = None,
    uncertain: bool = False,
) -> str:
    """Renders an attractive HTML Confidence & Status Card."""
    if not classification:
        return """
        <div style="padding: 16px; border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0; font-family: system-ui, sans-serif;">
            <div style="font-size: 14px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Vision Classification Status</div>
            <div style="margin-top: 8px; font-size: 15px; color: #94a3b8;">No food image analyzed yet. Upload a photo or select an example below to begin.</div>
        </div>
        """

    label = str(classification.get("label", "unknown_food")).replace("_", " ").title()
    confidence = float(classification.get("confidence", 0.0))
    conf_pct = confidence * 100.0
    source = classification.get("source", "vision-model")
    method = classification.get("method", "classification")

    if confidence >= 0.75 and not uncertain:
        badge_bg = "#ecfdf5"
        badge_border = "#10b981"
        badge_text = "#065f46"
        badge_icon = "🟢"
        badge_title = "High Confidence"
    elif confidence >= 0.50 and not uncertain:
        badge_bg = "#fefce8"
        badge_border = "#eab308"
        badge_text = "#854d0e"
        badge_icon = "🟡"
        badge_title = "Moderate Confidence"
    else:
        badge_bg = "#fef2f2"
        badge_border = "#ef4444"
        badge_text = "#991b1b"
        badge_icon = "🔴"
        badge_title = "Low Confidence / Honest Uncertainty"

    uncertainty_banner = ""
    if uncertain or confidence < 0.65:
        uncertainty_banner = f"""
        <div style="margin-top: 10px; padding: 10px; border-radius: 6px; background: #fee2e2; border-left: 4px solid #ef4444; font-size: 13px; color: #7f1d1d;">
            <strong>⚠️ Calibrated Uncertainty Notice:</strong> The classifier is not fully confident ({conf_pct:.1f}%). 
            Double-check if this matches your dish before relying on exact nutritional counts.
        </div>
        """

    return f"""
    <div style="padding: 16px; border-radius: 10px; background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-family: system-ui, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Classification Result</span>
            <span style="display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 9999px; background: {badge_bg}; border: 1px solid {badge_border}; font-size: 12px; font-weight: 600; color: {badge_text};">
                {badge_icon} {badge_title} ({conf_pct:.1f}%)
            </span>
        </div>
        <div style="margin-top: 8px; font-size: 20px; font-weight: 700; color: #0f172a;">{label}</div>
        <div style="margin-top: 4px; font-size: 12px; color: #64748b;">
            <span>Source: <code>{source}</code></span> &bull; 
            <span>Method: <code>{method}</code></span>
        </div>
        {uncertainty_banner}
    </div>
    """


def render_nutrition_card(nutrition: Optional[Dict[str, Any]] = None) -> str:
    """Renders an attractive HTML Nutrition Card displaying macro and micronutrients."""
    if not nutrition:
        return """
        <div style="padding: 16px; border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0; font-family: system-ui, sans-serif;">
            <div style="font-size: 14px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Nutritional Facts</div>
            <div style="margin-top: 8px; font-size: 15px; color: #94a3b8;">Nutritional breakdown will appear here once food is identified.</div>
        </div>
        """

    food_name = str(nutrition.get("food_name", "Food")).title()
    serving = nutrition.get("serving_size_g", 100)
    calories = nutrition.get("calories", 0.0)
    protein = nutrition.get("protein_g", 0.0)
    carbs = nutrition.get("carbs_g", 0.0)
    fat = nutrition.get("fat_g", 0.0)
    fiber = nutrition.get("fiber_g", 0.0)
    sodium = nutrition.get("sodium_mg", 0.0)
    source = nutrition.get("data_source", "Verified Source")

    return f"""
    <div style="padding: 18px; border-radius: 10px; background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-family: system-ui, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: baseline; border-bottom: 2px solid #0f172a; padding-bottom: 6px;">
            <div>
                <span style="font-size: 18px; font-weight: 800; color: #0f172a;">Nutrition Facts</span>
                <span style="margin-left: 8px; font-size: 13px; color: #64748b;">({food_name})</span>
            </div>
            <span style="font-size: 12px; font-weight: 600; color: #0284c7; background: #e0f2fe; padding: 2px 8px; border-radius: 4px;">{source}</span>
        </div>
        <div style="font-size: 12px; color: #64748b; margin-top: 4px; padding-bottom: 8px; border-bottom: 6px solid #0f172a;">
            Serving Size: <strong>{serving}g</strong>
        </div>
        
        <div style="display: flex; justify-content: space-between; align-items: baseline; padding: 8px 0; border-bottom: 3px solid #0f172a;">
            <span style="font-size: 15px; font-weight: 800; color: #0f172a;">Calories</span>
            <span style="font-size: 26px; font-weight: 900; color: #0f172a;">{calories:.0f} <span style="font-size: 14px; font-weight: 500; color: #64748b;">kcal</span></span>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px;">
            <div style="padding: 8px 10px; background: #f8fafc; border-radius: 6px;">
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">Protein</div>
                <div style="font-size: 16px; font-weight: 700; color: #0f172a;">{protein:.1f}g</div>
            </div>
            <div style="padding: 8px 10px; background: #f8fafc; border-radius: 6px;">
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">Total Carbohydrates</div>
                <div style="font-size: 16px; font-weight: 700; color: #0f172a;">{carbs:.1f}g</div>
            </div>
            <div style="padding: 8px 10px; background: #f8fafc; border-radius: 6px;">
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">Total Fat</div>
                <div style="font-size: 16px; font-weight: 700; color: #0f172a;">{fat:.1f}g</div>
            </div>
            <div style="padding: 8px 10px; background: #f8fafc; border-radius: 6px;">
                <div style="font-size: 11px; color: #64748b; text-transform: uppercase;">Dietary Fiber</div>
                <div style="font-size: 16px; font-weight: 700; color: #0f172a;">{fiber:.1f}g</div>
            </div>
        </div>

        <div style="margin-top: 8px; padding: 6px 10px; background: #f1f5f9; border-radius: 6px; display: flex; justify-content: space-between; font-size: 13px;">
            <span style="color: #475569;">Sodium</span>
            <span style="font-weight: 700; color: #0f172a;">{sodium:.0f} mg</span>
        </div>
    </div>
    """


def render_top5_card(top5: Optional[List[Dict[str, Any]]] = None) -> str:
    """Renders the top 5 classification candidates with visual progress bars."""
    if not top5:
        return """
        <div style="padding: 14px; border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0; font-family: system-ui, sans-serif;">
            <div style="font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Top Candidate Predictions</div>
            <div style="margin-top: 6px; font-size: 13px; color: #94a3b8;">Candidate breakdown will appear upon image upload.</div>
        </div>
        """

    items_html = []
    for idx, item in enumerate(top5[:5], 1):
        lbl = str(item.get("label", "")).replace("_", " ").title()
        conf = float(item.get("confidence", 0.0)) * 100.0
        bar_color = "#3b82f6" if idx == 1 else "#94a3b8"
        items_html.append(f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 2px;">
                <span style="font-weight: {'700' if idx == 1 else '500'}; color: #1e293b;">{idx}. {lbl}</span>
                <span style="font-weight: 600; color: #475569;">{conf:.1f}%</span>
            </div>
            <div style="width: 100%; height: 6px; background: #f1f5f9; border-radius: 3px; overflow: hidden;">
                <div style="width: {min(max(conf, 2.0), 100.0)}%; height: 100%; background: {bar_color}; border-radius: 3px;"></div>
            </div>
        </div>
        """)

    content = "".join(items_html)
    return f"""
    <div style="padding: 14px; border-radius: 10px; background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); font-family: system-ui, sans-serif;">
        <div style="font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;">Top Candidate Predictions</div>
        {content}
    </div>
    """


# Main pipeline event handlers
async def on_analyze_food(
    image: Optional[Image.Image],
    question: Optional[str],
    state: Dict[str, Any],
) -> Tuple[str, str, str, List[Dict[str, str]], Dict[str, Any]]:
    """Analyzes food image, retrieves nutrition, synthesizes reasoning, and updates state."""
    session_id = state.get("session_id", uuid.uuid4().hex)
    messages = list(state.get("messages", []))

    if image is None:
        warning_msg = "Please upload or select a food photo first to run the analysis."
        messages.append({"role": "assistant", "content": f"⚠️ {warning_msg}"})
        return (
            render_confidence_card(),
            render_nutrition_card(),
            render_top5_card(),
            messages,
            state,
        )

    # Convert PIL image to base64
    img_b64 = pil_to_base64_jpeg(image)

    # Prompt text for history
    user_text = question.strip() if (question and question.strip()) else "Analyze this food dish and provide nutritional insights."
    messages.append({"role": "user", "content": user_text})

    # Call orchestrator
    result = await orchestrator.analyze(
        image_b64=img_b64,
        question=question,
        session_id=session_id,
    )

    answer = result.get("answer", "")
    classification = result.get("classification", {})
    nutrition = result.get("nutrition", {})
    confidence_badge = result.get("confidence_badge", "Moderate (🟡)")
    uncertain = result.get("uncertain", False)
    top5 = classification.get("top5", [])

    messages.append({"role": "assistant", "content": answer})

    # Update session state
    new_state = {
        "session_id": session_id,
        "last_classification": classification,
        "last_nutrition": nutrition,
        "messages": messages,
    }

    conf_html = render_confidence_card(classification, confidence_badge, uncertain)
    nutr_html = render_nutrition_card(nutrition)
    top5_html = render_top5_card(top5)

    return conf_html, nutr_html, top5_html, messages, new_state


async def on_send_followup(
    question: str,
    state: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], str, Dict[str, Any]]:
    """Handles multi-turn follow-up queries using session memory."""
    q_text = question.strip() if question else ""
    messages = list(state.get("messages", []))
    session_id = state.get("session_id", uuid.uuid4().hex)

    if not q_text:
        return messages, "", state

    messages.append({"role": "user", "content": q_text})

    # Call orchestrator chat
    res = await orchestrator.chat(
        question=q_text,
        session_id=session_id,
    )

    answer = res.get("answer", "I could not process this follow-up query.")
    messages.append({"role": "assistant", "content": answer})

    new_state = dict(state)
    new_state["messages"] = messages

    return messages, "", new_state


def on_clear_all() -> Tuple[
    None, str, str, str, str, List[Dict[str, str]], str, Dict[str, Any]
]:
    """Clears inputs, reset cards, and starts a fresh session."""
    new_session_id = uuid.uuid4().hex
    return (
        None,
        "",
        render_confidence_card(),
        render_nutrition_card(),
        render_top5_card(),
        [],
        "",
        {"session_id": new_session_id, "messages": []},
    )


# Build Gradio interface
def build_app() -> gr.Blocks:
    """Constructs the Gradio user interface application."""
    with gr.Blocks(title="Multimodal Food Agent") as demo:
        # Session State
        session_state = gr.State(lambda: {"session_id": uuid.uuid4().hex, "messages": []})

        # Header
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
                    # 🥑 Multimodal Food Agent
                    **Multimodal Food Identification & Nutritional Intelligence** powered by Gemini Vision, USDA FoodData Central, Open Food Facts, and MCP.
                    """,
                )

        # Main Layout: 2 Columns
        with gr.Row():
            # Left Column: Upload, Cards, & Metrics
            with gr.Column(scale=5):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Food Photo",
                    sources=["upload", "clipboard", "webcam"],
                )
                initial_question = gr.Textbox(
                    label="Ask about this food (optional)",
                    placeholder="e.g., Is this good for a low-carb diet? What are the macros?",
                    lines=2,
                )

                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Food", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

                # Confidence and Status Badge
                confidence_card_html = gr.HTML(
                    value=render_confidence_card(),
                    label="Classification Status",
                )

                # Nutrition Breakdown Card
                nutrition_card_html = gr.HTML(
                    value=render_nutrition_card(),
                    label="Nutritional Breakdown",
                )

                # Top 5 Candidate Probabilities
                top5_card_html = gr.HTML(
                    value=render_top5_card(),
                    label="Top Candidates",
                )

            # Right Column: Multi-turn Chat
            with gr.Column(scale=6):
                # Compatible with Gradio 6.x and earlier versions
                try:
                    chatbot = gr.Chatbot(
                        label="Dietary Assistant Chat",
                        height=520,
                        type="messages",
                    )
                except TypeError:
                    chatbot = gr.Chatbot(
                        label="Dietary Assistant Chat",
                        height=520,
                    )

                with gr.Row():
                    followup_input = gr.Textbox(
                        placeholder="Ask follow-up questions about this food, recipe adjustments, or dietary goals...",
                        label="Follow-up Inquiry",
                        lines=2,
                        scale=5,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                # Educational Disclaimer Footer
                gr.Markdown(
                    """
                    <div style="font-size: 11px; color: #64748b; margin-top: 10px; line-height: 1.4;">
                        ⚠️ <em>Disclaimer: Nutritional information is derived from USDA FoodData Central and Open Food Facts. 
                        Estimates are provided for educational purposes and do not replace certified medical or clinical dietary advice.</em>
                    </div>
                    """
                )

        # Examples Row
        sample_dir = WORKSPACE_DIR / "eval" / "food_samples"
        examples_list = []
        if sample_dir.exists():
            sample_candidates = [
                ("pizza.jpg", "Is this suitable for a low-carb or keto diet?"),
                ("apple_pie.jpg", "What is the total sugar and carb breakdown?"),
                ("caesar_salad.jpg", "How much protein does this salad contain?"),
                ("hamburger.jpg", "What are the macro splits for muscle gain?"),
                ("salmon.jpg", "Is this food heart-healthy and rich in omega-3?"),
            ]
            for fname, q in sample_candidates:
                fpath = sample_dir / fname
                if fpath.exists():
                    examples_list.append([str(fpath), q])

        if examples_list:
            gr.Examples(
                examples=examples_list,
                inputs=[image_input, initial_question],
                label="Quick Test Food Samples (Click to test instantly)",
            )

        # Wire Event Handlers
        analyze_btn.click(
            fn=on_analyze_food,
            inputs=[image_input, initial_question, session_state],
            outputs=[
                confidence_card_html,
                nutrition_card_html,
                top5_card_html,
                chatbot,
                session_state,
            ],
        )

        send_btn.click(
            fn=on_send_followup,
            inputs=[followup_input, session_state],
            outputs=[chatbot, followup_input, session_state],
        )

        followup_input.submit(
            fn=on_send_followup,
            inputs=[followup_input, session_state],
            outputs=[chatbot, followup_input, session_state],
        )

        clear_btn.click(
            fn=on_clear_all,
            inputs=[],
            outputs=[
                image_input,
                initial_question,
                confidence_card_html,
                nutrition_card_html,
                top5_card_html,
                chatbot,
                followup_input,
                session_state,
            ],
        )

    return demo


# Create app instance
demo = build_app()


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
