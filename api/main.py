"""
FastAPI Gateway for Multimodal Food Agent.

Exposes REST endpoints:
- GET  /health              : Liveness check and service readiness status.
- POST /analyze             : Multimodal food analysis (accepts JSON or multipart/form-data).
- POST /chat                : Multi-turn conversational follow-ups within a session.
- GET  /session/{session_id}: Retrieve dialogue history and cached food context.
"""
import base64
import logging
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orchestrator.agent import FoodOrchestrator
from orchestrator.memory import default_session_memory

logger = logging.getLogger("api.main")

# Initialize FastAPI application
app = FastAPI(
    title="Multimodal Food Agent API",
    description="Multimodal AI Agent for food identification, nutritional retrieval, and dietary reasoning.",
    version="1.0.0",
)

# Enable CORS for frontend / dev / cross-origin testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator = FoodOrchestrator(memory=default_session_memory)


# --- Request and Response Schemas ---

class AnalyzeRequest(BaseModel):
    """Payload for JSON food analysis requests."""
    image_b64: Optional[str] = Field(
        None,
        description="Base64-encoded image string (with or without data URL prefix)."
    )
    question: Optional[str] = Field(
        None,
        description="Optional dietary question (e.g. 'Is this suitable for keto?')."
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for tracking ongoing dialogue."
    )


class AnalyzeResponse(BaseModel):
    """Structured response from multimodal food analysis."""
    session_id: str
    answer: str
    classification: Dict[str, Any]
    nutrition: Dict[str, Any]
    confidence_badge: str
    uncertain: bool


class ChatRequest(BaseModel):
    """Payload for conversational follow-ups."""
    session_id: str = Field(..., description="Active session ID containing prior food context.")
    question: str = Field(..., description="User follow-up dietary or nutritional question.")


class ChatResponse(BaseModel):
    """Response for conversational follow-ups."""
    session_id: str
    answer: str
    classification: Optional[Dict[str, Any]] = None
    nutrition: Optional[Dict[str, Any]] = None
    confidence_badge: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    api: str
    version: str
    gemini_api: str
    vision: Dict[str, Any]
    nutrition: Dict[str, Any]


# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health and liveness probe.
    Reports operational status of the API, Gemini LLM connection, and MCP tools.
    """
    gemini_status = "configured" if orchestrator._genai_configured else "offline / fallback_mode"
    return HealthResponse(
        status="healthy",
        api="online",
        version="1.0.0",
        gemini_api=gemini_status,
        vision={
            "status": "ready",
            "primary": "gemini-2.0-flash",
            "secondary": "efficientnet-b4",
            "confidence_threshold": 0.65,
        },
        nutrition={
            "status": "ready",
            "sources": [
                "USDA FoodData Central",
                "Open Food Facts",
                "Standard Reference Estimates",
            ],
        },
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="Analyze Food Image",
    description="Accepts JSON body or multipart/form-data image upload. Runs vision classifier, retrieves nutrition facts, and synthesizes dietary reasoning.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "image_b64": {"type": "string", "description": "Base64 encoded food image"},
                            "question": {"type": "string", "description": "Optional dietary question"},
                            "session_id": {"type": "string", "description": "Optional session identifier"},
                        },
                        "required": ["image_b64"],
                    }
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary", "description": "Image file"},
                            "image": {"type": "string", "format": "binary", "description": "Alternative file field"},
                            "question": {"type": "string", "description": "Optional dietary question"},
                            "session_id": {"type": "string", "description": "Optional session identifier"},
                            "image_b64": {"type": "string", "description": "Base64 image string"},
                        },
                    }
                },
            }
        }
    },
)
async def analyze_endpoint(request: Request) -> AnalyzeResponse:
    """
    Analyze food from image (JSON payload or multipart upload).
    """
    content_type = request.headers.get("content-type", "").lower()
    image_b64: Optional[str] = None
    question: Optional[str] = None
    session_id: Optional[str] = None

    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
            question_val = form.get("question")
            question = str(question_val).strip() if question_val else None

            session_val = form.get("session_id")
            session_id = str(session_val).strip() if session_val else None

            # Check for file upload
            upload = form.get("file") or form.get("image")
            if upload is not None and hasattr(upload, "read"):
                file_bytes = await upload.read()
                if file_bytes:
                    image_b64 = base64.b64encode(file_bytes).decode("utf-8")

            # Check if image_b64 was sent as form text field
            if not image_b64 and "image_b64" in form:
                b64_val = form.get("image_b64")
                if b64_val:
                    image_b64 = str(b64_val).strip()

        except Exception as e:
            logger.warning("Error reading multipart form: %s", e)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse multipart form: {str(e)}",
            )
    else:
        # Default or application/json
        try:
            body = await request.json()
            if isinstance(body, dict):
                image_b64 = body.get("image_b64")
                question = body.get("question")
                session_id = body.get("session_id")
        except Exception as e:
            logger.warning("Error reading JSON body: %s", e)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request must be valid JSON with 'image_b64' or multipart/form-data with image file.",
            )

    if not image_b64 or not str(image_b64).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required image data. Provide 'image_b64' in JSON or upload an image file.",
        )

    try:
        result = await orchestrator.analyze(
            image_b64=image_b64,
            question=question,
            session_id=session_id,
        )
        return AnalyzeResponse(**result)
    except Exception as e:
        logger.error("Analysis failed unexpectedly: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {str(e)}",
        )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request_data: ChatRequest) -> ChatResponse:
    """
    Follow-up conversation within a session.
    Retrieves previous food classification and nutrition context to answer questions.
    """
    if not request_data.question or not request_data.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question field cannot be empty.",
        )

    try:
        result = await orchestrator.chat(
            question=request_data.question.strip(),
            session_id=request_data.session_id.strip(),
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error("Chat endpoint error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing error: {str(e)}",
        )


@app.get("/session/{session_id}", tags=["Session"])
async def get_session_endpoint(session_id: str) -> Dict[str, Any]:
    """
    Retrieve full state, cached nutritional context, and dialogue history for a session.
    """
    session = orchestrator.memory.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' was not found.",
        )
    return session.to_dict()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
