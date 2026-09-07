from typing import Any, Dict, cast

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data.feedback_store import ATHSFeedbackStore
from detection.engine import ATHSThreatDetectionEngine
from detection.feature_loader import ML_FEATURE_COLUMNS
from preprocessing.pipeline import PreprocessingPipeline


# =========================================================
# ATHS API
# =========================================================

app = FastAPI(
    title="ATHS - Prompt Injection Threat Hypothesis System",
    description="API for prompt injection threat detection.",
    version="1.0.0",
)


# =========================================================
# Request Models
# =========================================================

class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Prompt to analyze.",
    )


class FeedbackRequest(BaseModel):
    prediction_id: int = Field(
        ...,
        description="ID of the prediction being reviewed.",
    )

    human_label: int = Field(
        ...,
        ge=0,
        le=1,
        description="0 = benign, 1 = malicious.",
    )

    attack_category: str | None = Field(
        default=None,
        description="Optional attack category.",
    )


# =========================================================
# Initialize ATHS Components
# =========================================================

print("Initializing ATHS API components...")

engine = ATHSThreatDetectionEngine()

preprocessor = PreprocessingPipeline(
    enable_embeddings=True
)

feedback_store = ATHSFeedbackStore()

print("ATHS API components initialized.")


# =========================================================
# Root Endpoint
# =========================================================

@app.get("/")
def root() -> Dict[str, str]:
    return {
        "system": "ATHS",
        "status": "running",
        "service": "Prompt Injection Threat Detection API",
    }


# =========================================================
# Health Endpoint
# =========================================================

@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "status": "healthy",
    }


# =========================================================
# Analyze Prompt
# =========================================================

@app.post("/analyze")
def analyze_prompt(
    request: AnalyzeRequest,
) -> Dict[str, Any]:

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Prompt text cannot be empty.",
        )

    try:

        # -------------------------------------------------
        # 1. Preprocess prompt
        # -------------------------------------------------

        processed = cast(
            Dict[str, Any],
            cast(Any, preprocessor).process_text(text),
        )

        # -------------------------------------------------
        # 2. Build 15-dimensional ML feature vector
        # -------------------------------------------------

        feature_sources = {
            **processed["nlp"],
            **processed["obfuscation"],
            **processed["security_features"],
        }

        feature_vector = np.asarray(
            [
                feature_sources[column]
                for column in ML_FEATURE_COLUMNS
            ],
            dtype=np.float32,
        )

        # -------------------------------------------------
        # 3. Get 384-dimensional semantic embedding
        # -------------------------------------------------

        embedding_data = processed.get(
            "semantic_embedding"
        )

        if not isinstance(embedding_data, list):
            raise ValueError(
                "Semantic embedding was not generated."
            )

        embedding = np.asarray(
            embedding_data,
            dtype=np.float32,
        )

        # -------------------------------------------------
        # 4. Run ATHS detection engine
        # -------------------------------------------------

        result = engine.analyze(
            text=text,
            feature_vector=feature_vector,
            embedding=embedding,
        )

        # -------------------------------------------------
        # 5. Store prediction
        # -------------------------------------------------

        prediction_id = feedback_store.save_prediction(
            text=text,
            ml_probability=float(
                result["ml"]["probability"]
            ),
            rule_score=float(
                result["rules"]["score"]
            ),
            semantic_score=float(
                result["semantic"]["score"]
            ),
            threat_score=float(
                result["threat_score"]
            ),
            decision=str(
                result["decision"]
            ),
            severity=str(
                result["severity"]
            ),
        )

        # -------------------------------------------------
        # 6. Add prediction ID to API response
        # -------------------------------------------------

        result["prediction_id"] = prediction_id

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"ATHS analysis failed: {str(exc)}",
        )


# =========================================================
# Submit Human Feedback
# =========================================================

@app.post("/feedback")
def submit_feedback(
    request: FeedbackRequest,
) -> Dict[str, Any]:

    try:

        feedback_store.add_feedback(
            prediction_id=request.prediction_id,
            human_label=request.human_label,
            attack_category=request.attack_category,
        )

        return {
            "status": "success",
            "prediction_id": request.prediction_id,
            "human_label": request.human_label,
            "attack_category": request.attack_category,
            "message": "Human feedback stored successfully.",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save feedback: {str(exc)}",
        )


# =========================================================
# Run API Directly
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )