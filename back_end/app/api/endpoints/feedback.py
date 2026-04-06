from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import get_prediction_service, get_db
from core.schemas import (
    PredictionFeedbackCreateRequest,
    PredictionFeedbackResponse,
)

router = APIRouter(prefix="/feedback")


@router.post("/", response_model=PredictionFeedbackResponse)
async def create_feedback(
    data: PredictionFeedbackCreateRequest,
    service=Depends(get_prediction_service),
    db: Session = Depends(get_db),
):
    try:
        feedback = service.save_prediction_feedback(
            db=db,
            feedback_data=data.model_dump(),
        )

        return PredictionFeedbackResponse(
            id=feedback.id,
            prediction_id=feedback.prediction_id,
            rating=feedback.rating,
            is_prediction_close=feedback.is_prediction_close,
            actual_fat_percentage=float(feedback.actual_fat_percentage) if feedback.actual_fat_percentage is not None else None,
            comment=feedback.comment,
            consent_to_retrain=feedback.consent_to_retrain,
            created_at=feedback.created_at,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save prediction feedback")