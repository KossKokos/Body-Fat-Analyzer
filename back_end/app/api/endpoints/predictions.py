from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_prediction_service
from core.schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict")
 

@router.post("/", response_model=PredictionResponse)
async def predict(
    data: PredictionRequest,
    service=Depends(get_prediction_service),
):
    try:
        user_input = data.model_dump()
        prediction = service.predict_fat_percentage(user_input)
        return PredictionResponse(**prediction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")
