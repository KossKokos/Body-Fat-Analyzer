from fastapi import APIRouter, Depends

from api.dependencies import get_prediction_service
from core.schemas import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict")

@router.post("/", response_model=PredictionResponse)
async def predict(
    data: PredictionRequest,
    service = Depends(get_prediction_service)
):
    user_input = data.model_dump()
    prediction = service.predict_fat_percentage(user_input)
    return PredictionResponse(**prediction) 

