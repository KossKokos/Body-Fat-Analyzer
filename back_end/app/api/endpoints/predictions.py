from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.schemas import PredictionRequest, PredictionResponse
from api.dependencies import get_db
from api.dependencies import get_prediction_service
from config.logger import logger

router = APIRouter(prefix="/predict")
 
@router.post("/", response_model=PredictionResponse)
async def predict(
    data: PredictionRequest,
    service=Depends(get_prediction_service),
    db: Session = Depends(get_db),
):
    try:
        user_input = data.model_dump()
        prediction = service.predict_fat_percentage(user_input)
        logger.info("Prediction calculation completed")


        # saved_prediction = service.save_prediction_history(
        #     db=db,
        #     user_data=user_input,
        #     result=prediction,
        # )
    
        # if saved_prediction is None:
        #     raise HTTPException(status_code=500, detail="Prediction succeeded but could not be saved")

        prediction["prediction_id"] = 1#saved_prediction.id

        logger.info("Building prediction response completed")
        return PredictionResponse(**prediction)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")