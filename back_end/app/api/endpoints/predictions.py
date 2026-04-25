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
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db.commit()
        logger.info("SELECT 1 FROM DB SUCCEEDED")
        
        user_input = data.model_dump()
        prediction = service.predict_fat_percentage(user_input)
        logger.info("Prediction calculation completed")

        saved_prediction = service.save_prediction_history(
            db=db,
            user_data=user_input,
            result=prediction,
        )
    
        if saved_prediction is None:
            raise HTTPException(status_code=500, detail="Prediction succeeded but could not be saved")
        logger.info("Prediction is saved successfully")

        prediction["prediction_id"] = saved_prediction.id

        logger.info("Building prediction response completed")
        return PredictionResponse(**prediction)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed")