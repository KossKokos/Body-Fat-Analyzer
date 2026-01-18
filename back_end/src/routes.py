import copy

from fastapi import APIRouter, status, Request

import src.schemas as schemas
import src.repository as repository

router = APIRouter(prefix="/model", tags=["model"])


@router.post('/predict',
             response_model=schemas.PredictionResponce | str,
             status_code=status.HTTP_200_OK,
             )
async def predict(
                        data: schemas.DataSchema, 
                        request: Request):
    data = copy.deepcopy(data)
    orig_data, validated_data = await repository.validate_data(data=data)
    new_features_data = await repository.add_engineered_features(df=validated_data)
    x_transformed = await repository.transform_x(x=new_features_data)
    
    prediction = await repository.get_predictions(x=x_transformed)  
    await repository.add_to_history(orig_data, prediction)  

    response = schemas.PredictionResponce(fat_percentage=prediction)
    return response