import copy
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .model_service import ModelService
from config.logger import logger
from ml.loading_script import add_engineered_features

class PredictionService:
    """
    Responsible for:
    1. Validating incoming prediction requests
    2. Orchestrating the full prediction pipeline
    3. Calling the right models in the right order
    4. Formatting the final response
    5. Logging predictions
    6. Error handling for predictions
    """
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service  # Gets models from here
        # self.history_service = history_service
        self.features: List[str] = [ 
            'age', 'gender', 'weight', 'height', 'max_bpm',
            'avg_bpm', 'resting_bpm', 'session_duration',
            'calories_burned', 'workout_type', 'water_intake',
            'workout_frequency', 'experience_level', 'bmi',
            'daily_meals_frequency', 'carbs', 'proteins',
            'fats', 'calories', 'diet_type', 'sugar_g', 
            'fat_percentage'
        ]
        self.input_features: List[str] = self.features[:-1] 

        self._encoded_features_list = [
            'age', 'weight', 'height', 'max_bpm', 'avg_bpm', 'resting_bpm',
            'session_duration', 'calories_burned', 'water_intake',
            'workout_frequency', 'experience_level', 'bmi', 'daily_meals_frequency',
            'carbs', 'proteins', 'fats', 'calories', 'sugar_g', 'gender_female',
            'gender_male', 'workout_type_cardio', 'workout_type_hiit',
            'workout_type_strength', 'workout_type_yoga', 'diet_type_balanced',
            'diet_type_keto', 'diet_type_low-carb', 'diet_type_paleo',
            'diet_type_vegan', 'diet_type_vegetarian'
        ]
        
        self.map_fat_class = {
            0: 'low',
            1: 'mid',
            2: 'high'
        }
                
    def _get_bmi(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data['bmi'] = float(data['weight']) / (float(data['height']))**2
        return data
    
    def _reorder_features(self, data: Dict[str, Any], desired_order: List[str]) -> Dict[str, Any]:
        reordered_data = {k: data[k] for k in desired_order}
        return reordered_data
    
    def _to_DataFrame(self, data: Dict[str, Any], columns: List[str]) -> pd.DataFrame:
        X = pd.DataFrame([data], columns=columns)
        return X

    def _fill_missing_fields(self, df: pd.DataFrame) -> None:
        for feature in self._encoded_features_list:
            if feature not in df.columns:
                df[feature] = 0.0

    def _encode_cat_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = pd.get_dummies(data=df)
        return df_encoded

    def _validate_input(self, user_data: Dict[str, Any]) -> pd.DataFrame:
        """Orders features in the correct order, returns a pd.DataFrame"""    
        data_copy = copy.deepcopy(user_data)
        # calculate bmi feature and add to input 
        data_copy = self._get_bmi(data=data_copy)
        # reorder features in the way model was trained
        data_reordered = self._reorder_features(data=data_copy, desired_order=self.input_features)
        # convert to DataFrame
        df = self._to_DataFrame(data=data_reordered, columns=self.input_features)
        return df

    def _preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        df = copy.deepcopy(data)
        # encode categorical features
        df_encoded = self._encode_cat_features(df=df)
        self._fill_missing_fields(df=df_encoded)
        # add engineered features
        df_fe = add_engineered_features(df=df_encoded)
        return df_fe
    
    def _scale_features(self, X: pd.DataFrame, scaler) -> np.ndarray:
        # transform/scale features
        X_array = X.to_numpy()
        X_trans = scaler.transform(X_array)
        return X_trans

    def _get_final_preds(self, preds: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        final_preds = preds.reshape(-1) + residuals.reshape(-1)
        # return final_preds
        return preds.reshape(-1)
    
    def _get_fat_class_name(self, fat_class: int) -> str:
        return self.map_fat_class[fat_class]

    def predict_fat_percentage(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. Validate input data
        2. Preprocess features
        3. Call classifier (via ModelService)
        4. Call appropriate regressor (via ModelService)
        5. Format response
        6. Log to history
        7. Return result
        """
        # Step 1: Validate
        validated = self._validate_input(user_data)
        
        # Step 2: Preprocess
        features = self._preprocess_features(validated)
    
        # Step 3: Scale features
        classification_scaler = self.model_service.get_classification_scaler()
        classification_features = self._scale_features(X=features, scaler=classification_scaler)
        # Step 4: Classify (using ModelService)
        classifier = self.model_service.get_classifier()
        fat_class_softmax = classifier.predict(classification_features)
        fat_class_int = np.argmax(fat_class_softmax).item()
        fat_class = self._get_fat_class_name(fat_class=fat_class_int)
        # Step 4: Scale features
        regressor_scaler = self.model_service.get_regression_scaler(fat_class=fat_class)
        regression_features = self._scale_features(X=features, scaler=regressor_scaler)
        
        # Step 5: Predict using base model
        base_model = self.model_service.get_base_model(fat_class)
        preds = base_model.predict(regression_features)
        
        # Step 6: Regress (using residual model) 
        regressor = self.model_service.get_regressor(fat_class)
        residuals = regressor.predict(regression_features)
        
        # Step 7: Combine predictions 
        final_preds = self._get_final_preds(preds, residuals)
        fat_percentage = round(final_preds.item(), 3)
        logger.info(
                    f"Recieved final prediction: {fat_percentage}",
                )
        # Step 8: Format
        result = {
            "fat_class": fat_class,
            "fat_percentage": float(fat_percentage),
            "timestamp": datetime.now()
        }
        # Step 6: Log
        # self.history_service.save_prediction(user_data, result)
        
        return result