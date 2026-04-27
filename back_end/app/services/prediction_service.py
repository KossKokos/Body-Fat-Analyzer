import copy
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .model_service import ModelService
from config.logger import logger
from config.settings import settings
from config import constants as cnsts
from ml.loading_script import add_engineered_features
from database.models import (
    PredictionHistory,
    PredictionFeedback,
    GenderEnum,
    WorkoutTypeEnum,
    DietTypeEnum,
    FatClassEnum,
)


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
    
    MIN_FAT_PERCENTAGE = 5.0
    MAX_FAT_PERCENTAGE = 40.0

    def __init__(self, model_service: ModelService):
        self.model_service = model_service
                
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
        for feature in cnsts.ENCODED_FEATURES:
            if feature not in df.columns:
                df[feature] = 0.0

    def _encode_cat_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = pd.get_dummies(data=df)
        return df_encoded

    def _validate_input(self, user_data: Dict[str, Any]) -> pd.DataFrame:
        data_copy = copy.deepcopy(user_data)
        data_copy = self._get_bmi(data=data_copy)
        data_reordered = self._reorder_features(
            data=data_copy,
            desired_order=list(cnsts.MODEL_INPUT_FEATURES),
        )
        df = self._to_DataFrame(
            data=data_reordered,
            columns=list(cnsts.MODEL_INPUT_FEATURES),
        )
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
        return final_preds
    
    def _get_fat_class_name(self, fat_class: int) -> str:
        return cnsts.FAT_CLASS_MAP[fat_class]

    def _normalize_fat_percentage(self, fat_percentage: float) -> float:
        """
        Clamp predicted fat percentage to the business-approved response range.
        """
        return round(min(max(float(fat_percentage), 5.0), 40.0), 3)

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
        try:
            # Step 1: Validate
            validated = self._validate_input(user_data)

            # Step 2: Preprocess
            features = self._preprocess_features(validated)

            # Step 3: Scale features for classification
            classification_scaler = self.model_service.get_classification_scaler()
            classification_features = self._scale_features(
                X=features,
                scaler=classification_scaler,
            )

            # Step 4: Classify
            classifier = self.model_service.get_classifier()
            fat_class_softmax = classifier.predict(classification_features)
            fat_class_int = np.argmax(fat_class_softmax).item()
            fat_class = self._get_fat_class_name(fat_class=fat_class_int)

            # Step 5: Scale features for regression
            regressor_scaler = self.model_service.get_regression_scaler(
                fat_class=fat_class
            )
            regression_features = self._scale_features(
                X=features,
                scaler=regressor_scaler,
            )

            # Step 6: Predict using base model
            base_model = self.model_service.get_base_model(fat_class)
            preds = base_model.predict(regression_features)

            # Step 7: Predict residuals
            regressor = self.model_service.get_regressor(fat_class)
            residuals = regressor.predict(regression_features)

            # Step 8: Combine predictions and normalize output
            final_preds = self._get_final_preds(preds, residuals)
            fat_percentage = self._normalize_fat_percentage(final_preds.item())

            logger.info(f"Received final prediction: {fat_percentage}")

            result = {
                "fat_class": fat_class,
                "fat_percentage": float(fat_percentage),
                "timestamp": datetime.now(),
            }
            return result

        except ValueError as e:
            logger.exception(
                "Prediction validation/value error",
                error=str(e),
            )
            raise

        except Exception as e:
            logger.exception(
                "Prediction pipeline failed",
                error=str(e),
            )
            raise
    
    # DB LOGIC
    def _to_gender_enum(self, value: str) -> GenderEnum:
        return GenderEnum(value)

    def _to_workout_type_enum(self, value: str) -> WorkoutTypeEnum:
        return WorkoutTypeEnum(value)

    def _to_diet_type_enum(self, value: str) -> DietTypeEnum:
        if value == "low-carb":
            return DietTypeEnum.low_carb
        return DietTypeEnum(value)

    def _to_fat_class_enum(self, value: str) -> FatClassEnum:
        return FatClassEnum(value)

    def create_row_prediction(self, user_data: Dict[str, Any], result: Dict[str, Any]) -> PredictionHistory:
        row = PredictionHistory(
            age=int(user_data["age"]),
            gender=self._to_gender_enum(user_data["gender"]),
            weight=user_data["weight"],
            height=user_data["height"],
            max_bpm=int(user_data["max_bpm"]),
            avg_bpm=int(user_data["avg_bpm"]),
            resting_bpm=int(user_data["resting_bpm"]),
            session_duration=user_data["session_duration"],
            calories_burned=int(user_data["calories_burned"]),
            workout_type=self._to_workout_type_enum(user_data["workout_type"]),
            workout_frequency=user_data["workout_frequency"],
            experience_level=user_data["experience_level"],
            calories=int(user_data["calories"]),
            carbs=int(user_data["carbs"]),
            proteins=int(user_data["proteins"]),
            fats=int(user_data["fats"]),
            sugar_g=int(user_data["sugar_g"]),
            diet_type=self._to_diet_type_enum(user_data["diet_type"]),
            daily_meals_frequency=int(user_data["daily_meals_frequency"]),
            water_intake=user_data["water_intake"],
            fat_class=self._to_fat_class_enum(result["fat_class"]),
            fat_percentage=result["fat_percentage"],
            model_version=settings.MODEL_VERSION,
            )
        return row

    def create_row_feedback(
            self, 
            feedback_data: Dict[str, Any], 
            prediction_id: int,
            actual_fat_percentage: float | None,
            consent_to_retrain: bool) -> PredictionFeedback:
        
        feedback = PredictionFeedback(
            prediction_id=prediction_id,
            rating=int(feedback_data["rating"]),
            is_prediction_close=feedback_data.get("is_prediction_close"),
            actual_fat_percentage=actual_fat_percentage,
            comment=feedback_data.get("comment"),
            consent_to_retrain=consent_to_retrain,
            consent_timestamp=datetime.now() if feedback_data.get("consent_to_retrain") else None,
            )
        return feedback

    def save_prediction_history(
        self,
        db: Session,
        user_data: Dict[str, Any],
        result: Dict[str, Any],
    ) -> PredictionHistory | None:
        try:
            row = self.create_row_prediction(user_data=user_data, result=result)

            db.add(row)
            db.commit()
            db.refresh(row)

            logger.info("Prediction history saved successfully")
            return row
        
        except Exception as e:
            db.rollback()
            logger.exception(
                "Failed to save prediction history",
                error=str(e),
            )

    def save_prediction_feedback(
        self,
        db: Session,
        feedback_data: Dict[str, Any],
    ) -> PredictionFeedback:
        try:
            prediction_id = feedback_data["prediction_id"]

            prediction = (
                db.query(PredictionHistory)
                .filter(PredictionHistory.id == prediction_id)
                .first()
            )

            if prediction is None:
                raise ValueError("Prediction not found")
            
            actual_fat_percentage = feedback_data.get("actual_fat_percentage")
            consent_to_retrain = feedback_data.get("consent_to_retrain", False)
            if consent_to_retrain and actual_fat_percentage is None:
                raise ValueError(
                    "actual_fat_percentage is required when consent_to_retrain is true"
            )

            feedback = self.create_row_feedback(
                feedback_data=feedback_data, 
                prediction_id=prediction_id,
                actual_fat_percentage=actual_fat_percentage, 
                consent_to_retrain=consent_to_retrain
            )

            db.add(feedback)
            db.commit()
            db.refresh(feedback)

            logger.info(
                "Prediction feedback saved successfully",
            )
            return feedback

        except ValueError:
            raise
        except Exception as e:
            db.rollback()
            logger.exception(
                "Failed to save prediction feedback",
                error=str(e),
            )
            raise