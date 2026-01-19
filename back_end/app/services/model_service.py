# app/services/model_service.py
import copy
import pickle
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from app.config.logger import logger
from app.utils.logger import log_execution_time, LoggerMixin, log_function_call
from app.config.settings import settings


class ModelService(LoggerMixin):
    """Service for managing ML models."""
    
    _instance = None
    models: Dict[str, Any] = {}
    features: List[str] = [ 
        'calories_burned', 'max_bpm', 'age',
        'weight', 'daily_meals_frequency', 'resting_bpm',
        'bmi', 'workout_frequency', 'water_intake',
        'session_duration', 'height', 'gender',
        'workout_type', 'fat_percentage'
    ]
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def load_models(cls):
        """Load all ML models."""
        instance = cls()
        
        with log_execution_time("loading_classifier", level="info"):
            instance._load_classifier()
        
        with log_execution_time("loading_regressors", level="info"):
            instance._load_regressors()
        
        instance.logger.info(
            "All models loaded successfully",
            classifier_loaded="classifier" in instance.models,
            regressors_loaded=list(instance.models.get("regressors", {}).keys())
        )
    
    def _load_classifier(self):
        """Load classification model."""
        classifier_path = settings.CLASSIFIER_PATH
        try:
            self.logger.debug(
                "Loading classifier",
                path=str(classifier_path)
            )
            
            # Load your classifier
            # Example: self.models["classifier"] = load_model(classifier_path)
            
            self.logger.info("Classifier loaded successfully")
            
        except Exception as e:
            self.logger.exception(
                "Failed to load classifier",
                error=str(e),
                path=str(classifier_path)
            )
            raise
    
    def _load_regressors(self):
        """Load regression models."""
        self.models["regressors"] = {}
        
        for fat_class, path in settings.REGRESSOR_PATHS.items():
            try:
                self.logger.debug(
                    f"Loading {fat_class} regressor",
                    path=str(path)
                )
                
                # Load regressor
                # Example: 
                # with open(path, 'rb') as f:
                #     self.models["regressors"][fat_class] = pickle.load(f)
                
                self.logger.info(
                    f"{fat_class} regressor loaded",
                    model_type="regressor"
                )
                
            except Exception as e:
                self.logger.exception(
                    f"Failed to load {fat_class} regressor",
                    error=str(e),
                    path=str(path)
                )
                raise

    def _load_base_models(self):
        """Load Base models."""
        self.models["base_models"] = {}
        
        for fat_class, path in settings.BASE_MODELS_PATHS.items():
            try:
                self.logger.debug(
                    f"Loading {fat_class} base model",
                    path=str(path)
                )
                
                # Load base model
                # Example: 
                # with open(path, 'rb') as f:
                #     self.models["regressors"][fat_class] = pickle.load(f)
                
                self.logger.info(
                    f"{fat_class} base model loaded",
                    model_type="base_model"
                )
                
            except Exception as e:
                self.logger.exception(
                    f"Failed to load {fat_class} base model",
                    error=str(e),
                    path=str(path)
                )
                raise

    def encode_cat_features(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(data=data)

    @log_function_call("info")
    def validate_data(self, data: Dict[str, Any]) -> Tuple[List[Any], pd.DataFrame]:
        """Orders features in the correct order, returns a pd.DataFrame"""    
        data_copy = copy.deepcopy(data)
        data_copy['bmi'] = data_copy['weight'] / (data_copy['height'])**2
        desired_order_list = self.features
        
        reordered_data = {k: data_copy[k] for k in desired_order_list}
        orig_data = list(reordered_data.values())

        X = pd.DataFrame([orig_data], columns=desired_order_list)
        X = self.encode_cat_features(X)
        # X_fe = add_engineered_features(X)
        # X_test = X_scaler.transform(X_fe)
        # ready for classification and then predictions
        return orig_data, X
    
    @log_function_call("info")
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction using loaded models."""
        try:
            self.logger.debug("Making prediction", features=features)
            
            # 1. Preprocess features
            processed = self._preprocess_features(features)
            
            # 2. Classify
            fat_class = self._classify(processed)
            
            # 3. Get predictions with base model
            preds = None

            # 4. Regress on residuals
            percentage = self._regress(fat_class, processed)
            
            # 5. Combine predicitons
            y_pred_full = None
        
            result = {
                "fat_class": fat_class,
                "fat_percentage": percentage,
                "confidence": 0.85  # Example
            }
            
            self.logger.info(
                "Prediction completed",
                result=result,
                fat_class=fat_class
            )
            
            return result
            
        except Exception as e:
            self.logger.exception(
                "Prediction failed",
                error=str(e),
                features=features
            )
            raise
    
    def _preprocess_features(self, features: Dict[str, float]) -> np.ndarray:
        """Preprocess input features."""
        with log_execution_time("feature_preprocessing", level="debug"):
            # Your preprocessing logic
            return np.array([1, 2, 3])
    
    def _classify(self, features: np.ndarray) -> str:
        """Classify fat percentage category."""
        with log_execution_time("classification", level="debug"):
            # Your classification logic
            return 'a'
    
    def _regress(self, fat_class: str, features: np.ndarray) -> float:
        """Predict exact fat percentage."""
        with log_execution_time("regression", level="debug"):
            # Your regression logic
            return 20.2