import pickle
from typing import Dict, Any
from pathlib import Path

import keras


from ml.loading_script import quantile_loss_lower, quantile_loss_upper
from utils.logger import log_execution_time, LoggerMixin
from config.settings import settings

# to inore messages

class ModelService(LoggerMixin):
    """Service for managing ML models."""
    
    _instance = None
    models: Dict[str, keras.Sequential | Any] = {}
    __map_objs = {
            'low': {'quantile_loss' : quantile_loss_lower},
            'high': {'quantile_loss': quantile_loss_upper}
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance

    def _get_custom_object(self, type_):
        return self.__map_objs[type_]

    def load_keras_model(self, filepath: Path | str) -> keras.Sequential:
        """Safe way to load Keras model with pickle"""
        
        try: 
            model = keras.models.load_model(filepath=filepath)
            return model # type: ignore
        except Exception as e:
            self.logger.exception(
                "Failed to load keras instance",
                error=str(e),
                path=str(filepath)
            )
            raise

    def load_instance_pkl(self, filename: str | Path):
        with open(filename, 'rb') as f:
            instance = pickle.load(f)
        return instance

    @classmethod
    def load_models(cls):
        """Load all ML models."""
        instance = cls()
        
        with log_execution_time("loading_classifier", level="info"):
            instance._load_classifier()
        
        with log_execution_time("loading_regressors", level="info"):
            instance._load_regressors()

        with log_execution_time("loading_base_models", level="info"):
            instance._load_base_models()
        
        with log_execution_time("loading_scalers", level="info"):
            instance._load_scalers()
        
        instance.logger.info(
            "All models loaded successfully",
            classifier_loaded="classifier" in instance.models,
            regressors_loaded=list(instance.models.get("regressors", {}).keys()),
            base_models_loaded=list(instance.models.get("base_models", {}).keys())
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
            self.models["classifier"] = self.load_keras_model(classifier_path)
            
            self.logger.info("Classifier loaded successfully")
            
        except Exception as e:
            self.logger.exception(
                "Failed to load classifier",
                error=str(e),
                path=str(classifier_path)
            )
            raise
    
    def _load_scalers(self):
        """Load scaler."""
        self.models["scalers"] = {}
        
        for type_, path in settings.SCALERS_PATHS.items():
            try:
                self.logger.debug(
                    f"Loading {type_} scaler",
                    path=str(path)
                )
                
                # Load scaler
                self.models["scalers"][type_] = self.load_instance_pkl(path)

                self.logger.info(
                    f"{type_} scaler loaded",
                    model_type="scaler"
                )
                
            except Exception as e:
                self.logger.exception(
                    f"Failed to load {type_} scaler",
                    error=str(e),
                    path=str(path)
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
                self.models["regressors"][fat_class] = self.load_keras_model(path)

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
                self.models["base_models"][fat_class] = self.load_instance_pkl(path)

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

    def get_classifier(self) -> keras.Sequential:
        self.logger.debug(
                f"Getting a classifier",
        )
        try:
            return self.models['classifier']
        except Exception as e:
            self.logger.exception(
                f"Failed to get classifier",
                error=str(e),
                )
            raise

    def get_classification_scaler(self):
        self.logger.debug(
            f"Getting a classification scaler",
        )
        try:
            return self.models['scalers']['class']
        except Exception as e:
            self.logger.exception(
                f"Failed to get classification scaler",
                error=str(e),
                )
            raise

    def get_regression_scaler(self, fat_class):
        self.logger.debug(
            f"Getting a scaler, class: {fat_class}",
        )
        try:
            return self.models['scalers'][fat_class]
        except Exception as e:
            self.logger.exception(
                f"Failed to get scaler, class: {fat_class}",
                error=str(e),
                )
            raise

    def get_regressor(self, fat_class: str) -> keras.Sequential:    
        self.logger.debug(
            f"Getting a regressor, class: {fat_class}",
        )
        try:
            return self.models['regressors'][fat_class]
        except Exception as e:
            self.logger.exception(
                f"Failed to get regressor, class: {fat_class}",
                error=str(e),
                )
            raise

    def get_base_model(self, fat_class: str) -> keras.Sequential:    
        self.logger.debug(
            f"Getting a base model, class: {fat_class}",
        )
        try:
            return self.models['base_models'][fat_class]
        except Exception as e:
            self.logger.exception(
                f"Failed to get a base_model, class: {fat_class}",
                error=str(e),
                )
            raise
    
    def __repr__(self):
        return f"ModelService({self.models})"