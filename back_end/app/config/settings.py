# app/config/settings.py
from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path


class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "Body Fat Percentage Predictor"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API
    API_V1_STR: str = "/api/v1"
    DOCS: bool = True
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False
    LOG_TO_FILE: bool = False
    LOG_FILE: Optional[str] = "logs/app.log"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Model paths
    CLASSIFIER_PATH: Path = MODELS_DIR / "classifiers" / "fat_percentage_classifier_v1"
    REGRESSOR_PATHS: dict = {
        "low": MODELS_DIR / "regressors" / "low_fat_residuals_regressor_v1",
        "mid": MODELS_DIR / "regressors" / "mid_fat_residuals_regressor_v1",
        "high": MODELS_DIR / "regressors" / "high_fat_residuals_regressor_v1",
    }
    BASE_MODELS_PATHS: dict = {
        "low": MODELS_DIR / "base_models" / "low_fat_base_model",
        "mid": MODELS_DIR / "base_models" / "mid_fat_base_model",
        "high": MODELS_DIR / "base_models" / "high_fat_base_model",
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()