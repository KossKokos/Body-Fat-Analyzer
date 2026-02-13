import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
from pathlib import Path

load_dotenv()
env_file = Path(__file__).parent.parent.parent / '.env'

class Settings(BaseSettings):
    # Application
    model_config = SettingsConfigDict(env_file=env_file, 
                                    env_file_encoding='utf-8')
    PROJECT_NAME: str = os.environ.get("PROJECT_NAME") # type: ignore
    VERSION: str = os.environ.get("VERSION") # type: ignore
    DEBUG: bool = True
    
    # API
    API_V1_STR: str = os.environ.get("API_V1_STR") # type: ignore
    DOCS: bool = True
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000", "http://localhost:5173"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False
    LOG_TO_FILE: bool = False
    LOG_FILE: Optional[str] = "logs/app.log"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "app" / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Model paths
    CLASSIFIER_PATH: Path = MODELS_DIR / "classifiers" / "fat_percentage_classifier_v1.keras"
    REGRESSOR_PATHS: dict = {
        "low": MODELS_DIR / "regressors" / "low_fat_residuals_regressor_v1.keras",
        "mid": MODELS_DIR / "regressors" / "mid_fat_residuals_regressor_v1.keras",
        "high": MODELS_DIR / "regressors" / "high_fat_residuals_regressor_v1.keras",
    }
    BASE_MODELS_PATHS: dict = {
        "low": MODELS_DIR / "base_models" / "low_fat_base_model.pkl",
        "mid": MODELS_DIR / "base_models" / "mid_fat_base_model.pkl",
        "high": MODELS_DIR / "base_models" / "high_fat_base_model.pkl",
    }
    SCALERS_PATHS: dict = {
        "class": MODELS_DIR / "scalers" / "classification_scaler.pkl",
        "low": MODELS_DIR / "scalers" / "low_fat_scaler.pkl",
        "mid": MODELS_DIR / "scalers" / "mid_fat_scaler.pkl",
        "high": MODELS_DIR / "scalers" / "high_fat_scaler.pkl",
    }
    
    # Database
    POSTGRES_DB: str = os.environ.get("POSTGRES_DB") # type: ignore
    POSTGRES_USER: str = os.environ.get("POSTGRES_DB") # type: ignore
    POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_DB") # type: ignore
    POSTGRES_PORT: str = os.environ.get("POSTGRES_DB") # type: ignore
    POSTGRES_HOST: str = os.environ.get("POSTGRES_DB") # type: ignore
    SQLALCHEMY_DATABASE_URL: str = os.environ.get("SQLALCHEMY_DATABASE_URL") # type: ignore

    # APP START UP
    APP_MAIN: str = os.environ.get("APP_MAIN") # type: ignore
    APP_HOST: str = os.environ.get("APP_HOST") # type: ignore
    APP_PORT: str = os.environ.get("APP_PORT") # type: ignore

settings = Settings()