import os

from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

env_file = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_file,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #  Database
    POSTGRES_DB: str = ""
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    POSTGRES_PORT: str = ""
    POSTGRES_HOST: str = ""
    SQLALCHEMY_DATABASE_URL: str = ""

    DB_SSL_MODE: str = ""
    DB_POOL_PRE_PING: bool = True
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE: int = 1800
    DB_ECHO: bool = False

    # DB instances
    REQUIRED_TABLES: List[str] = []
    # APP START UP
    APP_MAIN: str = ""
    APP_HOST: str = ""
    APP_PORT: str = ""

    PROJECT_NAME: str = "Body_Fat_Percentage_Predictor"
    VERSION: str = "1.0.0"

    ENVIRONMENT: str = ""
    DEBUG: bool = False
    DOCS: bool = False

    API_V1_STR: str = ""
    FRONTEND_URL: str = ""
    API_KEY: str = ""

    ALLOWED_ORIGINS: List[str] = []
    ALLOWED_HOSTS: List[str] = []

    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False
    LOG_TO_FILE: bool = False
    LOG_FILE: Optional[str] = "logs/app.log"

    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "app" / "models"
    DATA_DIR: Path = BASE_DIR / "data"

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

    @property
    def SECURITY_HEADERS(self) -> dict:
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "Content-Security-Policy": "default-src 'self'; connect-src 'self' http://localhost:8000;",
        }
        if self.ENVIRONMENT == "production":
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return headers

    APP_HOST: str = ""
    APP_MAIN: str = ""
    APP_PORT: str = ""

settings = Settings()