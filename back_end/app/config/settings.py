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

    PROJECT_NAME: str = "Body_Fat_Percentage_Predictor"
    VERSION: str = "1.0.0"

    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    DOCS: bool = False

    API_V1_STR: str = "/api"
    FRONTEND_URL: str = "http://localhost:5173"
    API_KEY: str = os.environ.get("API_KEY") # type: ignore

    ALLOWED_ORIGINS: list = [
        FRONTEND_URL,
        "http://localhost:3000", 
        "http://127.0.0.1:5173",
        "http://localhost:8000"
    ]
    ALLOWED_HOSTS: list = [
        "localhost",
        "127.0.0.1"
    ]

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

    APP_MAIN: str = os.environ.get("APP_MAIN") # type: ignore
    APP_HOST: str = os.environ.get("APP_HOST") # type: ignore
    APP_PORT: str = os.environ.get("APP_PORT") # type: ignore

settings = Settings()