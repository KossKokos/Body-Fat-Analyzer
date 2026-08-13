import os
import json

from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

env_file = Path(__file__).parent.parent.parent / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=env_file,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DB CREDENTIALS
    POSTGRES_DB: str = ""
    POSTGRES_USER: str = ""
    POSTGRES_PASSWORD: str = ""
    POSTGRES_PORT: str = ""
    POSTGRES_HOST: str = ""

    SQLALCHEMY_DATABASE_URL: str = ""    
    
    # DB ATTRIBUTES
    DB_SSL_MODE: str = "disable"
    DB_POOL_PRE_PING: bool = True
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE: int = 1800
    DB_ECHO: bool = False

    # DB INSTANCES
    REQUIRED_TABLES: List[str] = []
    
    # APP START UP
    APP_MAIN: str = ""
    APP_HOST: str = ""
    APP_PORT: str = ""

    # PROJECT INIT
    PROJECT_NAME: str = ""
    PROJECT_VERSION: str = ""

    # DEVELOPMENT
    ENVIRONMENT: str = ""
    DEBUG: bool = False
    DOCS: bool = False

    # URLS
    API_V1_STR: str = ""
    API_PREFIX: str = ""
    FRONTEND_URL: str = ""
    API_PREFIX: str = ""
    
    # API SECURITY
    API_KEY: str = ""
    ALLOWED_ORIGINS: List[str] = []
    ALLOWED_HOSTS: List[str] = []
    ALLOW_METHODS: List[str] = []
    ALLOW_HEADERS: List[str] = []
    EXPOSE_HEADERS: List[str] = []
    MAX_AGE: int = 800
    CALLS_PER_MINUTE: int = 100

    # LOGGER
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False
    LOG_TO_FILE: bool = False
    LOG_FILE: Optional[str] = "logs/app.log"

    # MODEL INFO
    MODEL_VERSION: str = ""

    # MODELS LOCATION
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "app" / "models"
    DATA_DIR: Path = BASE_DIR / "data"

    CLASSIFIER_PATH: Path = (
        MODELS_DIR
        / "classifiers"
        / "fat_percentage_ordinal_classifier_final.keras"
    )
    REGRESSOR_PATHS: dict = {
        "low": (
            MODELS_DIR
            / "regressors"
            / "low_fat_residuals_regressor_final.keras"
        ),
        "mid": (
            MODELS_DIR
            / "regressors"
            / "mid_fat_boundary_weighted_residuals_regressor_final.keras"
        ),
        "high": (
            MODELS_DIR
            / "regressors"
            / "high_fat_residuals_regressor_final.keras"
        ),
    }
    BASE_MODELS_PATHS: dict = {
        "low": (
            MODELS_DIR / "base_models" / "low_fat_base_model_final.pkl"
        ),
        "mid": (
            MODELS_DIR / "base_models" / "mid_fat_base_model_final.pkl"
        ),
        "high": (
            MODELS_DIR / "base_models" / "high_fat_base_model_final.pkl"
        ),
    }
    SCALERS_PATHS: dict = {
        "class": (
            MODELS_DIR / "scalers" / "classification_scaler_final.pkl"
        ),
        "low": MODELS_DIR / "scalers" / "low_fat_scaler_final.pkl",
        "mid": MODELS_DIR / "scalers" / "mid_fat_scaler_final.pkl",
        "high": MODELS_DIR / "scalers" / "high_fat_scaler_final.pkl",
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

    @field_validator(
        "ALLOWED_ORIGINS",
        "ALLOW_METHODS",
        "ALLOW_HEADERS",
        "EXPOSE_HEADERS",
        "ALLOWED_HOSTS",
        "REQUIRED_TABLES",
        mode="before",
    )
    @classmethod
    def parse_json_list(cls, value):
        if isinstance(value, str):
            return json.loads(value)
        return value

settings = Settings()
