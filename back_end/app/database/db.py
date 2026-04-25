from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from config.logger import logger

def _build_connect_args() -> dict:
    connect_args = {
        "application_name": "fat_predictor_api",
    }

    if settings.DB_SSL_MODE and settings.DB_SSL_MODE != "disable":
        connect_args["sslmode"] = settings.DB_SSL_MODE

    return connect_args

# Check for temporary startup log
safe_db_url = settings.SQLALCHEMY_DATABASE_URL.split("@")[-1]
logger.info(f"Database URL driver check: {settings.SQLALCHEMY_DATABASE_URL.split('://')[0]}://***@{safe_db_url}")

engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=settings.DB_POOL_PRE_PING,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=settings.DB_POOL_RECYCLE,
    echo=settings.DB_ECHO,
    connect_args=_build_connect_args(),
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)