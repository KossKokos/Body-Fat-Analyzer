import ssl

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import settings


def get_connect_args() -> dict:
    db_url = settings.SQLALCHEMY_DATABASE_URL
    db_ssl_mode = str(settings.DB_SSL_MODE).strip().lower()

    is_pg8000 = db_url.startswith("postgresql+pg8000://")
    ssl_required = db_ssl_mode == "require"

    print(f"APP DB_SSL_MODE={db_ssl_mode}", flush=True)
    print(f"APP DB DRIVER={db_url.split('://', 1)[0]}", flush=True)
    print(f"APP SSL REQUIRED={ssl_required}", flush=True)

    if is_pg8000 and ssl_required:
        return {
            "ssl_context": ssl.create_default_context(),
        }

    return {}


engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=settings.DB_POOL_PRE_PING,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=settings.DB_POOL_RECYCLE,
    echo=settings.DB_ECHO,
    connect_args=get_connect_args(),
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)