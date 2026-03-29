from contextlib import contextmanager

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from config.settings import settings
from database.db import SessionLocal

# API Key dependency (can be used on specific endpoints too)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Dependency to verify API key on specific endpoints"""
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
    return api_key


def get_prediction_service(request: Request):
    return request.app.state.prediction_service

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def db_transaction(session: Session):
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()