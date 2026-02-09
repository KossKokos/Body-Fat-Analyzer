from contextlib import contextmanager

from fastapi import Request
from sqlalchemy.orm import Session

from database.db import SessionLocal


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