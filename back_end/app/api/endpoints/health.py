from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text, bindparam
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.dependencies import get_db
from config.settings import settings
from config.logger import logger

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
def app_health_check():
    return {"status": "ok"}


@router.get("/db")
def db_health_check(
    db: Session = Depends(get_db),
):

    try:
        db.execute(text("SELECT 1")).scalar_one()

        required_tables = sorted(settings.REQUIRED_TABLES)
        stmt = text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN :required_tables
        """).bindparams(bindparam("required_tables", expanding=True))

        rows = db.execute(stmt, {"required_tables": required_tables}).fetchall()
        found_tables = sorted([row[0] for row in rows])
        
        if found_tables != required_tables:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service unavailable",
            )
        for table in required_tables:
            db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()

        return {"status": "ok"}

    except HTTPException:
        raise
    except SQLAlchemyError as exc:
        print(f"Database health check failed: {repr(exc)}", flush=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable",
        ) from exc