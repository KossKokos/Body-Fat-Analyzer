#!/usr/bin/env sh
set -e

echo "Runtime DB config check..."
python - <<'PY'
import os

db_url = os.getenv("SQLALCHEMY_DATABASE_URL", "")
ssl_mode = os.getenv("DB_SSL_MODE", "")

safe_url = db_url
if "@" in safe_url:
    safe_url = "***@" + safe_url.split("@", 1)[1]

print(f"DB_SSL_MODE={ssl_mode!r}")
print(f"SQLALCHEMY_DATABASE_URL={safe_url}")
print(f"URL_DRIVER={db_url.split('://', 1)[0] if '://' in db_url else 'INVALID'}")
print(f"URL_HAS_SSLMODE={'sslmode=' in db_url}")
PY

echo "Waiting for PostgreSQL..."
until pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do
  sleep 2
done

echo "Applying database migrations..."
alembic -c alembic.ini upgrade head

echo "Starting FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"