#!/usr/bin/env sh
set -e

echo "Waiting for PostgreSQL..."
until pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do
  sleep 2
done

echo "Applying database migrations..."
alembic -c alembic.ini upgrade head

echo "Starting FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"