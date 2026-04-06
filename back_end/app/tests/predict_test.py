from __future__ import annotations

import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_json_or_text(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


def main() -> int:
    load_dotenv()

    base_url = os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/")
    api_key = require_env("API_KEY")

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    payload = {
        "age": 30,
        "gender": "male",
        "weight": 75.0,
        "height": 1.78,
        "max_bpm": 185,
        "avg_bpm": 145,
        "resting_bpm": 62,
        "session_duration": 1.0,
        "calories_burned": 500,
        "workout_type": "cardio",
        "workout_frequency": 4.0,
        "experience_level": 2,
        "calories": 2200,
        "carbs": 250.0,
        "proteins": 140.0,
        "fats": 70.0,
        "sugar_g": 45.0,
        "diet_type": "balanced",
        "daily_meals_frequency": 3,
        "water_intake": 2.5,
    }

    url = f"{base_url}/api/predict/"

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        print(f"Status: {response.status_code}")
        print(get_json_or_text(response))
        return 0 if 200 <= response.status_code < 300 else 1
    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())