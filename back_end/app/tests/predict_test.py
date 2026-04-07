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
    url = f"{base_url}/api/predict/"

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

    valid_api_key = require_env("API_KEY")
    invalid_api_key = "wrong-api-key"

    valid_headers = {
        "X-API-Key": valid_api_key,
        "Content-Type": "application/json",
    }
    invalid_headers = {
        "X-API-Key": invalid_api_key,
        "Content-Type": "application/json",
    }

    def run_valid_key_test() -> bool:
        try:
            response = requests.post(url, headers=valid_headers, json=payload, timeout=20)
            print("=== Valid API key test ===")
            print(f"Status: {response.status_code}")
            print(get_json_or_text(response))
            return 200 <= response.status_code < 300
        except requests.RequestException as exc:
            print("=== Valid API key test ===")
            print(f"Request failed: {exc}")
            return False

    def run_invalid_key_test() -> bool:
        try:
            response = requests.post(url, headers=invalid_headers, json=payload, timeout=20)
            print("=== Invalid API key test ===")
            print(f"Status: {response.status_code}")
            print(get_json_or_text(response))
            return response.status_code in (401, 403)
        except requests.RequestException as exc:
            print("=== Invalid API key test ===")
            print(f"Request failed: {exc}")
            return False

    valid_ok = run_valid_key_test()
    invalid_ok = run_invalid_key_test()

    if valid_ok and invalid_ok:
        print("All prediction endpoint tests passed.")
        return 0

    print("One or more prediction endpoint tests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())