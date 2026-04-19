from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv

from predict_test import require_env, get_json_or_text


def create_prediction(base_url: str, headers: dict[str, str]) -> int:
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
    response = requests.post(url, headers=headers, json=payload, timeout=20)

    print("Prediction status:", response.status_code)
    print(get_json_or_text(response))

    if not (200 <= response.status_code < 300):
        raise RuntimeError("Prediction request failed")

    data = response.json()
    prediction_id = data.get("prediction_id")
    if not prediction_id:
        raise RuntimeError("Prediction response did not include prediction_id")

    return int(prediction_id)


def main() -> int:
    load_dotenv()

    base_url = os.getenv("APP_BASE_URL", "http://localhost:8000").rstrip("/")
    api_key = require_env("API_KEY")

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    try:
        prediction_id = create_prediction(base_url, headers)

        payload = {
            "prediction_id": prediction_id,
            "rating": 8,
            "is_prediction_close": True,
            "actual_fat_percentage": 23.2,
            "comment": "Nice",
            "consent_to_retrain": True,
        }

        url = f"{base_url}/api/feedback/"
        response = requests.post(url, headers=headers, json=payload, timeout=20)

        print(f"Feedback status: {response.status_code}")
        print(get_json_or_text(response))

        return 0 if 200 <= response.status_code < 300 else 1

    except requests.RequestException as exc:
        print(f"Request failed: {exc}")
        return 1
    except Exception as exc:
        print(f"Test failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())