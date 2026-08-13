"""Opt-in smoke test for the complete final on-disk model bundle.

Run with ``RUN_MODEL_ARTIFACT_SMOKE=1`` because loading every neural artifact
is intentionally heavier than the default contract suite.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


os.environ["DEBUG"] = "false"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config.settings import settings
from services.model_service import ModelService
from services.prediction_service import PredictionService


SAMPLE_PAYLOAD = {
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


@unittest.skipUnless(
    os.getenv("RUN_MODEL_ARTIFACT_SMOKE") == "1",
    "Set RUN_MODEL_ARTIFACT_SMOKE=1 to load the real model bundles.",
)
class ModelArtifactSmokeTests(unittest.TestCase):
    def test_final_bundle_loads_and_predicts(self) -> None:
        try:
            artifact_paths = (
                settings.CLASSIFIER_PATH,
                *settings.REGRESSOR_PATHS.values(),
                *settings.BASE_MODELS_PATHS.values(),
                *settings.SCALERS_PATHS.values(),
            )
            for path in artifact_paths:
                with self.subTest(path=path):
                    self.assertTrue(path.is_file(), f"Missing artifact: {path}")

            model_service = ModelService()
            model_service.load_models()
            result = PredictionService(
                model_service
            ).predict_fat_percentage(SAMPLE_PAYLOAD)

            self.assertIn(
                result["fat_class"],
                ("low", "mid", "high"),
            )
            self.assertGreaterEqual(
                result["fat_percentage"],
                PredictionService.MIN_FAT_PERCENTAGE,
            )
            self.assertLessEqual(
                result["fat_percentage"],
                PredictionService.MAX_FAT_PERCENTAGE,
            )
        finally:
            ModelService().models.clear()


if __name__ == "__main__":
    unittest.main()
