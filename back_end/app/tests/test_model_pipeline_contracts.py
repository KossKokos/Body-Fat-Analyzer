from __future__ import annotations

import itertools
import os
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np


os.environ["DEBUG"] = "false"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import constants as cnsts
from config.settings import settings
from services.prediction_service import PredictionService


ENGINEERED_FEATURES = (
    "hrr",
    "intensity_ratio",
    "hrr_per_age",
    "weekly_training_minutes",
    "weekly_training_hours",
    "calories_per_training_hour",
    "calories_per_minute",
    "calories_per_kg",
    "bmi_x_age",
    "bmi_x_training_hours",
    "protein_ratio",
    "carb_ratio",
    "fat_ratio",
    "sugar_per_calorie",
    "calories_per_meal",
    "high_intensity_workout",
    "strength_vs_cardio",
    "intensity_weighted_training_hours",
    "low_carb_diet",
    "plant_based_diet",
    "female_x_bmi",
    "male_x_bmi",
    "log1p_calories_burned",
    "log1p_weekly_training_hours",
    "log1p_calories_per_training_hour",
    "log1p_calories_per_kg",
    "log1p_calories",
)

EXPECTED_MODEL_FEATURES = (*cnsts.ENCODED_FEATURES, *ENGINEERED_FEATURES)

GENDERS = ("female", "male")
WORKOUT_TYPES = ("cardio", "hiit", "strength", "yoga")
DIET_TYPES = (
    "balanced",
    "keto",
    "low-carb",
    "paleo",
    "vegan",
    "vegetarian",
)


def make_payload(
    *,
    gender: str = "male",
    workout_type: str = "cardio",
    diet_type: str = "balanced",
) -> dict[str, Any]:
    return {
        "age": 30,
        "gender": gender,
        "weight": 75.0,
        "height": 1.78,
        "max_bpm": 185,
        "avg_bpm": 145,
        "resting_bpm": 62,
        "session_duration": 1.0,
        "calories_burned": 500,
        "workout_type": workout_type,
        "workout_frequency": 4.0,
        "experience_level": 2,
        "calories": 2200,
        "carbs": 250.0,
        "proteins": 140.0,
        "fats": 70.0,
        "sugar_g": 45.0,
        "diet_type": diet_type,
        "daily_meals_frequency": 3,
        "water_intake": 2.5,
    }


class IdentityScaler:
    def __init__(self) -> None:
        self.seen_shapes: list[tuple[int, ...]] = []

    def transform(self, values):
        array = np.asarray(values, dtype=np.float64)
        self.seen_shapes.append(array.shape)
        return array


class StaticPredictor:
    def __init__(self, output) -> None:
        self.output = np.asarray(output, dtype=np.float64)
        self.seen_shapes: list[tuple[int, ...]] = []

    def predict(self, values, *args, **kwargs):
        self.seen_shapes.append(np.asarray(values).shape)
        return self.output.copy()


class RecordingModelService:
    def __init__(self, classifier_output) -> None:
        self.classifier = StaticPredictor(classifier_output)
        self.classification_scaler = IdentityScaler()
        self.regression_scalers = {
            fat_class: IdentityScaler()
            for fat_class in ("low", "mid", "high")
        }
        self.base_models = {
            "low": StaticPredictor([14.0]),
            "mid": StaticPredictor([22.0]),
            "high": StaticPredictor([31.0]),
        }
        self.regressors = {
            "low": StaticPredictor([[0.25]]),
            "mid": StaticPredictor([[-0.50]]),
            "high": StaticPredictor([[0.75]]),
        }
        self.calls: list[tuple[str, str]] = []

    def get_classifier(self):
        return self.classifier

    def get_classification_scaler(self):
        return self.classification_scaler

    def get_regression_scaler(self, fat_class: str):
        self.calls.append(("scaler", fat_class))
        return self.regression_scalers[fat_class]

    def get_base_model(self, fat_class: str):
        self.calls.append(("base", fat_class))
        return self.base_models[fat_class]

    def get_regressor(self, fat_class: str):
        self.calls.append(("residual", fat_class))
        return self.regressors[fat_class]


class CanonicalFeatureSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = PredictionService(
            RecordingModelService([[0.8, 0.2]])
        )

    def test_model_feature_contract_contains_57_columns(self) -> None:
        self.assertEqual(len(cnsts.ENCODED_FEATURES), 30)
        self.assertEqual(len(ENGINEERED_FEATURES), 27)
        self.assertEqual(len(cnsts.MODEL_FEATURES), 57)
        self.assertTupleEqual(
            tuple(cnsts.MODEL_FEATURES),
            EXPECTED_MODEL_FEATURES,
        )

    def test_all_categorical_combinations_use_identical_column_order(self) -> None:
        combinations = itertools.product(
            GENDERS,
            WORKOUT_TYPES,
            DIET_TYPES,
        )

        for gender, workout_type, diet_type in combinations:
            with self.subTest(
                gender=gender,
                workout_type=workout_type,
                diet_type=diet_type,
            ):
                validated = self.service._validate_input(
                    make_payload(
                        gender=gender,
                        workout_type=workout_type,
                        diet_type=diet_type,
                    )
                )
                features = self.service._preprocess_features(validated)

                self.assertEqual(features.shape, (1, 57))
                self.assertTupleEqual(
                    tuple(features.columns),
                    tuple(cnsts.MODEL_FEATURES),
                )
                self.assertTrue(
                    np.isfinite(features.to_numpy(dtype=np.float64)).all()
                )

                self.assertEqual(
                    features.loc[0, [f"gender_{name}" for name in GENDERS]].sum(),
                    1.0,
                )
                self.assertEqual(
                    features.loc[
                        0,
                        [f"workout_type_{name}" for name in WORKOUT_TYPES],
                    ].sum(),
                    1.0,
                )
                self.assertEqual(
                    features.loc[
                        0,
                        [f"diet_type_{name}" for name in DIET_TYPES],
                    ].sum(),
                    1.0,
                )
                self.assertEqual(features.loc[0, f"gender_{gender}"], 1.0)
                self.assertEqual(
                    features.loc[0, f"workout_type_{workout_type}"],
                    1.0,
                )
                self.assertEqual(
                    features.loc[0, f"diet_type_{diet_type}"],
                    1.0,
                )


class FinalArtifactPathTests(unittest.TestCase):
    def test_all_configured_artifacts_use_final_names(self) -> None:
        artifact_paths = (
            settings.CLASSIFIER_PATH,
            *settings.REGRESSOR_PATHS.values(),
            *settings.BASE_MODELS_PATHS.values(),
            *settings.SCALERS_PATHS.values(),
        )

        self.assertEqual(len(artifact_paths), 11)
        for path in artifact_paths:
            with self.subTest(path=path):
                self.assertIn("_final.", path.name)


class ClassifierDecodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = PredictionService(
            RecordingModelService([[0.8, 0.2]])
        )

    def test_ordinal_decoder_uses_two_cumulative_thresholds(self) -> None:
        cases = (
            ([[0.49, 0.10]], 0),
            ([[0.80, 0.20]], 1),
            ([[0.90, 0.51]], 2),
        )

        for cumulative_probabilities, expected_class in cases:
            with self.subTest(
                cumulative_probabilities=cumulative_probabilities
            ):
                predicted_class = (
                    self.service._decode_classifier_prediction(
                        np.asarray(cumulative_probabilities),
                    )
                )
                self.assertEqual(predicted_class, expected_class)

    def test_decoder_rejects_mismatched_output_shapes(self) -> None:
        cases = (
            [[0.1, 0.8, 0.1]],
            [[0.1]],
            np.zeros((1, 2, 1)),
        )

        for predictions in cases:
            with self.subTest(predictions=np.asarray(predictions).shape):
                with self.assertRaises(ValueError):
                    self.service._decode_classifier_prediction(
                        np.asarray(predictions),
                    )


class MockedPipelineRoutingTests(unittest.TestCase):
    def test_pipeline_routes_to_the_ordinal_selected_regressor(self) -> None:
        model_service = RecordingModelService(
            [[0.80, 0.20]],
        )
        service = PredictionService(model_service)

        result = service.predict_fat_percentage(make_payload())

        self.assertEqual(result["fat_class"], "mid")
        self.assertEqual(result["fat_percentage"], 21.5)
        self.assertListEqual(
            model_service.calls,
            [
                ("scaler", "mid"),
                ("base", "mid"),
                ("residual", "mid"),
            ],
        )
        self.assertListEqual(
            model_service.classification_scaler.seen_shapes,
            [(1, 57)],
        )
        self.assertListEqual(
            model_service.regression_scalers["mid"].seen_shapes,
            [(1, 57)],
        )


if __name__ == "__main__":
    unittest.main()
