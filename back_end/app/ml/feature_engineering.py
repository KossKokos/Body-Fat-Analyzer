"""Feature engineering shared by model training and inference."""

import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 27 derived features used by the final model bundle."""

    features = df.copy()
    epsilon = 1e-9

    features["hrr"] = (
        features["max_bpm"] - features["resting_bpm"]
    ).clip(lower=0)
    features["intensity_ratio"] = (
        features["avg_bpm"] / (features["max_bpm"] + epsilon)
    )
    features["hrr_per_age"] = (
        features["hrr"] / (features["age"] + epsilon)
    )

    features["weekly_training_minutes"] = (
        features["workout_frequency"] * features["session_duration"]
    ).clip(lower=0)
    features["weekly_training_hours"] = (
        features["weekly_training_minutes"] / 60
    )
    features["calories_per_training_hour"] = (
        features["calories_burned"]
        / (features["weekly_training_hours"] + epsilon)
    )
    features["calories_per_minute"] = (
        features["calories_burned"]
        / (features["session_duration"] + epsilon)
    )
    features["calories_per_kg"] = (
        features["calories_burned"] / (features["weight"] + epsilon)
    )

    features["bmi_x_age"] = features["bmi"] * features["age"]
    features["bmi_x_training_hours"] = (
        features["bmi"] * features["weekly_training_hours"]
    )

    features["protein_ratio"] = (
        features["proteins"] / (features["calories"] + epsilon)
    )
    features["carb_ratio"] = (
        features["carbs"] / (features["calories"] + epsilon)
    )
    features["fat_ratio"] = (
        features["fats"] / (features["calories"] + epsilon)
    )
    features["sugar_per_calorie"] = (
        features["sugar_g"] / (features["calories"] + epsilon)
    )
    features["calories_per_meal"] = (
        features["calories"]
        / (features["daily_meals_frequency"] + epsilon)
    )

    features["high_intensity_workout"] = (
        (features["workout_type_hiit"] == 1)
        | (features["workout_type_cardio"] == 1)
    ).astype(float)
    features["strength_vs_cardio"] = (
        features["workout_type_strength"]
        - features["workout_type_cardio"]
    )
    features["intensity_weighted_training_hours"] = features[
        "weekly_training_hours"
    ] * (
        1.6 * features["workout_type_hiit"]
        + 1.2 * features["workout_type_cardio"]
        + 1.2 * features["workout_type_strength"]
        + 0.8 * features["workout_type_yoga"]
    )

    features["low_carb_diet"] = (
        features["diet_type_keto"]
        + features["diet_type_low-carb"]
        + features["diet_type_paleo"]
    ).clip(upper=1)
    features["plant_based_diet"] = (
        features["diet_type_vegan"]
        + features["diet_type_vegetarian"]
    ).clip(upper=1)

    features["female_x_bmi"] = (
        features["gender_female"] * features["bmi"]
    )
    features["male_x_bmi"] = (
        features["gender_male"] * features["bmi"]
    )

    for column in (
        "calories_burned",
        "weekly_training_hours",
        "calories_per_training_hour",
        "calories_per_kg",
        "calories",
    ):
        features[f"log1p_{column}"] = np.log1p(
            features[column].clip(lower=0)
        )

    return features
