import numpy as np
import pandas as pd

import src.schemas as schemas
from src.get_trained_instances import model, x_scaler, y_scaler
from src.user_history import history, columns, add_new_record

history_cols = columns
independent_features = columns[:-1]

scaler_order = ['calories_burned', 'max_bpm', 'age', 'weight', 'daily_meals_frequency',
       'resting_bpm', 'bmi', 'workout_frequency', 'water_intake',
       'session_duration', 'height', 'gender_female', 'gender_male',
       'workout_type_cardio', 'workout_type_hiit', 'workout_type_strength',
       'workout_type_yoga', 'hrr', 'intensity_ratio', 'hrr_per_age',
       'weekly_training_hours', 'calories_per_training_hour',
       'calories_per_kg', 'water_per_kg', 'calories_per_meal', 'bmi_x_age',
       'bmi_x_traininghours', 'high_intensity_workout', 'strength_vs_cardio',
       'intensity_weighted_training_hours', 'type_adjusted_calories',
       'age_x_traininghours', 'hrr_x_traininghours', 'bmi_x_caloriesperkg',
       'log1p_calories_burned', 'log1p_weekly_training_hours',
       'log1p_calories_per_training_hour', 'log1p_calories_per_kg']

async def validate_data(data: schemas.DataSchema):
    dict_data = data.model_dump()
    dict_data['bmi'] = dict_data['weight'] / (dict_data['height'])**2
    

    desired_order_list = independent_features
    
    reordered_data = {k: dict_data[k] for k in desired_order_list}
    print('Reorder success')
    
    orig_data = list(reordered_data.values())

    X = pd.DataFrame([orig_data], columns=independent_features)
    print('X created')
    X = pd.get_dummies(X)
    return orig_data, X

async def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new engineered features for predicting fat_percentage.
    Assumes columns are lower-case, e.g. 'max_bpm', 'resting_bpm', etc.
    """
    df = df.copy()
    eps = 1e-9  # avoid division by zero

    # ---------- Core physiology / intensity proxies ----------
    df["hrr"] = df["max_bpm"] - df["resting_bpm"]
    df["hrr"] = df["hrr"].clip(lower=0)

    df["intensity_ratio"] = df["resting_bpm"] / (df["max_bpm"] + eps)
    df["hrr_per_age"] = df["hrr"] / (df["age"] + eps)

    # ---------- Training load / volume proxies ----------
    df["weekly_training_hours"] = df["workout_frequency"] * df["session_duration"]
    df["weekly_training_hours"] = df["weekly_training_hours"].clip(lower=0)

    df["calories_per_training_hour"] = df["calories_burned"] / (df["weekly_training_hours"] + eps)
    df["calories_per_kg"] = df["calories_burned"] / (df["weight"] + eps)

    # ---------- Lifestyle proxies ----------
    df["water_per_kg"] = df["water_intake"] / (df["weight"] + eps)
    df["calories_per_meal"] = df["calories_burned"] / (df["daily_meals_frequency"] + eps)

    df["bmi_x_age"] = df["bmi"] * df["age"]
    df["bmi_x_traininghours"] = df["bmi"] * df["weekly_training_hours"]

    # ---------- Workout type composites ----------
    for col in [
        "workout_type_cardio", "workout_type_hiit",
        "workout_type_strength", "workout_type_yoga",
        "gender_female", "gender_male"
    ]:
        if col not in df.columns:
            df[col] = 0

    df["high_intensity_workout"] = (
        (df["workout_type_cardio"] == 1) | (df["workout_type_hiit"] == 1)
    ).astype(float)

    df["strength_vs_cardio"] = df["workout_type_strength"] - df["workout_type_cardio"]

    df["intensity_weighted_training_hours"] = df["weekly_training_hours"] * (
        1.6 * df["workout_type_hiit"]
        + 1.2 * df["workout_type_cardio"]
        + 1.2 * df["workout_type_strength"]
        + 0.8 * df["workout_type_yoga"]
    )

    df["type_adjusted_calories"] = df["calories_burned"] * (
        1.5 * df["workout_type_hiit"]
        + 1.2 * df["workout_type_cardio"]
        + 1.1 * df["workout_type_strength"]
            + 0.9 * df["workout_type_yoga"]
    )

    # ---------- Interaction features ----------
    df["age_x_traininghours"] = df["age"] * df["weekly_training_hours"]
    df["hrr_x_traininghours"] = df["hrr"] * df["weekly_training_hours"]
    df["bmi_x_caloriesperkg"] = df["bmi"] * df["calories_per_kg"]

    # ---------- Log transforms ----------
    for col in [
        "calories_burned",
        "weekly_training_hours",
        "calories_per_training_hour",
        "calories_per_kg",
    ]:
        df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))

    print("New features added")
    return df


async def transform_x(x: pd.DataFrame):
    try:
        x_transformed = x_scaler.transform(x)
    except ValueError:
        x = x.loc[:, scaler_order]
        x_transformed = x_scaler.transform(x)
    print('X transformed')
    return x_transformed

async def transform_y(y):
    y_transformed = y_scaler.transform(y)
    print('y transformed')
    return y_transformed

async def inverse_transform_y(y):
    y_orig = y_scaler.inverse_transform(y)
    print('y inv transformed')
    return y_orig

async def get_predictions(x) -> float:
    predictions = model.predict(x)
    predictions_inv_transformed = await inverse_transform_y(y=predictions)
    prediction = float(predictions_inv_transformed.item())
    
    print('predictins were made')
    return prediction 

async def add_to_history(x, y: float) -> None:
    array_x = np.array(x).flatten()
    array_y = np.array(y).flatten()
    print(array_x, array_x.shape, array_x.ndim)
    print(array_y, array_y.shape, array_y.ndim)

    new_row = np.concat([array_x, array_y], axis=0).flatten()
    print('new row was made through concatenation')

    new_row = pd.DataFrame([new_row], columns=history_cols)
    print('new row for history is created')
    add_new_record(new_row=new_row)
    print('new record is added')
