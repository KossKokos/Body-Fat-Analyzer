from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    age: int = Field(gt=0, le=100)
    gender: Literal["male", "female"]

    weight: float = Field(gt=2.0, le=635.0)
    height: float = Field(ge=0.2, le=2.72)

    max_bpm: int = Field(ge=60, le=230)
    avg_bpm: int = Field(ge=40, le=200)
    resting_bpm: int = Field(ge=30, le=120)

    session_duration: float = Field(ge=0.1, le=3.0)
    calories_burned: float = Field(ge=10, le=5000)

    workout_type: Literal["cardio", "hiit", "strength", "yoga"]
    workout_frequency: float = Field(ge=0.0, le=14.0)
    experience_level: int = Field(ge=1, le=3)

    calories: float = Field(ge=500, le=10000)
    carbs: float = Field(ge=0, le=1500)
    proteins: float = Field(ge=0, le=500)
    fats: float = Field(ge=0, le=500)
    sugar_g: float = Field(ge=0, le=1000)

    diet_type: Literal["vegan", "vegetarian", "paleo", "keto", "low-carb", "balanced"]
    daily_meals_frequency: float = Field(ge=1, le=10)
    water_intake: float = Field(ge=0.0, le=20.0)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    prediction_id: int
    fat_class: Literal["low", "mid", "high"]
    fat_percentage: float = Field(ge=0, le=100)
    timestamp: datetime


class PredictionFeedbackCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prediction_id: int = Field(gt=0)
    rating: float= Field(ge=0, le=10)
    is_prediction_close: bool | None = None
    actual_fat_percentage: float | None = Field(default=None, ge=0, le=100)
    comment: str | None = Field(default=None, max_length=2000)
    consent_to_retrain: bool = False


class PredictionFeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: int
    prediction_id: int
    rating: int
    is_prediction_close: bool | None = None
    actual_fat_percentage: float | None = None
    comment: str | None = None
    consent_to_retrain: bool
    created_at: datetime