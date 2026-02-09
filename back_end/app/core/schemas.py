from typing import Literal
from datetime import datetime

from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    # -----------------------
    # Demographics
    # -----------------------
    age: int = Field(gt=0, le=100, description="Age in years")

    gender: Literal["male", "female"]

    # -----------------------
    # Body metrics
    # -----------------------
    weight: float = Field(gt=2.0, le=635.0, description="Weight in kg")
    height: float = Field(ge=0.2, le=2.72, description="Height in meters")

    # -----------------------
    # Heart rate metrics
    # -----------------------
    max_bpm: int = Field(ge=60, le=230, description="Maximum heart rate")
    avg_bpm: int = Field(ge=40, le=200, description="Average workout BPM")
    resting_bpm: int = Field(ge=30, le=120, description="Resting heart rate")

    # -----------------------
    # Training
    # -----------------------
    session_duration: float = Field(
        ge=0.1, le=3.0, description="Workout session duration in hours"
    )

    calories_burned: float = Field(
        ge=10, le=5000, description="Calories burned per day"
    )

    workout_type: Literal[
        "cardio", "hiit", "strength", "yoga"
    ]

    workout_frequency: float = Field(
        ge=0.0, le=14.0, description="Workouts per week"
    )

    experience_level: int = Field(
        ge=1, le=3, description="1=beginner, 2=intermediate, 3=advanced"
    )
    # -----------------------
    # Nutrition
    # -----------------------
    calories: float = Field(
        ge=500, le=10000, description="Daily calorie intake"
    )

    carbs: float = Field(ge=0, le=1500, description="Daily carbs (g)")
    proteins: float = Field(ge=0, le=500, description="Daily protein (g)")
    fats: float = Field(ge=0, le=500, description="Daily fat (g)")
    sugar_g: float = Field(ge=0, le=1000, description="Daily sugar intake (g)")

    diet_type: Literal[
        "vegan", "vegetarian", "paleo", "keto," "low-carb", "balanced"
    ]
 
    daily_meals_frequency: float = Field(ge=1, le=10)

    water_intake: float = Field(
        ge=0.0, le=20.
    )

class PredictionResponse(BaseModel):
    fat_class: Literal[
        "low", "mid", "high"
    ]
    fat_percentage: float
    timestamp: datetime



features = [
        'age',
        'gender',
        'weight',
        'height',
        'max_bpm',
        'avg_bpm',
        'resting_bpm',
        'session_duration',
        'calories_burned',
        'workout_type',
        'water_intake',
        'workout_frequency',
        'experience_level',
        'bmi',
        'daily_meals_frequency',
        'carbs',
        'proteins',
        'fats',
        'calories',
        'diet_type',
        'sugar_g',]

