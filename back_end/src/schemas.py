from pydantic import BaseModel, Field


class DataSchema(BaseModel):
    age: int = Field(gt=0, le=100)
    gender: str = Field(pattern=r'^male$|^female$')
    weight: float = Field(gt=2.0, le=635.0)
    height: float = Field(ge=0.2, le=2.72)
    max_bpm: int = Field(ge=25, le=600)
    resting_bpm: int = Field(ge=25, le=600)
    session_duration: float = Field(ge=0.1, le=4.0)
    calories_burned: int = Field(ge=10)
    workout_type: str = Field(pattern=r"^strength$|^yoga$|^hiit$|^cardio$")
    water_intake: float = Field(ge=0.0, le=20.0)
    workout_frequency: float = Field(ge=0.0, le=7.0)
    daily_meals_frequency: float = Field(ge=0, le=10)

class PredictionResponce(BaseModel):
    fat_percentage: float