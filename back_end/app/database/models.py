import enum

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Numeric,
    Enum,
    CheckConstraint,
    ForeignKey,
    func,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class GenderEnum(enum.Enum):
    male = "male"
    female = "female"


class WorkoutTypeEnum(enum.Enum):
    cardio = "cardio"
    hiit = "hiit"
    strength = "strength"
    yoga = "yoga"


class DietTypeEnum(enum.Enum):
    vegan = "vegan"
    vegetarian = "vegetarian"
    paleo = "paleo"
    keto = "keto"
    low_carb = "low-carb"
    balanced = "balanced"


class FatClassEnum(enum.Enum):
    low = "low"
    mid = "mid"
    high = "high"


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)

    age = Column(Integer, nullable=False)
    gender = Column(Enum(GenderEnum, name="gender_enum"), nullable=False)

    weight = Column(Numeric(5, 2), nullable=False)
    height = Column(Numeric(4, 2), nullable=False)

    max_bpm = Column(Integer, nullable=False)
    avg_bpm = Column(Integer, nullable=False)
    resting_bpm = Column(Integer, nullable=False)

    session_duration = Column(Numeric(3, 2), nullable=False)
    calories_burned = Column(Integer, nullable=False)

    workout_type = Column(
        Enum(WorkoutTypeEnum, name="workout_type_enum"),
        nullable=False,
    )

    workout_frequency = Column(Numeric(3, 1), nullable=False)
    experience_level = Column(Integer, nullable=False)

    calories = Column(Integer, nullable=False)
    carbs = Column(Numeric(6, 1), nullable=False)
    proteins = Column(Numeric(6, 1), nullable=False)
    fats = Column(Numeric(6, 1), nullable=False)
    sugar_g = Column(Numeric(6, 1), nullable=False)

    diet_type = Column(
        Enum(DietTypeEnum, name="diet_type_enum"),
        nullable=False,
    )

    daily_meals_frequency = Column(Integer, nullable=False)
    water_intake = Column(Numeric(4, 1), nullable=False)

    fat_class = Column(
        Enum(FatClassEnum, name="fat_class_enum"),
        nullable=False,
        index=True,
    )
    fat_percentage = Column(Numeric(5, 2), nullable=False)

    model_version = Column(String(50), nullable=False, index=True)

    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False,
        index=True)
    
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    feedback_items = relationship(
        "PredictionFeedback",
        back_populates="prediction",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint("age BETWEEN 1 AND 100", name="ck_prediction_history_age"),
        CheckConstraint("weight BETWEEN 2 AND 635", name="ck_prediction_history_weight"),
        CheckConstraint("height BETWEEN 0.2 AND 2.72", name="ck_prediction_history_height"),

        CheckConstraint("max_bpm BETWEEN 60 AND 230", name="ck_prediction_history_max_bpm"),
        CheckConstraint("avg_bpm BETWEEN 40 AND 200", name="ck_prediction_history_avg_bpm"),
        CheckConstraint("resting_bpm BETWEEN 30 AND 120", name="ck_prediction_history_resting_bpm"),

        CheckConstraint("session_duration BETWEEN 0.1 AND 3.0", name="ck_prediction_history_session_duration"),
        CheckConstraint("calories_burned BETWEEN 10 AND 5000", name="ck_prediction_history_calories_burned"),
        CheckConstraint("workout_frequency BETWEEN 0 AND 14", name="ck_prediction_history_workout_frequency"),
        CheckConstraint("experience_level BETWEEN 1 AND 3", name="ck_prediction_history_experience_level"),

        CheckConstraint("calories BETWEEN 500 AND 10000", name="ck_prediction_history_calories"),
        CheckConstraint("daily_meals_frequency BETWEEN 1 AND 10", name="ck_prediction_history_daily_meals_frequency"),
        CheckConstraint("water_intake BETWEEN 0 AND 20", name="ck_prediction_history_water_intake"),

        CheckConstraint("fat_percentage BETWEEN 0 AND 100", name="ck_prediction_history_fat_percentage"),
    )


class PredictionFeedback(Base):
    __tablename__ = "prediction_feedback"

    id = Column(Integer, primary_key=True, index=True)

    prediction_id = Column(
        Integer,
        ForeignKey("prediction_history.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    rating = Column(Integer, nullable=False)
    is_prediction_close = Column(Boolean, nullable=True)
    actual_fat_percentage = Column(Numeric(5, 2), nullable=True)
    comment = Column(Text, nullable=True)

    # if user gives consent to store data
    consent_to_retrain = Column(Boolean, nullable=False, server_default="false")
    consent_timestamp = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    prediction = relationship(
        "PredictionHistory",
        back_populates="feedback_items",
    )

    __table_args__ = (
        CheckConstraint(
            "rating BETWEEN 0 AND 10",
            name="ck_prediction_feedback_rating_range",
        ),
        CheckConstraint(
            "actual_fat_percentage IS NULL OR actual_fat_percentage BETWEEN 0 AND 100",
            name="ck_prediction_feedback_actual_fat_percentage_range",
        )
    )