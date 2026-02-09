import enum

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Numeric,
    Enum,
    CheckConstraint,
    ForeignKey,
    func,
)
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


# -----------------------
# Enums (PostgreSQL)
# -----------------------
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


# -----------------------
# Main table
# -----------------------
class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True)

    # -----------------------
    # User basics
    # -----------------------
    age = Column(Integer, nullable=False)
    gender = Column(
        Enum(GenderEnum, name="gender_enum"), 
        nullable=False
        )

    # -----------------------
    # Body metrics
    # -----------------------
    weight = Column(Numeric(5, 2), nullable=False)   # kg
    height = Column(Numeric(4, 2), nullable=False)   # meters

    # -----------------------
    # Heart rate
    # -----------------------
    max_bpm = Column(Integer, nullable=False)
    avg_bpm = Column(Integer, nullable=False)
    resting_bpm = Column(Integer, nullable=False)

    # -----------------------
    # Training
    # -----------------------
    session_duration = Column(Numeric(3, 2), nullable=False)  # hours
    calories_burned = Column(Integer, nullable=False)

    workout_type = Column(
        Enum(WorkoutTypeEnum, name="workout_type_enum"),
        nullable=False,
    )

    workout_frequency = Column(Numeric(3, 1), nullable=False)  # per week
    experience_level = Column(Integer, nullable=False)

    # -----------------------
    # Nutrition
    # -----------------------
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
    water_intake = Column(Numeric(4, 1), nullable=False)  # liters

    # -----------------------
    # ML output
    # -----------------------
    prediction = Column(Numeric(5, 2), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    model_version = Column(String, nullable=False)

    # -----------------------
    # Constraints
    # -----------------------
    __table_args__ = (
        CheckConstraint("age BETWEEN 1 AND 100"),
        CheckConstraint("weight BETWEEN 2 AND 635"),
        CheckConstraint("height BETWEEN 0.2 AND 2.72"),

        CheckConstraint("max_bpm BETWEEN 60 AND 230"),
        CheckConstraint("avg_bpm BETWEEN 40 AND 200"),
        CheckConstraint("resting_bpm BETWEEN 30 AND 120"),

        CheckConstraint("session_duration BETWEEN 0.1 AND 3.0"),
        CheckConstraint("calories_burned BETWEEN 10 AND 5000"),
        CheckConstraint("workout_frequency BETWEEN 0 AND 14"),
        CheckConstraint("experience_level BETWEEN 1 AND 3"),

        CheckConstraint("calories BETWEEN 500 AND 10000"),
        CheckConstraint("daily_meals_frequency BETWEEN 1 AND 10"),
        CheckConstraint("water_intake BETWEEN 0 AND 20"),
    )


class PredictionFeedback(Base):
    __tablename__ = "prediction_feedback"

    id = Column(Integer, primary_key=True)

    prediction_id = Column(
        Integer,
        ForeignKey(
            "prediction_history.id",
            ondelete="CASCADE"
        ),
        nullable=False,
        index=True,
    )

    rating = Column(
        Integer,
        nullable=False,
        server_default="5",
    )

    comment = Column(
        String(500),
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint(
            "rating BETWEEN 0 AND 10",
            name="ck_prediction_feedback_rating_range",
        ),
    )

    # Optional ORM convenience
    prediction = relationship(
        "PredictionHistory",
        backref="feedback",
        passive_deletes=True,
    )

