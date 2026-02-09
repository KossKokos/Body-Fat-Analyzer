import os
import random
import pickle

from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import Lasso


def read_data(filename: Path):
  # reading data
  data = pd.read_csv(filename)
  # renaming columns
  data = data.rename(columns={
    'Age': 'age',
    'Gender': 'gender',
    'Weight (kg)': 'weight',
    'Height (m)': 'height',
    'Max_BPM': 'max_bpm',
    'Avg_BPM': 'avg_bpm',
    'Resting_BPM': 'resting_bpm',
    'Session_Duration (hours)': 'session_duration',
    'Calories_Burned': 'calories_burned',
    'Workout_Type': 'workout_type',
    'Fat_Percentage': 'fat_percentage',
    'Water_Intake (liters)': 'water_intake',
    'Workout_Frequency (days/week)': 'workout_frequency',
    'Experience_Level': 'experience_level',
    'BMI': 'bmi',
    'Daily meals frequency': 'daily_meals_frequency',
    'Physical exercise': 'physical_exercise',

    'Carbs': 'carbs',
    'Proteins': 'proteins',
    'Fats': 'fats',
    'Calories': 'calories',

    'meal_name': 'meal_name',
    'meal_type': 'meal_type',
    'diet_type': 'diet_type',

    'sugar_g': 'sugar_g',
    'sodium_mg': 'sodium_mg',
    'cholesterol_mg': 'cholesterol_mg',
    'serving_size_g': 'serving_size_g',
    'cooking_method': 'cooking_method',
    'prep_time_min': 'prep_time_min',
    'cook_time_min': 'cook_time_min',
    'rating': 'rating',

    'Name of Exercise': 'exercise_name',
    'Sets': 'sets',
    'Reps': 'reps',
    'Benefit': 'benefit',
    'Burns Calories (per 30 min)': 'burns_calories_per_30_min',
    'Target Muscle Group': 'target_muscle_group',
    'Equipment Needed': 'equipment_needed',
    'Difficulty Level': 'difficulty_level',
    'Body Part': 'body_part',
    'Type of Muscle': 'muscle_type',
    'Workout': 'workout',

    'BMI_calc': 'bmi_calc',
    'cal_from_macros': 'cal_from_macros',
    'pct_carbs': 'pct_carbs',
    'protein_per_kg': 'protein_per_kg',
    'pct_HRR': 'pct_hrr',
    'pct_maxHR': 'pct_maxhr',
    'cal_balance': 'cal_balance',
    'lean_mass_kg': 'lean_mass_kg',
    'expected_burn': 'expected_burn',

    'Burns Calories (per 30 min)_bc': 'burns_calories_per_30_min_bc',
    'Burns_Calories_Bin': 'burns_calories_bin'
})
  # select only informative features
  data_selected_features = data[
       ['age',
        'gender',
        'weight',
        'height',
        'max_bpm',
        'avg_bpm',
        'resting_bpm',
        'session_duration',
        'calories_burned',
        'workout_type',
        'fat_percentage',
        'water_intake',
        'workout_frequency',
        'experience_level',
        'bmi',
        'daily_meals_frequency',
        'physical_exercise',
        'carbs',
        'proteins',
        'fats',
        'calories',
        'diet_type',
        'sugar_g',
        ]]

  # drop Physical exercise feature
  data_selected_features = data_selected_features.drop('physical_exercise', axis=1)
  return data_selected_features

def get_important_features(data):
  data_no_fat = data.copy(deep=True)
  y = data_no_fat.pop('fat_percentage')
  X = data_no_fat[data_no_fat.describe().columns]
  # Add intercept
  X = sm.add_constant(X)
  # Fit OLS model
  model = sm.OLS(y, X).fit()

  importance_df = (
      model.summary2().tables[1]
      .rename(columns={"Coef.": "coefficient"})
      [["coefficient", "P>|t|"]]
      .sort_values("P>|t|")
  )
  importance_df['Statistically Significant?'] = importance_df.apply(func=lambda x: x['P>|t|'] < 0.05, axis=1)
  importance_df.drop('const', axis=0, inplace=True)
  important_features = list(importance_df[importance_df['Statistically Significant?'] == True].index)

  important_features = list(set(
     important_features + 
     ['age', 'weight', 'height', 'session_duration', 
      'gender', 'workout_type', 'calories_burned', 
      'workout_frequency', 'fat_percentage'
      ]
     ))
  data_important_features = data[important_features]
  return data_important_features


def get_encoded_cat_features(data: pd.DataFrame) -> pd.DataFrame:
  data_copy = data.copy(deep=True)
#   important_features = get_important_features(data=data_copy)
  data_encoded = pd.get_dummies(data_copy, dtype='float64')
  data_encoded = data_encoded.rename(columns={f: f.lower() for f in data_encoded.columns})
  return data_encoded


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates engineered features for predicting fat_percentage.
    Uses ONLY columns that are confirmed to exist.
    """
    df = df.copy()
    eps = 1e-9  # avoid division by zero

    # =====================================================
    # Cardio / physiology
    # =====================================================
    df["hrr"] = (df["max_bpm"] - df["resting_bpm"]).clip(lower=0)

    df["intensity_ratio"] = df["avg_bpm"] / (df["max_bpm"] + eps)

    df["hrr_per_age"] = df["hrr"] / (df["age"] + eps)

    # =====================================================
    # Training volume & efficiency
    # =====================================================
    df["weekly_training_minutes"] = (
        df["workout_frequency"] * df["session_duration"]
    ).clip(lower=0)

    df["weekly_training_hours"] = df["weekly_training_minutes"] / 60

    df["calories_per_training_hour"] = (
        df["calories_burned"] / (df["weekly_training_hours"] + eps)
    )

    df["calories_per_minute"] = (
        df["calories_burned"] / (df["session_duration"] + eps)
    )

    df["calories_per_kg"] = (
        df["calories_burned"] / (df["weight"] + eps)
    )

    # =====================================================
    # Body composition interactions
    # =====================================================
    df["bmi_x_age"] = df["bmi"] * df["age"]
    df["bmi_x_training_hours"] = df["bmi"] * df["weekly_training_hours"]

    # =====================================================
    # Nutrition ratios
    # =====================================================
    df["protein_ratio"] = df["proteins"] / (df["calories"] + eps)
    df["carb_ratio"] = df["carbs"] / (df["calories"] + eps)
    df["fat_ratio"] = df["fats"] / (df["calories"] + eps)

    df["sugar_per_calorie"] = df["sugar_g"] / (df["calories"] + eps)

    df["calories_per_meal"] = (
        df["calories"] / (df["daily_meals_frequency"] + eps)
    )

    # =====================================================
    # Workout-type composites
    # =====================================================
    df["high_intensity_workout"] = (
        (df["workout_type_hiit"] == 1) |
        (df["workout_type_cardio"] == 1)
    ).astype(float)

    df["strength_vs_cardio"] = (
        df["workout_type_strength"] - df["workout_type_cardio"]
    )

    df["intensity_weighted_training_hours"] = df["weekly_training_hours"] * (
        1.6 * df["workout_type_hiit"] +
        1.2 * df["workout_type_cardio"] +
        1.2 * df["workout_type_strength"] +
        0.8 * df["workout_type_yoga"]
    )

    # =====================================================
    # Diet-type composites
    # =====================================================
    df["low_carb_diet"] = (
        df["diet_type_keto"] +
        df["diet_type_low-carb"] +
        df["diet_type_paleo"]
    ).clip(upper=1)

    df["plant_based_diet"] = (
        df["diet_type_vegan"] +
        df["diet_type_vegetarian"]
    ).clip(upper=1)

    # =====================================================
    # Gender interactions
    # =====================================================
    df["female_x_bmi"] = df["gender_female"] * df["bmi"]
    df["male_x_bmi"] = df["gender_male"] * df["bmi"]

    # =====================================================
    # Log transforms (stable, informative)
    # =====================================================
    for col in [
        "calories_burned",
        "weekly_training_hours",
        "calories_per_training_hour",
        "calories_per_kg",
        "calories"
    ]:
        df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))

    return df



def save_scaler(scaler, filename: str | Path):
  with open(filename, 'wb') as f:
    pickle.dump(scaler, f, pickle.HIGHEST_PROTOCOL)


def get_classification_data(data: pd.DataFrame):
  classification_data = data.copy(deep=True)
  quantile_33, quantile_66 = classification_data.fat_percentage.quantile([0.33, 0.66])

  def get_class(X):
    fat_percentage = X['fat_percentage']
    if fat_percentage < quantile_33:
      return 0.0
    elif fat_percentage >= quantile_33 and fat_percentage <= quantile_66:
      return 1.0
    return 2.0
  classification_data['class'] = classification_data.apply(func=get_class, axis=1) # type: ignore
  return classification_data

def get_train_test_classification(data: pd.DataFrame, scaler):
  X = data.copy()
  X.pop('fat_percentage') # remove fat_percentage
  y = X.pop('class') # class is new target feature
  
  features_names = X.columns # storing feature_names for feature engineering

  X_fe = add_engineered_features(X)

  # splitting data into train and test/val, 80% - train; 20% - test/validation
  X_train, X_test_val, y_train, y_test_val = train_test_split(X_fe, y, test_size=0.2, shuffle=True, random_state=SEED)
  # splitting data into test and val, 10% - test; 10% - validation
  X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, shuffle=True, random_state=SEED)

  # converting everything to numpy for transforming
  X_train, X_test, X_val = X_train.to_numpy(), X_test.to_numpy(), X_val.to_numpy()
  y_train, y_test, y_val = y_train.to_numpy(), y_test.to_numpy(), y_val.to_numpy()

  # reshaping our 1D vectors into 2D
  y_train, y_test, y_val = y_train.reshape(-1, 1), y_test.reshape(-1, 1), y_val.reshape(-1, 1)

  
  # training scalers on training data and transforming test/val data using it
  X_train_trans = scaler.fit_transform(X_train) # just once
  X_test_trans = scaler.transform(X_test)
  X_val_trans = scaler.transform(X_val)

  splits = {
     "X_train": X_train, "X_test": X_test, "X_val": X_val,
     "y_train": y_train, "y_test": y_test, "y_val": y_val,
     "X_train_trans": X_train_trans, "X_test_trans": X_test_trans, 
     "X_val_trans": X_val_trans,
  }
  return splits

def train_get_classifier(splits: dict):
  input_shape = (splits['X_train_trans'].shape[1],)

  class_weights = compute_class_weight('balanced',
                                      classes=np.unique(splits['y_train']),
                                      y=splits['y_train'].flatten())
  class_weights_dict = dict(enumerate(class_weights))
  
  c_model = keras.Sequential(name="fat_percentage_classifier_v1")

  # Input layer (explicitly named)
  c_model.add(keras.layers.Input(shape=input_shape, name="input_features"))

  # Hidden Layer 1: Feature extraction
  c_model.add(keras.layers.Dense(
      128,
      activation="elu",
      kernel_regularizer=keras.regularizers.l2(1e-4),
      name="hidden_layer_0_elu"
  ))

  # Hidden Layer 2: Feature processing
  c_model.add(keras.layers.Dense(
      64,
      activation="gelu",
      kernel_regularizer=keras.regularizers.l2(1e-5),
      name="hidden_layer_1_gelu"
  ))

  # Hidden Layer 3: Feature compression
  c_model.add(keras.layers.Dense(
      32,
      activation="gelu",
      kernel_regularizer=keras.regularizers.l2(1e-5),
      name="hidden_layer_3_gelu"
  ))

  # Output layer
  c_model.add(keras.layers.Dense(
      3,
      activation="softmax",
      name="class_probabilities"
  ))

  reduce_lr = keras.callbacks.ReduceLROnPlateau(
      monitor="val_loss",
      factor=0.2,
      patience=4,
      min_lr=1e-6,
      verbose=1
  )

  early_stopping = keras.callbacks.EarlyStopping(
      patience=15,
      restore_best_weights=True
  )

  optimizer = keras.optimizers.RMSprop(1e-3)

  c_model.compile(
      loss=keras.losses.SparseCategoricalCrossentropy(),
      optimizer=optimizer, # type: ignore
      metrics=['accuracy']
  )

  c_model.fit(
      splits['X_train_trans'], splits['y_train'],
      epochs=40,
      validation_data=(splits['X_val_trans'], splits['y_val']),
      callbacks=[reduce_lr, early_stopping],
      class_weight=class_weights_dict,
      verbose=0 # type: ignore
  )
  return c_model

  # with open(f'{c_model.name}.pkl', 'wb') as f:
  #   pickle.dump(c_model, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open('fat_percentage_classifier_v1.pkl', 'rb') as f:
#   c_model = pickle.load(f)

def get_data_for_regressors(data: pd.DataFrame):
  low_fat_data = data[data['class'] == 0]
  mid_fat_data = data[data['class'] == 1]
  high_fat_data = data[data['class'] == 2]

  return low_fat_data, mid_fat_data, high_fat_data

def get_regressor_splits(data: pd.DataFrame, scaler: MinMaxScaler):
  X = data.copy(deep=True)
  X.pop('class') # remove fat_percentage
  y = X.pop('fat_percentage') # class is new target feature
  features_names = X.columns # storing feature_names for feature engineering

  X_fe = add_engineered_features(X)

  # splitting data into train and test/val, 80% - train; 20% - test/validation
  X_train, X_test_val, y_train, y_test_val = train_test_split(X_fe, y, test_size=0.1, shuffle=True, random_state=SEED)
  # splitting data into test and val, 10% - test; 10% - validation
  X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, shuffle=True, random_state=SEED)

  # converting everything to numpy for transforming
  X_train, X_test, X_val = X_train.to_numpy(), X_test.to_numpy(), X_val.to_numpy()
  y_train, y_test, y_val = y_train.to_numpy(), y_test.to_numpy(), y_val.to_numpy()

  # reshaping our 1D vectors into 2D
  y_train, y_test, y_val = y_train.reshape(-1, 1), y_test.reshape(-1, 1), y_val.reshape(-1, 1)

  # training scalers on training data and transforming test/val data using it
  X_train_trans = scaler.fit_transform(X_train)
  X_test_trans = scaler.transform(X_test)
  X_val_trans = scaler.transform(X_val)

  splits = {
     "X_train": X_train, "X_test": X_test, "X_val": X_val,
     "y_train": y_train, "y_test": y_test, "y_val": y_val,
     "X_train_trans": X_train_trans, "X_test_trans": X_test_trans, 
     "X_val_trans": X_val_trans,
  }
  return splits


def save_model(model, path):
    """Safe way to save Keras model"""
    model.save(path)

    # config = model.get_config()
    # weights = model.get_weights()
    
    # with open(filename, 'wb') as f:
    #     pickle.dump({'config': config, 'weights': weights, 'custom_objects': custom_objects}, f)

def load_model_pickle(filename):
    """Safe way to load Keras model with pickle"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict) and 'config' in data:
    # Recreate as Sequential model
      model = keras.Sequential.from_config(
          data['config'],
          custom_objects=data['custom_objects']
      )
      if 'weights' in data:
          model.set_weights(data['weights'])
      return model
    raise ValueError(f"file: {filename} is not supported")

def save_base_model_pkl(model, filename):
  with open(filename, 'wb') as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL) 

def load_base_model_pkl(filename):
  with open(filename, 'rb') as f:
    base_model = pickle.load(f)
  return base_model

def get_base_model(splits: dict):
  base_model = Lasso(max_iter=2000)
  base_model.fit(splits['X_train_trans'], splits['y_train'])
  return base_model

def get_base_predictions(base_model, splits):
  y_train_pred = base_model.predict(splits['X_train_trans'])
  y_val_pred   = base_model.predict(splits['X_val_trans'])
  y_test_pred  = base_model.predict(splits['X_test_trans'])

  y_train_resid = splits['y_train'].flatten() - y_train_pred
  y_val_resid   = splits['y_val'].flatten() - y_val_pred
  y_test_resid  = splits['y_test'].flatten() - y_test_pred

  preds = {
     "y_train_resid": y_train_resid,
     "y_val_resid": y_val_resid,
     "y_test_resid": y_test_resid
  }
  return preds


@keras.utils.register_keras_serializable(package='MyPachage', name='quantile_loss_lower_v1')
def quantile_loss_lower(y_true, y_pred, q=0.2):
    """Custom quantile loss function for lower penalty"""
    e = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q*e, (q-1)*e))

@keras.utils.register_keras_serializable(package='MyPachage', name='quantile_loss_upper_v1')
def quantile_loss_upper(y_true, y_pred, q=0.75):
    """Custom quantile loss function for upper penalty"""
    e = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q*e, (q-1)*e))


def train_get_regressor(splits, preds, name, loss_func):
  # Option 1: Standard Sequential with named layers
  input_shape = (splits['X_train_trans'].shape[1],)
  
  model = keras.Sequential(name=name)

  # Input layer (explicitly named)
  model.add(keras.layers.Input(shape=input_shape, name="input_features"))

  # Hidden Layer 1: Feature extraction
  model.add(keras.layers.Dense(
      64,
      activation="elu",
      kernel_regularizer=keras.regularizers.l2(1e-4),
      name="hidden_layer_0_elu"
  ))

  # Hidden Layer 2: Feature processing
  model.add(keras.layers.Dense(
      32,
      activation="elu",
      kernel_regularizer=keras.regularizers.l2(1e-4),
      name="hidden_layer_1_elu"
  ))

  # Output layer
  model.add(keras.layers.Dense(
      1,
      activation="linear",
      name="residuals_prediction"
  ))

  optimizer = keras.optimizers.Adam(1e-3)

  model.compile(
      loss=loss_func,
      optimizer=optimizer, # type: ignore
      metrics=["mae"],
  )

  early_stopping = keras.callbacks.EarlyStopping(
              patience=30,
              restore_best_weights=True)

  reduce_lr = keras.callbacks.ReduceLROnPlateau(
              patience=6,
              factor=0.2,
              min_lr=1e-6)

  model.fit(
      splits['X_train_trans'],
      preds['y_train_resid'],
      validation_data=(splits['X_val_trans'], preds['y_val_resid']),
      epochs=80,
      callbacks=[early_stopping, reduce_lr],
      verbose=0 # type: ignore
  )
  
  return model


if __name__ == '__main__':
  SEED = 42

  os.environ["PYTHONHASHSEED"] = str(SEED)
  random.seed(SEED)
  np.random.seed(SEED)
  tf.random.set_seed(SEED)

  classification_scaler = MinMaxScaler()
  regressor_scalers = {
   'low_fat_scaler' : MinMaxScaler(),
   'mid_fat_scaler' : MinMaxScaler(),
   'high_fat_scaler' : MinMaxScaler()
  }
  
  models_path = Path(__file__).parent.parent / "models"   

  # Preparing data
  filepath = Path(__file__).parent / 'Final_data.csv'
  data = read_data(filename=filepath)
  encoded_data = get_encoded_cat_features(data=data)

  # Classification
  classification_data = get_classification_data(data=encoded_data)
  splits = get_train_test_classification(data=classification_data, scaler=classification_scaler)
  classifier = train_get_classifier(splits=splits)
  classifier_path = models_path / "classifiers" / f"{classifier.name}.keras"
  class_scaler_path = models_path / "scalers" / "classification_scaler.pkl"
  save_model(model=classifier, path=classifier_path)
  save_scaler(scaler=classification_scaler, filename=class_scaler_path)
  
  # Data for regressors
  low_fat_data, mid_fat_data, high_fat_data = get_data_for_regressors(data=classification_data)  
  
  # Low-fat regressor
  low_fat_splits = get_regressor_splits(data=low_fat_data, scaler=regressor_scalers['low_fat_scaler'])
  low_base_model = get_base_model(splits=low_fat_splits)
  low_fat_preds = get_base_predictions(base_model=low_base_model, splits=low_fat_splits)
  
  l_base_model_path = models_path / "base_models" / "low_fat_base_model.pkl"
  save_base_model_pkl(model=low_base_model, filename=l_base_model_path)
  
  low_fat_regressor = train_get_regressor(
                        splits=low_fat_splits, 
                        preds=low_fat_preds,
                        name="low_fat_residuals_regressor_v1",
                        loss_func=quantile_loss_lower)
  
  l_regressor_path = models_path / "regressors" / f"{low_fat_regressor.name}.keras"
  l_scaler_path = models_path / "scalers" / "low_fat_scaler.pkl"
  save_model(model=low_fat_regressor, path=l_regressor_path)
  save_scaler(scaler=regressor_scalers['low_fat_scaler'], filename=l_scaler_path)

  # Mid-fat regressor
  mid_fat_splits = get_regressor_splits(data=mid_fat_data, scaler=regressor_scalers['mid_fat_scaler'])
  mid_base_model = get_base_model(splits=mid_fat_splits)
  mid_fat_preds = get_base_predictions(base_model=mid_base_model, splits=mid_fat_splits)
  
  m_base_model_path = models_path / "base_models" / "mid_fat_base_model.pkl"
  save_base_model_pkl(model=mid_base_model, filename=m_base_model_path)

  mid_fat_regressor = train_get_regressor(
                        splits=mid_fat_splits, 
                        preds=mid_fat_preds,
                        name="mid_fat_residuals_regressor_v1",
                        loss_func=keras.losses.Huber(delta=1.0))
  m_regressor_path = models_path / "regressors" / f"{mid_fat_regressor.name}.keras"
  m_scaler_path = models_path / "scalers" / "mid_fat_scaler.pkl"
  save_model(model=mid_fat_regressor, path=m_regressor_path)
  save_scaler(scaler=regressor_scalers['mid_fat_scaler'], filename=m_scaler_path)

  # High-fat regressor
  high_fat_splits = get_regressor_splits(data=high_fat_data, scaler=regressor_scalers['high_fat_scaler'])
  high_base_model = get_base_model(splits=high_fat_splits)
  high_fat_preds = get_base_predictions(base_model=high_base_model, splits=high_fat_splits)

  h_base_model_path = models_path / "base_models" / "high_fat_base_model.pkl"
  save_base_model_pkl(model=high_base_model, filename=h_base_model_path)

  high_fat_regressor = train_get_regressor(
                        splits=high_fat_splits, 
                        preds=high_fat_preds,
                        name="high_fat_residuals_regressor_v1",
                        loss_func=quantile_loss_upper)

  h_regressor_path = models_path / "regressors" / f"{high_fat_regressor.name}.keras"
  h_scaler_path = models_path / "scalers" / "high_fat_scaler.pkl"
  save_model(model=high_fat_regressor, path=h_regressor_path)
  save_scaler(scaler=regressor_scalers['high_fat_scaler'], filename=h_scaler_path)