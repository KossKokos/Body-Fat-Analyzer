import pandas as pd
import os
from pathlib import Path

path_to_history = os.path.join(Path(__file__).parent.parent, 'history.csv')

columns = [
 'calories_burned',
 'max_bpm',
 'age',
 'weight',
 'daily_meals_frequency',
 'resting_bpm',
 'bmi',
 'workout_frequency',
 'water_intake',
 'session_duration',
 'height',
 'gender',
 'workout_type',
 'fat_percentage'
 ]

def load_data():
    try:
        history = pd.read_csv(path_to_history) 
    except FileNotFoundError:
        history = pd.DataFrame(columns=columns)
        history.to_csv(path_to_history, index=False)
    return history     

def write_data(history: pd.DataFrame):
    try:
        history.to_csv(path_to_history, index=False)
    except:
        raise SystemError('Could not save new history object')

history: pd.DataFrame = load_data()

def add_new_record(new_row: pd.DataFrame):
    
    old_history: pd.DataFrame = load_data()
    print('old history is loaded')
    new_history = pd.concat([old_history, new_row], axis=0)
    print('new history is created')
    write_data(history=new_history)
    print('new history record was added to history.csv file')