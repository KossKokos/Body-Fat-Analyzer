import pickle
import os

from sklearn.preprocessing import MinMaxScaler
from keras import Sequential

from pathlib import Path

path_to_model = 'trained_model'
path_to_x_scaler = os.path.join(Path(__file__).parent.parent, 'x_scaler')
path_to_y_scaler = os.path.join(Path(__file__).parent.parent, 'y_scaler')


def load_model(path_to_model):
    with open(path_to_model, 'rb') as f:
        model = pickle.load(f)
    return model

def load_scaler(path_to_scaler):
    with open(path_to_scaler, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

model: Sequential = load_model(path_to_model=path_to_model)
x_scaler: MinMaxScaler = load_scaler(path_to_scaler=path_to_x_scaler)
y_scaler: MinMaxScaler = load_scaler(path_to_scaler=path_to_y_scaler)