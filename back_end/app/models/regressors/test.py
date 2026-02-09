import tensorflow as tf
import keras
import pickle
from pathlib import Path


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

__map_objs = {
            'low': {'quantile_loss' : quantile_loss_lower},
            'high': {'quantile_loss': quantile_loss_upper}
    }

def _get_custom_object(type_):
    return __map_objs[type_]

def load_model_pickle(filename: Path | str, type_ = None) -> keras.Sequential:
    """Safe way to load Keras model with pickle"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict) and 'config' in data:
    # Recreate as Sequential model
        model = keras.Sequential.from_config(
            data['config'],
            custom_objects=type_ if type_ is None else _get_custom_object(type_) 
        )
        if 'weights' in data:
            model.set_weights(data['weights'])
        return model
    raise ValueError(f"file: {filename} is not supported")

model_path = Path(__file__).parent / 'low_fat_residuals_regressor_v1'
model = load_model_pickle(filename=model_path, type_=None)
model.summary()

print('low' in ['low', 'high'])