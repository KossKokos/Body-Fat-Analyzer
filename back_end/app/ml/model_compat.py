"""Keras compatibility loader for the final inference artifacts.

The models were saved by Keras 3.15. Keras 3.12 cannot deserialize their
architecture configs because newer Dense configs include
``quantization_config``. The weights themselves are compatible, so this module
reconstructs the exact inference architectures and loads weights directly from
the ``.keras`` archives.

Keep these builders synchronized with ``loading_script.ipynb``. They are only
for the four final neural-model artifacts; scalers and Lasso base models
continue to load normally through pickle.
"""

from pathlib import Path
from typing import Callable

import keras

from ml.ordinal_classifier import OrderedCumulativeProbabilities


MODEL_INPUT_SIZE = 57

ORDINAL_CLASSIFIER_FILENAME = (
    "fat_percentage_ordinal_classifier_final.keras"
)
RESIDUAL_REGRESSOR_FILENAMES = frozenset(
    {
        "low_fat_residuals_regressor_final.keras",
        "mid_fat_boundary_weighted_residuals_regressor_final.keras",
        "high_fat_residuals_regressor_final.keras",
    }
)


def _build_ordinal_classifier() -> keras.Model:
    """Recreate the exact two-boundary ordinal classifier architecture."""

    inputs = keras.layers.Input(
        shape=(MODEL_INPUT_SIZE,),
        name="input_features",
    )
    x = keras.layers.Dense(
        128,
        activation="elu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="hidden_layer_0_elu",
    )(inputs)
    x = keras.layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=keras.regularizers.l2(1e-5),
        name="hidden_layer_1_gelu",
    )(x)
    x = keras.layers.Dense(
        32,
        activation="gelu",
        kernel_regularizer=keras.regularizers.l2(1e-5),
        name="hidden_layer_2_gelu",
    )(x)
    ordinal_score = keras.layers.Dense(
        1,
        use_bias=False,
        name="ordinal_score",
    )(x)
    cumulative_probabilities = OrderedCumulativeProbabilities(
        name="ordered_class_probabilities",
    )(ordinal_score)

    return keras.Model(
        inputs=inputs,
        outputs=cumulative_probabilities,
        name="fat_percentage_ordinal_classifier_final",
    )


def _build_residual_regressor(model_name: str) -> keras.Model:
    """Recreate the shared final residual-regressor architecture."""

    model = keras.Sequential(name=model_name)
    model.add(
        keras.layers.Input(
            shape=(MODEL_INPUT_SIZE,),
            name="input_features",
        )
    )
    model.add(
        keras.layers.Dense(
            64,
            activation="elu",
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name="hidden_layer_0_elu",
        )
    )
    model.add(
        keras.layers.Dense(
            32,
            activation="elu",
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name="hidden_layer_1_elu",
        )
    )
    model.add(
        keras.layers.Dense(
            1,
            activation="linear",
            name="residuals_prediction",
        )
    )
    return model


def _classifier_builder(_: str) -> keras.Model:
    return _build_ordinal_classifier()


def _residual_builder(filename: str) -> keras.Model:
    return _build_residual_regressor(Path(filename).stem)


_ARTIFACT_BUILDERS: dict[str, Callable[[str], keras.Model]] = {
    ORDINAL_CLASSIFIER_FILENAME: _classifier_builder,
    **{
        filename: _residual_builder
        for filename in RESIDUAL_REGRESSOR_FILENAMES
    },
}


def load_final_keras_model(filepath: Path | str) -> keras.Model:
    """Build and weight-load one known final Keras artifact.

    The artifact filename selects the required architecture. An unknown
    filename is rejected explicitly to prevent accidentally loading weights
    into a structurally similar but semantically different model.
    """

    path = Path(filepath)
    try:
        builder = _ARTIFACT_BUILDERS[path.name]
    except KeyError as exc:
        supported = ", ".join(sorted(_ARTIFACT_BUILDERS))
        raise ValueError(
            f"Unsupported final Keras artifact: {path.name}. "
            f"Expected one of: {supported}"
        ) from exc

    if not path.is_file():
        raise FileNotFoundError(f"Final Keras artifact not found: {path}")

    model = builder(path.name)
    model.load_weights(path)
    return model
