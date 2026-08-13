"""Runtime support for the final ordinal body-fat classifier."""

import numpy as np
import keras


@keras.utils.register_keras_serializable(
    package="BodyFat",
    name="ordered_cumulative_probabilities_final",
)
class OrderedCumulativeProbabilities(keras.layers.Layer):
    """Produce two ordered P(class > boundary) probabilities."""

    def build(self, input_shape):
        self.first_threshold = self.add_weight(
            name="first_threshold",
            shape=(),
            initializer=keras.initializers.Constant(-0.7),
            trainable=True,
        )
        self.raw_threshold_gap = self.add_weight(
            name="raw_threshold_gap",
            shape=(),
            initializer=keras.initializers.Constant(0.6),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, score):
        second_threshold = (
            self.first_threshold
            + keras.ops.softplus(self.raw_threshold_gap)
        )
        probability_above_low = keras.ops.sigmoid(
            score - self.first_threshold
        )
        probability_above_mid = keras.ops.sigmoid(
            score - second_threshold
        )
        return keras.ops.concatenate(
            [probability_above_low, probability_above_mid],
            axis=-1,
        )


def decode_ordinal_predictions(cumulative_probabilities) -> np.ndarray:
    """Convert two cumulative probabilities into ordered classes 0, 1, or 2."""

    probabilities = np.asarray(cumulative_probabilities)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "Ordinal classifier must return two cumulative probabilities "
            "per sample."
        )
    return np.sum(probabilities >= 0.5, axis=1).astype(int)
