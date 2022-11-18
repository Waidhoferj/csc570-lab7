from tensorflow.keras import layers, Sequential, activations
import tensorflow as tf


class ResidualClassifier(Sequential):
    """
    CNN model with residual layers
    """

    def __init__(
        self,
    ):
        super().__init__(
            [
                ResidualBlock(16),
                layers.MaxPool2D(),
                ResidualBlock(32),
                layers.MaxPool2D(),
                ResidualBlock(32),
                layers.MaxPool2D(),
                ResidualBlock(64),
                layers.MaxPool2D(),
                ResidualBlock(64),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(1),
            ],
            name="ResidualClassifier",
        )


class SimpleResidual(Sequential):
    def __init__(
        self,
    ):
        super().__init__(
            [
                ResidualBlock(12),
                layers.MaxPool2D(),
                layers.Flatten(),
                layers.Dense(32, activation="relu"),
                layers.Dense(1),
            ],
        )


class ResidualBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.long_path = Sequential(
            [
                layers.Conv2D(
                    filters,
                    kernel_size=(3, 3),
                    padding="same",
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(
                    filters * 2,
                    kernel_size=(3, 3),
                    padding="same",
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(
                    filters,
                    kernel_size=(3, 3),
                    padding="same",
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )
        self.short_path = Sequential(
            [
                layers.Conv2D(
                    filters,
                    kernel_size=(3, 3),
                    padding="same",
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "short_path": self.short_path,
                "long_path": self.long_path,
            }
        )
        return config

    def call(self, inputs):
        x = self.long_path(inputs)
        y = self.short_path(inputs)
        return activations.relu(x + y)
