from tensorflow.keras import layers, Sequential
import tensorflow as tf


class VGGEsque(Sequential):
    def __init__(self):
        super().__init__(
            [
                layers.Conv2D(
                    16, (3, 3), kernel_regularizer=tf.keras.regularizers.L2(0.1)
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPool2D(),
                layers.Conv2D(
                    32, (3, 3), kernel_regularizer=tf.keras.regularizers.L2(0.1)
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPool2D(),
                layers.Conv2D(
                    64, (3, 3), kernel_regularizer=tf.keras.regularizers.L2(0.1)
                ),
                layers.Conv2D(
                    64, (3, 3), kernel_regularizer=tf.keras.regularizers.L2(0.1)
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPool2D(padding="same"),
                layers.Conv2D(
                    128, (3, 3), kernel_regularizer=tf.keras.regularizers.L2(0.1)
                ),
                layers.Conv2D(
                    128, (3, 3), kernel_regularizer=tf.keras.regularizers.L2(0.1)
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPool2D(padding="same"),
                layers.Flatten(),
                # layers.Dense(5096, activation="relu"),
                # layers.Dense(128, activation="relu"),
                # layers.Dense(64, activation="relu"),
                layers.Dense(1),
            ]
        )
