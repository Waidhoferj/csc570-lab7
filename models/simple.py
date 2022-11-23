import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class SimpleClassifier(Sequential):
    """
    Basic model described by Sumona
    """

    def __init__(
        self,
    ):
        super().__init__(
            [
                layers.Conv2D(8, (3, 3), activation="tanh"),
                layers.MaxPool2D(pool_size=2, strides=1),
                layers.Conv2D(8, (3, 3), activation="tanh"),
                layers.MaxPool2D(pool_size=2, strides=1),
                layers.Conv2D(16, (3, 3), activation="tanh"),
                layers.MaxPool2D(pool_size=2, strides=1),
                layers.Flatten(),
                layers.Dense(1),
            ],
            name="Experimental",
        )
