import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class BaseModel(Sequential):
    """
    Basic model described by Sumona
    """

    def __init__(
        self,
    ):
        super().__init__(
            [
                layers.Conv2D(12, (3, 3), activation="relu"),
                layers.MaxPool2D(pool_size=2, strides=1),
                layers.Conv2D(24, (3, 3), activation="relu"),
                layers.MaxPool2D(pool_size=2, strides=1),
                layers.Flatten(),
                layers.Dense(1),
            ],
            name="BaseModel",
        )
