import tensorflow as tf
from tensorflow.keras import Model


class ModelCreator:
    """
    Generates models with specific initialization and compilation configurations
    """

    def __init__(
        self,
        model_class: Model,
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
        *args,
        **kwargs
    ):
        self.model_class = model_class
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.model_args = args
        self.model_kwargs = kwargs

    def __call__(self) -> Model:
        model = self.model_class(*self.model_args, **self.model_kwargs)
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics,
        )
        return model
