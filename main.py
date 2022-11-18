from typing import Tuple, Sequence
import numpy as np
import pandas as pd
import os
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from models.model_creator import ModelCreator
from models.residual import ResidualClassifier, SimpleResidual
from models.base import BaseModel
from datetime import datetime

from sklearn.model_selection import KFold


def load_data() -> Tuple[Sequence[float], Sequence[float]]:
    label_data = loadmat("data/label.mat")
    # you will have to change this path based on your configuration
    label = label_data["Y"]
    label = np.squeeze(label, axis=1).astype("float32")

    x_data = loadmat("data/2D_img.mat")["imgData"]
    x_data = x_data.transpose((2, 0, 1))
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = (x_data - x_data.min()) / (x_data.max() - x_data.min())
    return x_data, label


def train_model(
    model: Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs=400,
    batch_size=128,
    validation_split=0.2,
    verbose=True,
    weighted=False,
    log_dir=None,
):
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10)]
    if log_dir is not None:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,
                embeddings_freq=0,
                update_freq="epoch",
            )
        )

    if weighted:
        ones_weight = np.sum(y_train == 0) / len(y_train)
        zeros_weight = 1.0 - ones_weight
        class_weight = {0: zeros_weight, 1: ones_weight}
    else:
        class_weight = {0: 1.0, 1: 1.0}

    hist = model.fit(
        x_train,
        y_train,
        validation_split=validation_split,
        callbacks=callbacks,
        class_weight=class_weight,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )
    return model, hist


def evaluate_model(model_builder: ModelCreator, seed=42, verbose=False):
    """
    Performs Kfold cross validation on a model.
    Outputs metrics to the logs folder
    """
    X, y = load_data()
    accuracies = []
    now = datetime.now().strftime(r"%Y-%m-%d_%H:%M:%S")
    model = model_builder()
    run_folder = os.path.join(f"logs/{model.name}__{now}")
    os.makedirs(run_folder, exist_ok=True)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    for i_train, i_val in kfold.split(X, y):
        model = model_builder()
        x_train, y_train = X[i_train], y[i_train]
        x_val, y_val = X[i_val], y[i_val]
        model, _ = train_model(
            model, x_train, y_train, verbose=verbose, log_dir=run_folder
        )
        _, accuracy = model.evaluate(x_val, y_val)
        accuracies.append(accuracy)
    print("Total CV accuracy", np.mean(accuracies) * 100.0)


if __name__ == "__main__":
    evaluate_model(ModelCreator(SimpleResidual), verbose=True)
