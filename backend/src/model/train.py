import json
import pickle
from tqdm import tqdm
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.core import FrozenDict
import optax

from src.model.model import CNN
from src.utils.utils import log_message, check_jax_device


def model_init(img_size: int, model: nn.Module, key) -> FrozenDict:
    """
    Args:
        img_size : the size of my data
        model : CNN model
        key : PRNG key

    Returns:
        model's parameters
    """
    dummy_input = jnp.ones((1, img_size, img_size, 3))
    params = model.init(key, dummy_input)
    return params


def optimization_fn(learning_rate: float,
                    model: nn.Module,
                    params: FrozenDict
                    ) -> train_state.TrainState:
    """
    Creates an object that holds the model, its parameters, and the optimizer.

    Args:
        learning_rate: Learning rate for the optimizer.
        model: CNN model.
        params: Model's params.

    Returns:
        Train state object.
    """
    optimizer = optax.adamw(learning_rate=learning_rate)
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    return model_state


def compute_loss_acc(state: train_state.TrainState,
                     params: FrozenDict,
                     batch: Tuple[jnp.ndarray, jnp.ndarray]
                     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the softmax cross-entropy loss and accuracy for a given batch.

    Args:
        state: The current model state containing parameters
                and apply function.
        params: The model parameters (PyTree of weights).
        batch: A tuple containing (input data, labels).

    Returns:
        Tuple of loss and accuracy.
    """
    input_data, labels = batch
    logits = state.apply_fn(params, input_data)  # Forward pass
    num_classes = logits.shape[-1]
    labels = jax.nn.one_hot(labels, num_classes)

    loss = optax.softmax_cross_entropy(logits, labels).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    return loss, acc


@jax.jit
def train_step(state: train_state.TrainState,
               batch: Tuple[jnp.ndarray,
                            jnp.ndarray]
               ) -> Tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray]:
    grad_fn = jax.value_and_grad(compute_loss_acc, argnums=1, has_aux=True)
    (loss, acc), grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc


@jax.jit
def eval_step(state: train_state.TrainState,
              batch: Tuple[jnp.ndarray, jnp.ndarray]):
    loss, acc = compute_loss_acc(state, state.params, batch)
    return loss, acc


def train_model(state: train_state.TrainState,
                train,
                valid,
                num_epochs=2):
    train_acc, train_loss, valid_acc, valid_loss = [], [], [], []

    for epoch in tqdm(range(num_epochs)):
        train_batch_loss, train_batch_acc = [], []
        valid_batch_loss, valid_batch_acc = [], []

        for t_batch in train:
            state, loss, acc = train_step(state, t_batch)
            train_batch_loss.append(loss)
            train_batch_acc.append(acc)

        for v_batch in valid:
            val_loss, val_acc = eval_step(state, v_batch)
            valid_batch_loss.append(val_loss)
            valid_batch_acc.append(val_acc)

        train_loss.append(np.mean(jax.device_get(train_batch_loss)))
        train_acc.append(np.mean(jax.device_get(train_batch_acc)))
        valid_loss.append(np.mean(jax.device_get(valid_batch_loss)))
        valid_acc.append(np.mean(jax.device_get(valid_batch_acc)))

        log_message(epoch=epoch,
                    epoch_train_loss=train_loss[-1],
                    epoch_train_acc=train_acc[-1],
                    epoch_valid_loss=valid_loss[-1],
                    epoch_valid_acc=valid_acc[-1],
                    level="TRAIN")

    return state, train_acc, train_loss, valid_acc, valid_loss


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key, 2)

    train_data = pickle.load(open("data/processed/train_data.pkl", "rb"))
    valid_data = pickle.load(open("data/processed/valid_data.pkl", "rb"))

    train_data = [(jnp.array(img),
                   jnp.array(label)) for img, label in train_data]
    valid_data = [(jnp.array(img),
                   jnp.array(label)) for img, label in valid_data]

    with open("src/params.json") as f:
        PARAMS = json.load(f)["CNN"]

    model = CNN(num_classes=PARAMS["NUM_CLASS"])
    log_message("Initializing the model")
    params = model_init(PARAMS["IMG_SIZE"], model, subkey)

    log_message("Creating the optimization function")
    model_state = optimization_fn(PARAMS["LEARNING_RATE"], model, params)

    log_message("Start training")
    check_jax_device()
    _, train_acc, train_loss, \
        valid_acc, valid_loss = train_model(model_state,
                                            train_data,
                                            valid_data)
    log_message("Training finished", "DONE")
