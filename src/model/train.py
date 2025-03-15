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

def model_init(img_size:int,
               model:nn.Module,
               key
               ) -> FrozenDict:
    """
    Args:
        img_size : the size of my data
        model : CNN model
        key : PRNG key

    Returns:
        model's parameters
    """
    # Init the params
    dummy_input = jnp.ones((1, img_size, img_size, 3)) # The same shape as my data
    params = model.init(key, dummy_input)

    return params

def optimization_fn(learning_rate:float,
                    model:nn.Module,
                    params:FrozenDict
                    ) -> train_state.TrainState:
    """
    It creates an object that holds the model, its parameters, and the optimizer,
    so you can efficiently update and apply the model during training.
    Args:
        learning_rate
        model : CNN model
        params : Model's params

    Returns:
        train state object
    """
    optimizer = optax.adamw(learning_rate=learning_rate)
    model_state = train_state.TrainState.create(
        apply_fn = model.apply,
        params = params,
        tx = optimizer
    )

    return model_state

def compute_loss_acc(state: train_state.TrainState,
                     params: FrozenDict,
                     batch: Tuple[jnp.ndarray, jnp.ndarray]
                     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the softmax cross-entropy loss and accuracy for a given batch.

    Args:
        state (train_state.TrainState): The current model state containing parameters and apply function.
        params (FrozenDict): The model parameters (PyTree of weights).
        batch (Tuple[jnp.ndarray, jnp.ndarray]): A tuple containing (input data, labels).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The computed loss and accuracy.
    """
    # Unpack the batch (images, labels)
    input, labels = batch

    # Raw score from the model before applying softmax
    logits = state.apply_fn(params, input) # Forward pass (not probabilities)

    # Convert integer labels to one-hot inside the traced function
    num_classes = logits.shape[-1]
    labels = jax.nn.one_hot(labels, num_classes)  # Apply within JIT-traced context

    # Computes the softmax cross-entropy loss for classification, and average it over the batch
    loss = optax.softmax_cross_entropy(logits, labels).mean()
    # Compare the predicted class indices with the actual labels.
    acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))

    return loss, acc

@jax.jit
def train_step(state:train_state.TrainState,
               batch:Tuple[jnp.ndarray, jnp.ndarray]
               ) -> Tuple[train_state.TrainState, jnp.ndarray, jnp.ndarray]:

    grad_fn = jax.value_and_grad(
        compute_loss_acc, # Function to calculate the loss
        argnums = 1, # Parameters are second argument of the function
        has_aux = True # Function has additional outputs, here accuracy
    )
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(state, state.params, batch)

    # Update the params
    state = state.apply_gradients(grads = grads)

    return state, loss, acc

@jax.jit
def eval_step(state:train_state.TrainState,
              batch:Tuple[jnp.ndarray, jnp.ndarray]
              ):
    # Determine the accuracy of the model
    loss, acc = compute_loss_acc(state, state.params, batch)

    return loss, acc

def train_model(state:train_state.TrainState,
                train:Tuple[jnp.ndarray, jnp.ndarray],
                valid:Tuple[jnp.ndarray, jnp.ndarray],
                num_epochs=2
                ):
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

        epoch_train_loss = np.mean(jax.device_get(train_batch_loss))
        epoch_train_acc = np.mean(jax.device_get(train_batch_acc))
        epoch_valid_loss = np.mean(jax.device_get(valid_batch_loss))
        epoch_valid_acc = np.mean(jax.device_get(valid_batch_acc))

        train_acc.append(epoch_train_acc)
        valid_loss.append(epoch_valid_loss)
        train_loss.append(epoch_train_loss)
        valid_acc.append(epoch_valid_acc)

    log_message(epoch=epoch, epoch_train_loss=epoch_train_loss, epoch_train_acc=epoch_train_acc,
                    epoch_valid_loss=epoch_valid_loss, epoch_valid_acc=epoch_valid_acc, level="TRAIN")

    return state, train_acc, train_loss, valid_acc, valid_loss

if __name__ == "__main__":
    # PRNG (Pseudo Random Number Generator)
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key, 2)

    # Load data
    train_data = pickle.load(open("data/processed/train_data.pkl", "rb"))
    valid_data = pickle.load(open("data/processed/valid_data.pkl", "rb"))

    # Convert the data to JAX numpy
    train_data = [(jnp.array(img), jnp.array(label)) for img, label in train_data]
    valid_data = [(jnp.array(img), jnp.array(label)) for img, label in valid_data]

    # Load the parameters
    with open("src/params.json") as f:
        PARAMS = json.load(f)

    PARAMS = PARAMS['CNN']

    # Parameters
    IMG_SIZE = PARAMS["IMG_SIZE"]
    NUM_CLASS = PARAMS["NUM_CLASS"]
    LEARNING_RATE = PARAMS["LEARNING_RATE"]
    NUM_EPOCHS = PARAMS["NUM_EPOCHS"]

    model = CNN(num_classes=NUM_CLASS)

    # Initialize the model
    log_message("Initializing the model")
    params = model_init(IMG_SIZE, model, subkey)

    # Optimization
    log_message("Creating the optimization function")
    model_state = optimization_fn(LEARNING_RATE, model, params)

    # Training the model
    log_message("Start training")
    check_jax_device()
    _, train_acc, train_loss, valid_acc, valid_loss = train_model(model_state, train_data, valid_data)
    log_message("Training finished", "DONE")