# Using Jax to train the model
import pickle
import numpy as np
import jax
from jax import numpy as jnp
import optax
from flax.training import train_state

from src.model.model import CNN


def compute_loss_acc(state, params, batch):
    input, labels = batch
    input = jnp.transpose(input, (0, 2, 3, 1))  # Ensure NHWC format

    logits = state.apply_fn(params, input)

    # Convert integer labels to one-hot inside the traced function
    num_classes = logits.shape[-1]
    labels = jax.nn.one_hot(labels, num_classes)  # Apply within JIT-traced context

    print("Logits shape:", logits.shape)  # Debugging
    print("Labels shape after one-hot:", labels.shape)

    loss = optax.softmax_cross_entropy(logits, labels).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))

    return loss, acc

# JAX CNN training
@jax.jit
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(
        compute_loss_acc, # Function to calculate the loss
        argnums=1, # Parameters are second argument of the function
        has_aux=True # Function has additional outputs, here accuracy
    )

    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(state, state.params, batch)

    # Perform params update
    state = state.apply_gradients(grads=grads)

    # Return the updated state and the loss
    return state, loss, acc

@jax.jit
def eval_step(state, batch):
    # Determine the accuracy of the model
    loss, acc = compute_loss_acc(state, state.params, batch)
    return loss, acc

train_acc, train_loss, test_acc, test_loss = [], [], [], []

def train_model(state, train_loader, valid_loader, num_epochs=20):
    # training loop
    for epoch in range(num_epochs):
        train_batch_loss, train_batch_acc = [], []
        val_batch_loss, val_batch_acc = [], []

        for train_batch in train_loader:
            state, loss, acc = train_step(state, train_batch)
            train_batch_loss.append(loss)
            train_batch_acc.append(acc)

        for val_batch in valid_loader:
            val_loss, val_acc = eval_step(state, val_batch)
            val_batch_loss.append(val_loss)
            val_batch_acc.append(val_acc)

        # Calculate the mean loss and accuracy for the train
        epoch_train_loss = np.mean(train_batch_loss)
        epoch_train_acc = np.mean(train_batch_acc)

        # Calculate the mean loss and accuracy for the validation set
        epoch_val_loss = np.mean(val_batch_loss)
        epoch_val_acc = np.mean(val_batch_acc)

        # Append the loss and accuracy
        train_acc.append(epoch_train_acc)
        train_loss.append(epoch_train_loss)
        test_acc.append(epoch_val_acc)
        test_loss.append(epoch_val_loss)


        print(
            f"Epoch: {epoch + 1}, loss: {epoch_train_loss}, acc: {epoch_train_acc}, val_loss: {epoch_val_loss}, val_acc: {epoch_val_acc}"
        )

    return state


if __name__ == "__main__":
    IMG_SIZE = 32
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key, 2)

    # Initialize the model
    model = CNN(num_classes=43)
    inp = jnp.ones((1, IMG_SIZE, IMG_SIZE, 3))
    params = model.init(subkey, inp)

    # Test the model
    model.apply(params, inp)

    # Training State for JAX CNN Model
    learning_rate = 0.0001

    optimizer = optax.adam(learning_rate=learning_rate)
    model_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

    train_data = pickle.load(open("data/processed/train_data.pkl", "rb"))
    valid_data = pickle.load(open("data/processed/valid_data.pkl", "rb"))

    train_model(model_state, train_data, valid_data, num_epochs=200)