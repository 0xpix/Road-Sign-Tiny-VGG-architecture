"""

"""

from termcolor import colored
import jax

def log_message(message=None, level="INFO", epoch=None, epoch_train_loss=None, epoch_train_acc=None, epoch_valid_loss=None, epoch_valid_acc=None):
    """Write logs to file and print in real-time with distinct colors."""
    colors = {
        "INFO": "cyan",     # ℹ️ Cyan for informational messages
        "DONE": "green",    # ✅ Green for successful completion
        "ERROR": "red",     # ❌ Red for errors
        "WARN": "yellow",   # ⚠️ Yellow for warnings
        "PASS": "blue",     # 🟦 Blue for passed tests
        "FAIL": "magenta",  # 🟪 Magenta for failed tests
        "TRAIN": "white"    # ⚪ White for training logs
    }

    if level == "TRAIN" and all(v is not None for v in [epoch, epoch_train_loss, epoch_train_acc, epoch_valid_loss, epoch_valid_acc]):
        formatted_msg = f"Epoch: {epoch + 1} -  loss: {epoch_train_loss} - acc: {epoch_train_acc} - valid_loss: {epoch_valid_loss} - valid_acc: {epoch_valid_acc}"
    else:
        formatted_msg = f"[{level}] {message}"

    print(colored(formatted_msg, colors.get(level, "white")))  # Default to white if level is unknown


def check_jax_device():
    """
    Checks if JAX is using a GPU or CPU and prints device details.
    """
    devices = jax.devices()

    if any("cuda" in str(d).lower() or "gpu" in str(d).lower() for d in devices):
        log_message("✅ JAX is using the GPU")
    else:
        log_message("⚠️ JAX is running on the CPU. Install CUDA-enabled JAX for GPU support.")