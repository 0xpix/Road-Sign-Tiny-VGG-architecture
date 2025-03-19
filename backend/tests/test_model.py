import unittest

import jax
import jax.numpy as jnp

from src.model.model import CNN
from src.utils.utils import log_message

class TestCNNModel(unittest.TestCase):
    def setUp(self):
        """Set up the model and random input for testing."""
        self.IMG_SIZE = 32
        self.NUM_CLASS = 43
        self.key = jax.random.PRNGKey(42)
        self.key, subkey = jax.random.split(self.key, 2)

        # Initialize the model
        self.model = CNN(num_classes=self.NUM_CLASS)
        self.inp = jnp.ones((1, self.IMG_SIZE, self.IMG_SIZE, 3))
        self.params = self.model.init(subkey, self.inp)

    def test_model_output_shape(self):
        """Test whether the model output shape matches the number of classes."""
        output = self.model.apply(self.params, self.inp)
        expected_shape = (1, self.NUM_CLASS)  # Batch size 1, NUM_CLASS output neurons
        self.assertEqual(output.shape, expected_shape, f"Expected {expected_shape}, but got {output.shape}")
        log_message("Model output shape test passed", "PASS")

    def test_model_output_type(self):
        """Test whether the model output is a JAX array."""
        output = self.model.apply(self.params, self.inp)
        self.assertIsInstance(output, jnp.ndarray, "Output should be a JAX array")
        log_message("Model output type test passed", "PASS")

if __name__ == "__main__":
    unittest.main()
