import json

import flax.linen as nn


class CNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), padding="SAME")
        x = nn.max_pool(x, window_shape=(2, 2), padding="SAME")
        x = nn.Conv(128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), padding="SAME")
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)

        return x


if __name__ == "__main__":
    with open("src/params.json") as f:
        PARAMS = json.load(f)

    NUM_CLASS = PARAMS['CNN']['NUM_CLASS']
    model = CNN(num_classes=NUM_CLASS)
