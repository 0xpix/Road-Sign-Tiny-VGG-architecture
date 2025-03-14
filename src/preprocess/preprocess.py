import pandas as pd
import pickle

from PIL import Image

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from src.utils.utils import log_message

class CustomTestData():
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extracting the necessary columns
        img_path = self.data.loc[idx, "Path"]
        labels = self.data.loc[idx, "ClassId"]

        # Get the acutual image
        img_path = "data/raw/gtsrb-german-traffic-sign/" + img_path
        img = Image.open(img_path).convert("RGB")

        # Apply the transformations
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(labels, dtype=torch.long)


def process(type="train", PATH=None, transform=None, test_df=None):
    """
    Function to process the data
        :param type: str: Type of the data
        :param PATH: str: Path to the data
        :param transform: torchvision.transform: Transformations to be applied
        :param test_df: pd.DataFrame: Test DataFrame (needed for test processing)
    """
    if type == "train":
        # Load the data
        data = datasets.ImageFolder(root=PATH, transform=transform)

        # Split the data
        train_size = int(0.8 * len(data))
        valid_size = len(data) - train_size
        train, valid = random_split(data, [train_size, valid_size])

        # Create the dataloaders
        train_loader = DataLoader(train, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid, batch_size=32, shuffle=False)

        return train_loader, valid_loader

    else:
        test = CustomTestData(test_df, transform)
        test_loader = DataLoader(test, batch_size=32, shuffle=False)

        return test_loader

def get_batched(data_loader):
    """
    Function to get the data in batches
        :param data_loader: DataLoader: DataLoader object
    """
    data_list = []
    for img, label in data_loader:
        data_list.append((img.numpy(), torch.tensor(label, dtype=torch.long).numpy()))
    return data_list

def pickle_save(data, output_path):
    """
    Function to save the data
        :param data: Any: Data to be saved
        :param output_path: str: Path to save the data
    """
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

if __name__  == "__main__":
    # Paths
    TRAIN_PATH = "data/raw/gtsrb-german-traffic-sign/Train"
    TEST_PATH = "data/raw/gtsrb-german-traffic-sign/Test.csv"

    # Reading the data
    test_df = pd.read_csv(TEST_PATH)

    # Creating the transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ])

    # Starting the preprocessing
    log_message('Preprocessing the data')
    log_message('Create DataLoaders')
    train_loader, valid_loader = process("train", TRAIN_PATH, transform)
    test_loader = process("test", PATH=None, transform=transform, test_df=test_df)
    log_message('DataLoaders created', 'DONE')

    log_message("Convert the data to Numpy")
    train_data = get_batched(train_loader)
    valid_data = get_batched(valid_loader)
    test_data = get_batched(test_loader)
    log_message("Data converted", "DONE")

    # Saving the data
    pickle_save(train_data, "data/processed/train_data.pkl")
    pickle_save(valid_data, "data/processed/valid_data.pkl")
    pickle_save(test_data, "data/processed/test_data.pkl")
    log_message('Processed Data saved \n', 'DONE')

