"""
    Download data from Kaggle: GTSRB - German Traffic Sign Recognition Dataset by Mykola


"""
import zipfile
import os
import opendatasets as od # type: ignore

from src.utils.utils import log_message


# GTSRB - German Traffic Sign Recognition Dataset by Mykola
DATASET_URL = "https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
TARGET_PATH = "data/raw/"

log_message("Downloading GTSRB - German Traffic Sign Recognition Dataset by Mykola")
od.download(DATASET_URL, data_dir=TARGET_PATH)

# remove the zip file
os.remove(os.path.join(TARGET_PATH, "gtsrb-german-traffic-sign.zip"))

log_message("✅ Download complete", "DONE")



