import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import ks_2samp, chi2_contingency
import kagglehub
import shutil,os
import wandb

#Copy the dataset files to the project directory
path = kagglehub.dataset_download("nikhil7280/weather-type-classification")

destination = os.getcwd ()
shutil.copytree (path,os.path.join(destination, "dataset"),
dirs_exist_ok = True)

#Load the main accidents file
df_raw = pd.read_csv("dataset/weather_classification_data.csv",
low_memory = False
)

print(f"Shape:{df_raw.shape}")
print(f"Columns:{list(df_raw.columns[:10])}")



