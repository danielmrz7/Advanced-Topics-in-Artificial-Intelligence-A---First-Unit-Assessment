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

#inicializa o carregamento dos dados
#job_type é a tarefa q vai ser feita, ou seja, vai carregar todos os dados
#nomeia o projeto
wandb.init(
    project="weather-classification",
    job_type="load_raw",
    name="load_weather_data"
)
#vai criar um artefato para versionar e para armazenar os dados
#cria a categoria
#e adiciona a descrição
artifact = wandb.Artifact (
    name = "raw_data", 
    type = "dataset",
    description = "Wheater Type raw dataset from Kaggle"
)
#nome do arquivo temporário
#salva o dataframe
#manda o arquivo csv pro artefato
temp_path = "temp_raw.csv"
df_raw.to_csv(temp_path, index=False )
artifact.add_file(temp_path)

#envia o artefato para o wandb
wandb.log_artifact(artifact)


wandb.summary["rows"]= len(df_raw) #lista as linhas   
wandb.summary["columns"]= list(df_raw.columns) #lista os nomes das colunas

#finaliza o carregamento
wandb.finish()


