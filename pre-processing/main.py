import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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

wandb.summary["rows"]= len(df_raw) #lista as linhas   
wandb.summary["columns"]= list(df_raw.columns) #lista os nomes das colunas

#tratamento de dados
le_weather = LabelEncoder()
df_raw['Weather Type'] = le_weather.fit_transform(df_raw['Weather Type'])

cloud_order = {'clear': 0, 'partly cloudy': 1, 'cloudy': 2, 'overcast': 3}
df_raw['Cloud Cover'] = df_raw['Cloud Cover'].map(cloud_order)

df_raw = pd.get_dummies(df_raw, columns=['Season', 'Location'])

def remove_outliers(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

colunas_para_normalizar = ['Temperature', 'Humidity', 'Wind Speed','Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
scaler = MinMaxScaler()
df_raw[colunas_para_normalizar] = scaler.fit_transform(df_raw[colunas_para_normalizar])

df_processed = remove_outliers(df_raw)

temp_path = "temp_processed.csv"
df_raw.to_csv(temp_path, index=False )
artifact.add_file(temp_path)

#envia o artefato para o wandb
wandb.log_artifact(artifact)

wandb.summary["rows"]= len(df_raw) #lista as linhas   
wandb.summary["columns"]= list(df_raw.columns) #lista os nomes das colunas

#finaliza o carregamento
wandb.finish()
