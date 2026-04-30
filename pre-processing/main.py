import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import kagglehub
import shutil,os
import wandb

#faz o download do dataset no kaggle
path = kagglehub.dataset_download("nikhil7280/weather-type-classification")

#copia o dataset baixado para a pasta "dataset" no diretório atual do projeto
destination = os.getcwd ()
shutil.copytree (path,os.path.join(destination, "dataset"),
dirs_exist_ok = True)

#le o arquivo CSV principal com os dados de classificação do clima
df_raw = pd.read_csv("dataset/weather_classification_data.csv",
low_memory = False 
)

#exibe informações do dataset de linhas e colunas
print(f"Shape:{df_raw.shape}")
print(f"Columns:{list(df_raw.columns[:10])}")

#inicializa o projeto no Weights & Biases
#nomeia o projeto
#job_type é o tipo de tarefa carregamento de dados brutos
#name é o nome do job pra essa tarefa
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

#nomeia do arquivo temporário
#salva o dataframe
#manda o arquivo csv pro artefato
temp_path = "temp_raw.csv"
df_raw.to_csv(temp_path, index=False )
artifact.add_file(temp_path)

#registra os dados tratados no wandb como um resumo
wandb.summary["rows"]= len(df_raw) #lista as linhas   
wandb.summary["columns"]= list(df_raw.columns) #lista os nomes das colunas


#tratamento 
#converte a coluna 'Weather Type' de valores categóricos para números em ordem (0, 1, 2...)
#labelEncoder define um valor inteiro único para cada tipo de clima
le_weather = LabelEncoder()
df_raw['Weather Type'] = le_weather.fit_transform(df_raw['Weather Type'])

#mapeamento dos valores
#mapeia os valores de cloud cover para números ordinais com posiçoes
#cria o dicionário para armazer os tipos de clima
#mantem a ordem: clear < partly <  cloudy < overcast
#começa de limpo < parcialmente nublado < nublado < nublado porem com nuvens mais escuras
cloud_order = {'clear': 0, 'partly cloudy': 1, 'cloudy': 2, 'overcast': 3}
df_raw['Cloud Cover'] = df_raw['Cloud Cover'].map(cloud_order)

#converte as variáveis categóricas em colunas binárias
#cria colunas para cada combinação de Season e Location
df_raw = pd.get_dummies(df_raw, columns=['Season', 'Location'])

#função para remover outleiers usando IQR
#função disponível em => https://www.dio.me/articles/tratando-valores-outliers-em-um-dataframe-usando-python
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

#executa a função para remover os outliers do dataset
df_processed = remove_outliers(df_raw)

#normalização
#cria uma lista com as colunas numéricas que serão normalizadas para o binário
colunas_para_normalizar = ['Temperature', 'Humidity', 'Wind Speed','Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']

#inicializa o normalizador MinMaxScaler com a escala valores para -1, 1
#formula: (X - X_min) / (X_max - X_min)
scaler = MinMaxScaler(feature_range=(-1, 1))

#normaliza as colunas nas colunas selecionadas
#a função fit_transform => aprende os parametros e aplica a transformação
df_raw[colunas_para_normalizar] = scaler.fit_transform(df_raw[colunas_para_normalizar])


#salva o dataset em um arquivo csv temporário
#onde este arquivo vai ser adicionado ao artefato para comparação com dados brutos brutos do raw
temp_path = "temp_processed.csv"
df_raw.to_csv(temp_path, index=False )
artifact.add_file(temp_path)

#envia o artefato para o Weights & Biases
wandb.log_artifact(artifact)

#atualiza o resumo com informações depois do processamento
wandb.summary["rows"]= len(df_raw) #lista as linhas   
wandb.summary["columns"]= list(df_raw.columns) #lista os nomes das colunas

#encerrra a sessão do wandb
wandb.finish()
