import pandas as pd
import numpy as np
import wandb
import torch
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. SETUP E CONFIGURAÇÃO (Baseado no notebook)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

config = {
    "test_size": 0.2,
    "random_state": 42,
    "target_col": "Weather Type",
    "feature_range": (-1, 1)
}

# Inicializa o W&B para a fase de Análise e Seleção
run = wandb.init(
    project="weather-classification",
    job_type="feature_selection",
    name="ensemble_importance_analysis",
    config=config
)


df = pd.read_csv("../pre-processing/temp_processed.csv")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


config = {
    "test_size": 0.2,
    "random_state": 42,
    "target_col": "Weather Type",
    "feature_range": (-1, 1)
}

run = wandb.init(
    project="weather-classification",
    job_type="feature_selection",
    name="ensemble_importance_analysis",
    config=config
)

# 4. CÁLCULO DE IMPORTÂNCIA (ENSEMBLE)
X = df.drop(columns=[config["target_col"]])
y = df[config["target_col"]]

print("Calculando importância das variáveis...")

# Método 1: Correlação de Pearson
corr_scores = X.corrwith(y).abs()

# Método 2: Mutual Information (Relações não-lineares)
mi_scores = mutual_info_classif(X, y, random_state=config["random_state"])
mi_series = pd.Series(mi_scores, index=X.columns)

# Método 3: Feature Importance de Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=config["random_state"])
rf.fit(X, y)
rf_series = pd.Series(rf.feature_importances_, index=X.columns)

def normalize(s):
    return (s - s.min()) / (s.max() - s.min())

# Combinando os três métodos (Média Simples)
final_score = (normalize(corr_scores) + normalize(mi_series) + normalize(rf_series)) / 3
final_ranking = final_score.sort_values(ascending=False)

# 6. LOG DE RESULTADOS E ARTEFATOS NO W&B
# Criando tabela rica para o Dashboard
table_data = []
for col in X.columns:
    table_data.append([
        col, 
        corr_scores[col], 
        mi_series[col], 
        rf_series[col], 
        final_score[col]
    ])

importance_table = wandb.Table(
    data=table_data, 
    columns=["Feature", "Pearson", "Mutual_Info", "Random_Forest", "Combined_Score"]
)

wandb.log({
    "importance_table": importance_table,
    "feature_ranking_plot": wandb.plot.bar(importance_table, "Feature", "Combined_Score", title="Ranking de Importância Final")
})

# Salva o ranking final como resumo
wandb.summary["top_feature"] = final_ranking.index[0]
wandb.summary["least_important"] = final_ranking.index[-1]

print("\n--- Ranking Final ---")
print(final_ranking)

wandb.finish()