# src/data_processing.py
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """Carrega os dados de treino e teste a partir dos caminhos dos arquivos CSV."""
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print("Dados carregados com sucesso.")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Erro ao carregar dados: {e}")
        return None, None

def preprocess_data(train_df, test_df):
    """Executa todo o pré-processamento nos dataframes."""
    # Separar labels e features
    Y_train_raw = train_df["label"]
    X_train_raw = train_df.drop(labels=["label"], axis=1)

    # Normalização
    X_train_normalized = X_train_raw / 255.0
    test_normalized = test_df / 255.0

    # Reshape para o formato de imagem (28x28x1)
    X_train_reshaped = X_train_normalized.values.reshape(-1, 28, 28, 1)
    test_reshaped = test_normalized.values.reshape(-1, 28, 28, 1)

    # One-Hot Encoding dos labels
    Y_train_encoded = to_categorical(Y_train_raw, num_classes=10)

    # Divisão em treino e validação
    # Usando random_state para garantir a reprodutibilidade
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_reshaped, Y_train_encoded, test_size=0.1, random_state=2
    )
    
    print("Pré-processamento concluído.")
    return X_train, X_val, Y_train, Y_val, test_reshaped