import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

def carregar_dados():
    """Carrega o conjunto de dados Iris."""
    print("Carregando dados...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    return X, y

def preparar_dados(X, y):
    """Prepara os dados para treinamento."""
    print("Preparando dados...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def treinar_modelo(X_train, y_train):
    """Treina o modelo de Random Forest."""
    print("Treinando modelo...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

def avaliar_modelo(modelo, X_test, y_test):
    """Avalia o desempenho do modelo."""
    print("\nAvaliando modelo...")
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia do modelo: {acuracia:.2%}")
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred, target_names=['Setosa', 'Versicolor', 'Virginica']))

def visualizar_importancia_features(modelo, feature_names):
    """Cria um gráfico de importância das características."""
    print("Gerando visualização...")
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Importância das Características")
    plt.bar(range(len(importancias)), importancias[indices])
    plt.xticks(range(len(importancias)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('importancia_features.png')
    print("Visualização salva como 'importancia_features.png'")

def salvar_modelo(modelo, scaler):
    """Salva o modelo e o scaler treinados."""
    print("\nSalvando modelo...")
    with open('modelo_iris.pkl', 'wb') as f:
        pickle.dump({'modelo': modelo, 'scaler': scaler}, f)
    print("Modelo salvo como 'modelo_iris.pkl'")

def main():
    # Carrega os dados
    X, y = carregar_dados()
    
    # Prepara os dados
    X_train, X_test, y_train, y_test, scaler = preparar_dados(X, y)
    
    # Treina o modelo
    modelo = treinar_modelo(X_train, y_train)
    
    # Avalia o modelo
    avaliar_modelo(modelo, X_test, y_test)
    
    # Visualiza a importância das características
    visualizar_importancia_features(modelo, X.columns)
    
    # Salva o modelo
    salvar_modelo(modelo, scaler)

if __name__ == "__main__":
    main() 