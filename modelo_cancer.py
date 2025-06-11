import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def carregar_dados():
    """Carrega o conjunto de dados de câncer de mama."""
    print("Carregando dados...")
    
    # nomes das colunas baseados no arquivo .names
    colunas = ['classe', 'idade', 'menopausa', 'tamanho_tumor', 'inv_nodes',
               'node_caps', 'deg_malig', 'mama', 'quadrante', 'irradiacao']
    
    # caminho para o arquivo de dados
    caminho_dados = os.path.join('datasets', 'breast-cancer.data')
    
    # carrega os dados
    df = pd.read_csv(caminho_dados, names=colunas)
    print(f"Total de registros carregados: {len(df)}")
    
    return df

def analisar_distribuicao_dados(df):
    """Analisa a distribuição dos dados por característica."""
    print("\nAnalisando distribuição dos dados...")
    
    # configuração do estilo das visualizações
    plt.style.use('seaborn')
    
    # criando um grid de subplots para as características principais
    caracteristicas = ['idade', 'menopausa', 'tamanho_tumor', 'inv_nodes', 'deg_malig']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, caracteristica in enumerate(caracteristicas):
        sns.countplot(data=df, x=caracteristica, hue='classe', ax=axes[idx])
        axes[idx].set_title(f'Distribuição por {caracteristica}')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('distribuicao_caracteristicas.png')
    print("Visualização da distribuição salva como 'distribuicao_caracteristicas.png'")

def analisar_correlacoes(df_encoded):
    """Analisa as correlações entre as características."""
    print("\nAnalisando correlações entre características...")
    
    # calculando a matriz de correlação
    corr_matrix = df_encoded.corr()
    
    # criando o mapa de calor
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação entre Características')
    plt.tight_layout()
    plt.savefig('correlacoes.png')
    print("Matriz de correlação salva como 'correlacoes.png'")

def analise_temporal(df):
    """Analisa a distribuição temporal por faixa etária."""
    print("\nRealizando análise temporal...")
    
    # criando gráfico de distribuição de idade vs. recorrência
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='idade', y='tamanho_tumor', hue='classe')
    plt.title('Distribuição de Tamanho do Tumor por Idade e Classe')
    plt.tight_layout()
    plt.savefig('analise_temporal.png')
    print("Análise temporal salva como 'analise_temporal.png'")

def gerar_estatisticas_descritivas(df):
    """Gera estatísticas descritivas dos dados."""
    print("\nGerando estatísticas descritivas...")
    
    # calculando estatísticas por grupo
    stats = df.groupby('classe').agg({
        'idade': ['count', 'mean'],
        'deg_malig': ['mean', 'max'],
        'tamanho_tumor': ['mean', 'max']
    }).round(2)
    
    # salvando estatísticas em um arquivo
    with open('estatisticas_descritivas.txt', 'w', encoding='utf-8') as f:
        f.write("Estatísticas Descritivas por Grupo:\n\n")
        f.write(str(stats))
    
    print("Estatísticas descritivas salvas em 'estatisticas_descritivas.txt'")

def preparar_dados(df):
    """Prepara os dados para treinamento."""
    print("\nPreparando dados...")
    
    # tratando valores ausentes
    df['node_caps'] = df['node_caps'].replace('?', df['node_caps'].mode()[0])
    df['quadrante'] = df['quadrante'].replace('?', df['quadrante'].mode()[0])
    
    # convertendo todas as colunas categóricas para numéricas
    df_encoded = df.copy()
    le = LabelEncoder()
    for coluna in df_encoded.columns:
        df_encoded[coluna] = le.fit_transform(df_encoded[coluna])
    
    # separando features e target
    X = df_encoded.drop('classe', axis=1)
    y = df_encoded['classe']
    
    # dividindo em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, df_encoded

def treinar_modelo(X_train, y_train):
    """Treina o modelo Random Forest."""
    print("\nTreinando modelo...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo

def avaliar_modelo(modelo, X_test, y_test):
    """Avalia o desempenho do modelo."""
    print("\nAvaliando modelo...")
    y_pred = modelo.predict(X_test)
    
    # calculando métricas
    acuracia = accuracy_score(y_test, y_pred)
    print(f"\nAcurácia do modelo: {acuracia:.2%}")
    
    # relatório de classificação
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Sem recorrência', 'Com recorrência']))
    
    # matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.savefig('matriz_confusao.png')
    print("\nMatriz de confusão salva como 'matriz_confusao.png'")

def visualizar_importancia_features(modelo, feature_names):
    """Cria um gráfico de importância das características."""
    print("\nGerando visualização da importância das características...")
    importancias = modelo.feature_importances_
    indices = np.argsort(importancias)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Importância das Características")
    plt.bar(range(len(importancias)), importancias[indices])
    plt.xticks(range(len(importancias)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('importancia_features.png')
    print("Visualização salva como 'importancia_features.png'")

def main():
    # carrega os dados
    df = carregar_dados()
    
    # análises exploratórias
    analisar_distribuicao_dados(df)
    gerar_estatisticas_descritivas(df)
    analise_temporal(df)
    
    # prepara os dados
    X_train, X_test, y_train, y_test, df_encoded = preparar_dados(df)
    
    # análise de correlações (após encoding)
    analisar_correlacoes(df_encoded)
    
    # treina o modelo
    modelo = treinar_modelo(X_train, y_train)
    
    # avalia o modelo
    avaliar_modelo(modelo, X_test, y_test)
    
    # visualiza a importância das características
    feature_names = ['idade', 'menopausa', 'tamanho_tumor', 'inv_nodes',
                    'node_caps', 'deg_malig', 'mama', 'quadrante', 'irradiacao']
    visualizar_importancia_features(modelo, feature_names)

if __name__ == "__main__":
    main() 
