# Projeto de IA - Classificação de Câncer de Mama

Este é um projeto de Machine Learning que utiliza dados do Instituto de Oncologia da Universidade Medical Centre de Ljubljana, Iugoslávia, para prever a recorrência de câncer de mama em pacientes.

## Conjunto de Dados

O conjunto de dados contém informações sobre pacientes com câncer de mama, incluindo:
- Idade
- Status da menopausa
- Tamanho do tumor
- Envolvimento dos nódulos
- Presença de cápsulas nos nódulos
- Grau de malignidade
- Localização (mama esquerda/direita)
- Quadrante da mama
- Histórico de radioterapia

## Requisitos

- Python 3.7 ou superior
- PIP (gerenciador de pacotes Python)

## Instalação

1. Clone este repositório
2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # No Linux/Mac
venv\Scripts\activate     # No Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Executando o projeto

Para executar o projeto, simplesmente rode:
```bash
python modelo_cancer.py
```

O script irá:
1. Carregar o conjunto de dados de câncer de mama
2. Realizar análises exploratórias detalhadas
3. Preparar e pré-processar os dados
4. Treinar um modelo Random Forest
5. Avaliar o desempenho do modelo
6. Gerar visualizações e insights

## Resultados e Insights

O modelo gera diversos arquivos de análise e visualização:

### Análises Exploratórias
- `distribuicao_caracteristicas.png`: Mostra a distribuição das principais características por classe
- `estatisticas_descritivas.txt`: Contém estatísticas detalhadas sobre idade, grau de malignidade e tamanho do tumor por grupo
- `analise_temporal.png`: Análise da relação entre idade, tamanho do tumor e recorrência
- `correlacoes.png`: Matriz de correlação entre todas as características

### Avaliação do Modelo
- `matriz_confusao.png`: Mostra o desempenho do modelo em termos de verdadeiros/falsos positivos/negativos
- `importancia_features.png`: Mostra quais características são mais importantes para a previsão

## Interpretação dos Resultados

Os resultados podem ser interpretados da seguinte forma:

1. **Distribuição das Características**: 
   - Permite identificar padrões na idade, tamanho do tumor e outros fatores
   - Ajuda a entender a prevalência de diferentes características nos casos de recorrência

2. **Correlações**:
   - Mostra quais características estão mais relacionadas entre si
   - Identifica fatores que podem ter influência mútua no prognóstico

3. **Análise Temporal**:
   - Relaciona idade com tamanho do tumor e recorrência
   - Ajuda a identificar grupos de risco por faixa etária

4. **Importância das Características**:
   - Indica quais fatores são mais relevantes para prever a recorrência
   - Auxilia médicos a focar nos indicadores mais importantes 
