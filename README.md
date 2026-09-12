# Pipeline Preditivo MNIST — Comparação de Modelos de Machine Learning

Projeto avaliativo do Módulo 2, com foco no desenvolvimento de um pipeline
completo de Machine Learning para classificação de dígitos manuscritos
(dataset MNIST), incluindo comparação entre algoritmos clássicos e uma
rede neural, além de testes de robustez com dados fora da distribuição
de treino e imagens manuscritas próprias.

## 📋 Sobre o projeto

O projeto percorre todo o ciclo de um problema de classificação supervisionada:
desde a análise exploratória dos dados brutos até a avaliação crítica dos
modelos em cenários adversos (classes nunca vistas e imagens reais fora
do dataset original). Foram treinados e comparados três modelos distintos —
**KNN**, **Random Forest** e uma **Rede Neural (MLP)** — com ajuste de
hiperparâmetros validado em um conjunto de validação isolado.

## 🎯 Resultados principais (conjunto de teste)

| Modelo | Acurácia (Teste) | Tempo de Treino | Tempo de Predição (10.500 imgs) |
|---|---|---|---|
| KNN (`n_neighbors=5`, `weights=distance`) | 97,10% | 0,08s | 42,14s |
| Random Forest (`n_estimators=100`, `max_depth=None`) | 96,63% | 27,38s | 0,42s |
| **Rede Neural — MLP** (`128/64 neurônios`, `lr=0.001`) | **97,66%** ⭐ | 92,95s | 1,55s |

> A Rede Neural apresentou a melhor acurácia geral e foi eleita o modelo
> final para os testes de inferência com imagens próprias (Desafio C).
> O KNN, apesar de treinar quase instantaneamente, mostrou um custo de
> predição muito superior aos demais — evidenciando o trade-off entre
> tempo de treino e tempo de inferência entre os algoritmos.

## 🧪 Desafios de robustez (Fase 5)

- **Desafio A/B — Class Masking + Teste OOD**: um Random Forest foi
  treinado sem nunca ver os dígitos **4** e **7**. Ao ser forçado a
  classificar imagens exclusivamente desses dígitos, ~84% delas foram
  classificadas como **9**, com confiança média de 57% — evidenciando o
  fenômeno de *falsa certeza (overconfidence)*: o modelo nunca "admite"
  desconhecimento, mesmo diante de padrões nunca vistos no treino.
- **Desafio C — Inferência com imagens manuscritas próprias**: o dígito
  **7** foi desenhado em papel (foto) e no Paint/GIMP. A versão digital
  (Paint) foi classificada corretamente com 99,89% de confiança; a foto
  do papel foi classificada incorretamente como **9** (40,4%), com o
  dígito correto (7, 24,7%) e o dígito 4 (19,7%) como segunda e terceira
  opções — reproduzindo, na prática, o mesmo "trio de confusão"
  identificado no teste OOD, e ilustrando o conceito de *distribution shift*
  entre dados de treino padronizados e imagens do mundo real.

## 🛠️ Tecnologias utilizadas

- Python 3.11
- scikit-learn (KNN, Random Forest, métricas)
- TensorFlow / Keras (Rede Neural MLP)
- pandas, numpy
- matplotlib, seaborn (visualizações e matrizes de confusão)
- Pillow (PIL) e scipy (pré-processamento de imagens manuscritas)

## 📂 Estrutura do repositório

Mnist_Predictive_Pipeline/
├── data/ # Imagens manuscritas próprias (Desafio C)
├── mnist_pipeline.ipynb # Notebook principal com todo o pipeline
├── requirements.txt # Dependências do projeto
└── README.md


## 🚀 Como rodar o projeto

```bash
# 1. Clonar o repositório
git clone https://github.com/rodrigo-seibt/Mnist_Predictive_Pipeline
cd Mnist_Predictive_Pipeline

# 2. Criar e ativar o ambiente virtual
python -m venv venv
source venv/Scripts/activate      # Git Bash (Windows)
# venv\Scripts\activate           # PowerShell (Windows)
# source venv/bin/activate        # Mac/Linux

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Abrir o notebook no VS Code (ou Jupyter) e selecionar o kernel do venv
code .
```

Depois de aberto, execute as células em ordem (`Kernel > Restart Kernel and Run All Cells`).
O carregamento inicial do MNIST via `fetch_openml` pode levar alguns minutos.

## 📊 Fases do projeto

| Fase | Conteúdo |
|---|---|
| **1** | Carregamento do MNIST e Análise Exploratória (EDA): dimensionalidade, distribuição de classes, visualização dos dígitos |
| **2** | Normalização dos pixels (0-255 → 0.0-1.0) e divisão estratificada em Treino (70%) / Validação (15%) / Teste (15%) |
| **3** | Treinamento e ajuste de hiperparâmetros dos 3 modelos (KNN, Random Forest, Rede Neural MLP), validados no conjunto de validação |
| **4** | Avaliação comparativa final no conjunto de teste: acurácia, matrizes de confusão, tempo de treino/predição |
| **5.1 / 5.2** | Desafio de Class Masking (ocultação dos dígitos 4 e 7 no treino) e teste de generalização extrema (OOD) |
| **5.3** | Pipeline de pré-processamento e inferência com imagens manuscritas próprias (papel e digital) |

## 🌳 Estratégia de branches

Desenvolvimento organizado em `main` (entrega final) → `develop` (integração)
→ `feature/*` (uma branch por fase do projeto), seguindo o fluxo Git Flow
simplificado.

## 🎥 Vídeo de apresentação

[Link do vídeo aqui, após gravação]

## 👤 Autor

Rodrigo Seibt