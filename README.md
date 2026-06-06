# ESG Compass — Classificação de Maturidade ESG

> Projeto 3 | Machine Learning I — CESAR School

---

## Equipe

| Nome | GitHub |
|------|--------|
| Felipe Salustiano Moreira | [@FelipeSalustiano](https://github.com/FelipeSalustiano) |
| Helder Cordeiro de Barros Filho | [@hcdebarros](https://github.com/Hcdebarros) |
| Lucas Rafael Gomes Maia da Silva | [@LucasRafaelG](https://github.com/LucasRafaelG) |
| Lucas Jerônimo Mendes | [@lucas-jmendes](https://github.com/lucas-jmendes) |
| José Kevin Sales | [@jkevinsaless](https://github.com/jkevinsaless) |

---

## Informações Acadêmicas

- **Instituição:** CESAR School
- **Disciplina:** Machine Learning I
- **Projeto:** Projeto 3

---

## Sobre a Solução

**ESG Compass** é um sistema de classificação de maturidade ESG (Environmental, Social and Governance) de empresas, desenvolvido com técnicas de Machine Learning supervisionado. A partir de dados públicos extraídos via **Kaggle API**, o modelo classifica empresas em diferentes níveis de maturidade ESG, permitindo comparações setoriais e identificação de padrões de boas práticas de sustentabilidade.

O pipeline de dados segue a **Arquitetura Medallion** (Bronze → Silver → Gold), é orquestrado com **Apache Airflow** via DAGs, e os dados são persistidos no **Supabase** e versionados em arquivos **Parquet** por camada. O modelo de classificação é construído com **XGBoost**, os experimentos são rastreados com **MLflow**, e os resultados são exibidos em dashboards interativos via **Streamlit** (Dashboard Preditivo) e **Metabase** (Dashboard Analítico). Toda a solução é containerizada com **Docker**.

Toda a etapa de análise exploratória dos dados, testes e desenvolvimento dos modelos de Machine Learning foi realizada por meio de Jupyter Notebooks, que serviram como ambiente principal para experimentação e validação das abordagens propostas.

O projeto foi inicialmente concebido com o objetivo de desenvolver uma solução preditiva para a empresa Edenred, através de um relatório corporativo. Durante o processo, foram conduzidas análises estatísticas, etapas de preparação dos dados, treinamento e avaliação de modelos. Entretanto, após os experimentos realizados, constatou-se que o conjunto de dados disponibilizado possuía uma quantidade limitada de registros, o que inviabilizou a construção de um modelo preditivo com desempenho satisfatório e capacidade adequada de generalização.

Diante dessa limitação e visando a continuidade saudável do projeto, foi adotada uma base de dados alternativa relacionada à maturidade ESG de empresas. Sobre esse novo conjunto de dados foram executadas todas as etapas do ciclo de Machine Learning, incluindo análise exploratória, tratamento dos dados, treinamento, testes e validação de um modelo de previsão, permitindo demonstrar de forma prática a aplicação das técnicas e ferramentas propostas.


**Repositório:** [github.com/FelipeSalustiano/esg-maturity-classification](https://github.com/FelipeSalustiano/esg-maturity-classification)

**Google Sites do Projeto:** [Google Sites](https://sites.google.com/cesar.school/projeto3-grupo4/home?authuser=1)

---

## Arquitetura da Pipeline

![Arquitetura da Pipeline](assets/arquitetura_pipeline.png)

> A pipeline é inteiramente orquestrada pelo **Apache Airflow** (DAGs) e containerizada com **Docker**. O fluxo vai da extração via Kaggle API, passa pelas camadas Bronze → Silver → Gold (Supabase + Parquet), pelo pipeline de ML com XGBoost, rastreamento de experimentos no MLflow, até os dashboards no Streamlit e Metabase.

---

## Estrutura do Projeto

```
esg-maturity-classification/
├── airflow/
│   ├── config/
│   ├── dags/
│   │   └── elt_esg_dag.py            # DAG de orquestração do pipeline ELT
│   ├── logs/
│   └── plugins/
├── data/                             # Arquitetura Medallion
│   ├── bronze/
│   │   └── esg_reporting_bronze.parquet
│   ├── silver/
│   │   └── esg_reporting_silver.parquet
│   └── gold/
│       └── esg_reporting_gold.parquet
├── metabase-data/                    # Dados do dashboard analítico
├── mlruns/                           # Experimentos e artefatos do MLflow
├── model/                            # Modelos serializados
├── notebooks/
│   ├── kaggle_data_analysis.ipynb
│   └── original_data_analysis.ipynb
├── src/
│   ├── elt/
│   │   ├── extract.py                # Extração via Kaggle API
│   │   ├── silver_transform.py       # Transformação camada Silver
│   │   ├── gold_transform.py         # Transformação camada Gold
│   │   └── load.py                   # Carga no Supabase
│   ├── model_evaluation/
│   │   └── xgb_evaluate.py           # Avaliação do modelo XGBoost
│   ├── model_pipeline/
│   │   └── pipeline.py               # Pipeline completo de ML
│   ├── model_training/
│   │   └── xgb_train.py              # Treinamento do modelo XGBoost
│   └── preprocessing/
│       ├── data_division.py          # Divisão treino/teste
│       ├── train_preprocessing.py    # Encoding dados de treino
│       └── test_preprocessing.py     # Encoding dados de teste
├── .env.exemple                      # Exemplo de variáveis de ambiente
├── dashboard.py                      # Dashboard Streamlit
├── docker-compose.yaml               # Configuração dos serviços Docker
├── Dockerfile
├── mlflow.db                         # Banco de dados do MLflow
└── requirements.txt                  # Dependências Python
```

---

## Pré-requisitos

Antes de iniciar, certifique-se de ter instalado:

- [Python 3.10+](https://www.python.org/downloads/)
- [Docker](https://www.docker.com/get-started) e [Docker Compose](https://docs.docker.com/compose/install/)
- [Git](https://git-scm.com/)

---

## Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/FelipeSalustiano/esg-maturity-classification.git
cd esg-maturity-classification
```

### 2. Configure as variáveis de ambiente

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.exemple .env
```

### 3. Execução com Docker 

Sobe todos os serviços (Airflow, MLflow, Metabase, banco de dados, etc.) com um único comando:

```bash
docker compose up -d
```

Subindo Streamlit:

```bash
streamlit run dashboard.py
```



Acesse os serviços:

| Serviço | URL | Usuário | Senha |
|---|---|---|---|
| Airflow | [http://localhost:8080](http://localhost:8080) | `airflow` | `airflow` |
| MLflow | [http://localhost:5000](http://localhost:5001) | — | — |
| Streamlit | [http://localhost:8501](http://localhost:8501) | — | — |
| Metabase | [http://localhost:3000](http://localhost:3000) | — | — |

Para parar os serviços:

```bash
docker compose down
```

---

## Principais Tecnologias

| Tecnologia | Versão | Uso |
|---|---|---|
| XGBoost | 3.2.0 | Modelo de classificação |
| scikit-learn | 1.7.2 | Pré-processamento e avaliação |
| pandas | 2.3.3 | Manipulação de dados |
| numpy | 2.2.6 | Operações numéricas |
| scipy | 1.15.3 | Computação científica |
| MLflow | 3.12.0 | Rastreamento de experimentos e registro de modelos |
| Apache Airflow | — | Orquestração do pipeline (DAGs) |
| Streamlit | 1.45.1 | Dashboard preditivo interativo |
| Plotly | 6.1.2 | Visualizações interativas |
| matplotlib | 3.10.3 | Visualizações estáticas |
| SQLAlchemy | 2.0.49 | ORM e conexão com banco de dados |
| psycopg2-binary | 2.9.12 | Driver PostgreSQL (Supabase) |
| kagglehub | 1.0.1 | Extração de dados via Kaggle API |
| PyArrow | 20.0.0 | Leitura e escrita de arquivos Parquet |
| joblib | 1.5.0 | Serialização de modelos |
| python-dotenv | 1.2.2 | Gerenciamento de variáveis de ambiente |
| requests | 2.34.1 | Requisições HTTP |
| Docker | — | Containerização |

---

## Contato

Em caso de dúvidas, abra uma [issue](https://github.com/FelipeSalustiano/esg-maturity-classification/issues) no repositório.