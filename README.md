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

**ESG Compass** é um sistema de classificação de maturidade ESG (Environmental, Social and Governance) de empresas, desenvolvido com técnicas de Machine Learning supervisionado. A partir de dados públicos e relatórios corporativos, o modelo classifica empresas em diferentes níveis de maturidade ESG, permitindo comparações setoriais e identificação de padrões de boas práticas de sustentabilidade.

O pipeline de dados é orquestrado com **Apache Airflow**, os modelos são construídos com **scikit-learn** e **XGBoost**, e toda a solução é containerizada com **Docker**.

**Repositório:** [github.com/FelipeSalustiano/esg-maturity-classification](https://github.com/FelipeSalustiano/esg-maturity-classification)

**Google Sites do Projeto:** [Google Sites](https://sites.google.com/cesar.school/projeto3-grupo4/home?authuser=1)

---

## Estrutura do Projeto

```
esg-maturity-classification/
├── airflow/
│   └── dags/           # DAGs de orquestração do pipeline de dados
├── data/               # Arquitetura medalhão
│   └── bronze/
│   └── gold/
│   └── silver/                 
├── notebooks/          # Notebooks de exploração, treinamento e avaliação
├── src/                # Código-fonte dos módulos Python
├── docker-compose.yaml # Configuração dos serviços Docker
└── requirements.txt    # Dependências Python
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

### 2. Opção A — Execução com Docker (recomendado)

Sobe todos os serviços (Airflow, banco de dados, etc.) com um único comando:

```bash
docker compose up -d
```

Acesse a interface do Airflow em: [http://localhost:8080](http://localhost:8080)
- **Usuário:** `airflow`
- **Senha:** `airflow`

Para parar os serviços:

```bash
docker compose down
```

### 3. Opção B — Execução local (sem Docker)

**Crie e ative um ambiente virtual:**

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**Instale as dependências:**

```bash
pip install -r requirements.txt
```

**Execute os notebooks:**

```bash
jupyter notebook notebooks/
```

---

## 📊 Principais Tecnologias

| Tecnologia | Uso |
|---|---|
| scikit-learn | Modelagem e avaliação |
| XGBoost | Modelo de classificação |
| pandas / numpy | Manipulação de dados |
| matplotlib / seaborn | Visualização |
| Apache Airflow | Orquestração do pipeline |
| Docker | Containerização |
| Jupyter Notebook | Exploração e análise |

---

## Contato

Em caso de dúvidas, abra uma [issue](https://github.com/FelipeSalustiano/esg-maturity-classification/issues) no repositório.
