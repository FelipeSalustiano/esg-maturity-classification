import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import OrdinalEncoder
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os

logging.basicConfig(level=logging.INFO)


def gold_transformer(silver_file: str, filepath: str):
    
    try:
        df = pd.read_csv(silver_file)

        # Encoding de Sim e Não -> 1 e 0
        sim_extended_cols = ["possui_politica_esg", "inventario_auditado"]
        for col in sim_extended_cols:
            df[col] = df[col].apply(lambda x: "Sim" if str(x).startswith("Sim") else x)
 
        binary_cols = [
            "possui_processos_legais",
            "recebeu_sancao",
            "trabalho_digno",
            "possui_certificacoes",
            "possui_politica_esg",
            "treinamento_sustentabilidade",
            "possui_compliance",
            "controla_impactos_ambientais",
            "gestao_agua",
            "eficiencia_energetica",
            "gerenciamento_residuos",
            "calcula_pegada_carbono",
            "inventario_auditado",
            "possui_diversidade",
            "possui_voluntariado",
            "possui_saude_mental",
            "compras_sustentaveis",
            "criterio_fornecedores",
            "clausulas_esg_fornecedores",
            "treinamento_fornecedores",
            "diversidade_fornecedores",
            "auditoria_fornecedores",
            "dados_confirmados"
        ]
        for col in binary_cols:
            df[col] = df[col].map({"Sim": 1, "Não": 0}).fillna(0).astype(int)

        # OrdinalEncoder consciente — qtd_colaboradores 
        qtd_col_order = [
            "Até 9 (Microempresa)",
            "10 a 99 (Pequeno porte)",
            "100 a 999 (Médio porte)",
            "1.000 ou mais (Grande porte)",
        ]
        enc_qtd = OrdinalEncoder(
            categories=[qtd_col_order],
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        )
        df["qtd_colaboradores"] = (
            enc_qtd.fit_transform(df[["qtd_colaboradores"]]).flatten().astype(float)
        )

        # OrdinalEncoder consciente — faturamento_anual 
        fat_order = [
            "Prefere não informar",
            "Até R$ 500 mil",
            "De R$ 500 mil a R$ 1,5 milhões",
            "De R$ 1,5 milhões a R$ 3 milhões",
            "Acima de R$ 3 milhões",
        ]
        enc_fat = OrdinalEncoder(
            categories=[fat_order],
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        )
        df["faturamento_anual"] = (
            enc_fat.fit_transform(df[["faturamento_anual"]]).flatten().astype(float)
        )

        # Colunas invertidas para indicadores negativos 
        df["possui_processos_legais_inv"] = 1 - df["possui_processos_legais"]
        df["recebeu_sancao_inv"] = 1 - df["recebeu_sancao"]

        # Feature engineering extra 
        df["hora_inicio"] = pd.to_datetime(df["hora_inicio"])
        df["hora_conclusao"] = pd.to_datetime(df["hora_conclusao"])
        df["tempo_preenchimento_min"] = (
            (df["hora_conclusao"] - df["hora_inicio"])
            .dt.total_seconds()
            .div(60)
            .round(2)
        )

        # Interação ordinal porte × faturamento
        df["porte_x_faturamento"] = (
            df["qtd_colaboradores"].astype(float)
            * df["faturamento_anual"].astype(float)
        )

        # Soma bruta de todas as práticas ESG declaradas
        df["total_praticas_esg"] = df[[c for c in binary_cols if c in df.columns]].sum(axis=1)

        # Scores por pilar 
        cols_governanca = [
            "possui_processos_legais_inv",  # invertida
            "recebeu_sancao_inv",           # invertida
            "trabalho_digno",
            "possui_certificacoes",
            "possui_politica_esg",
            "possui_compliance",
        ]
        cols_social = [
            "treinamento_sustentabilidade",
            "possui_diversidade",
            "possui_voluntariado",
            "possui_saude_mental",
            "treinamento_fornecedores",
            "diversidade_fornecedores",
        ]
        cols_ambiental = [
            "controla_impactos_ambientais",
            "gestao_agua",
            "eficiencia_energetica",
            "gerenciamento_residuos",
            "calcula_pegada_carbono",
            "inventario_auditado",
        ]
        cols_cadeia = [
            "compras_sustentaveis",
            "criterio_fornecedores",
            "clausulas_esg_fornecedores",
            "auditoria_fornecedores",
        ]

        df["score_governanca"] = df[[c for c in cols_governanca if c in df.columns]].mean(axis=1).round(2)
        df["score_social"]     = df[[c for c in cols_social     if c in df.columns]].mean(axis=1).round(2)
        df["score_ambiental"]  = df[[c for c in cols_ambiental  if c in df.columns]].mean(axis=1).round(2)
        df["score_cadeia"]     = df[[c for c in cols_cadeia     if c in df.columns]].mean(axis=1).round(2)

        # Score ESG total ponderado
        df["score_esg_total"] = (
            df["score_governanca"] * 0.30
            + df["score_ambiental"] * 0.30
            + df["score_social"] * 0.25
            + df["score_cadeia"] * 0.15
        ).round(2)

        # Target: maturidade_esg (baixa / media / alta) por tercis 
        labels = ["baixa", "media", "alta"]
        
        df["maturidade_esg"] = pd.cut(
            df["score_esg_total"],
            bins=3,
            labels=labels,
        )

        # Drop: apenas cargo e colunas de data
        cols_to_drop = [
            "cargo",
            "hora_inicio",
            "hora_conclusao",
            "ultima_modificacao",
            "possui_processos_legais_inv",  # intermediária, já absorvida no score
            "recebeu_sancao_inv",           # intermediária, já absorvida no score
        ]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # Persistência: CSV + banco 
        load_dotenv()

        USER = os.getenv("user")
        PASSWORD = quote_plus(os.getenv("password"))
        HOST = os.getenv("host")
        PORT = os.getenv("port")
        DATABASE = os.getenv("database")
        
        engine = create_engine(
            f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
        )
        df.to_csv(filepath, index=False)
        df.to_sql(
            name="ESGreportingGOLD",
            con=engine,
            if_exists="replace",
            index=False,
        )
        logging.info("Criação de tabela GOLD concluída.")

        return df

    except Exception as e:
        logging.error(e)