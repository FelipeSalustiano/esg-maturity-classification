import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import OrdinalEncoder
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os

logging.basicConfig(level=logging.INFO)

def gold_transformer(filepath: str):

    try:
        load_dotenv()

        USER = os.getenv("user")
        PASSWORD = quote_plus(os.getenv("password"))
        HOST = os.getenv("host")
        PORT = os.getenv("port")
        DATABASE = os.getenv("database")

        engine = create_engine(
            f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
        )

        query = "SELECT * FROM esg_reporting_silver"
        df = pd.read_sql(query, engine)
        
        # Dropando colunas que não seram utilizadas no modelo de ML
        df.drop(
            "name", 
            axis=1, 
            inplace=True
        )

        # Transfomando todos os valores para lower para padronização
        df = df.map(lambda x: x.lower() if isinstance(x, str) else x)

        # Aplicando OneHotEncoder na coluna "exchange"
        df = pd.get_dummies(
            data=df,
            columns=["exchange"],
            dtype=int
        )

        # Tropando uma das colunas para evitar multicolinearidade
        df.drop("exchange_new york stock exchange, inc.", axis=1, inplace=True)

        # Aplicando OrdinalEncoder na target
        classes = {
            "medium": 0,
            "high": 1,
        }
        df["total_level"] = df["total_level"].map(classes).astype("Int64")

        df.to_csv(filepath, index=False)
        df.to_sql(
            name="esg_reporting_gold",
            con=engine,
            if_exists="replace",
            index=False
        )
        logging.info("Criação de tabela GOLD concluída.")
        
    except Exception as e:
        logging.error(e)