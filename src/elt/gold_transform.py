import pandas as pd
import logging
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

        # Encoding binária da Target
        df["target"] = (
            df["total_level"] == "high"
        ).astype(int)

        # Utilizando as features do cenário B (contém no notebook "kaggle_data_analysis" -> tópico 4.5)
        FEATURES_B = [
            "exchange",
            "currency",
            "industry"
        ]
        gold_df = df[FEATURES_B + ["target"]]

        # Salva parquet
        gold_df.to_parquet(filepath, index=False)

        # Salva PostgreSQL
        gold_df.to_sql(
            name="esg_reporting_gold",
            con=engine,
            if_exists="replace",
            index=False
        )

        logging.info("Criação da camada GOLD concluída.")

    except Exception as e:
        logging.error(e)