from sklearn.model_selection import train_test_split as sklearn_train_test_split
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus
import logging
import os

logging.basicConfig(level=logging.INFO)


def get_train_test_split() -> tuple[pd.DataFrame, pd.DataFrame] | None:

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

        query = "SELECT * FROM esg_reporting_gold"
        df = pd.read_sql(query, engine)

        # Divisão de bases de treino e teste
        df_train, df_test = sklearn_train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["target"]
        )

        logging.info("Divisão de bases em TREINO e TESTE realizada.")
        return df_train, df_test

    except Exception as e:
        logging.error(e)
        return None