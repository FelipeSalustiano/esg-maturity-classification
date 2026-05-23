from sqlalchemy import create_engine
from src.elt.extract import extract_data
from dotenv import load_dotenv
from urllib.parse import quote_plus
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO)

def return_data(medalion_name: str) -> pd.Dataframe:

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

        query = f"SELECT * FROM esg_reporting{medalion_name.lower().strip()}"
        df = pd.read_sql(query, engine)

        logging.info(f"Dados do nível {medalion_name} retornado!")
        return df

    except Exception as e:
        logging.error(e)