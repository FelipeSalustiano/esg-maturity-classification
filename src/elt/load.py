from sqlalchemy import create_engine
from src.elt.extract import extract_data
from dotenv import load_dotenv
from urllib.parse import quote_plus
import logging
import os

logging.basicConfig(level=logging.INFO)

def load_data(filepath: str):

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

        df = extract_data()
        df.to_parquet(filepath, index=False)
        df.to_sql(
            name="esg_reporting_bronze",
            con=engine,
            if_exists="replace",
            index=False
        )
        logging.info("Hospedagem de dados no banco realizada.")

    except Exception as e:
        logging.error(e)