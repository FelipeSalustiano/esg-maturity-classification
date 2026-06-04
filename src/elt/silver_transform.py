import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus
import logging
import os

logging.basicConfig(level=logging.INFO)

def silver_transformer(filepath: str):

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

        query = "SELECT * FROM esg_reporting_bronze"

        df = pd.read_sql(query, engine)
        
        # Dropnado ducplicatas e nulos (se a linha toda for nula)
        df.drop_duplicates(inplace=True)
        df.dropna(how="all")

        # Dropando colunas sem poder preditivo
        df.drop(columns=[
            'ticker', 
            'name', 
            'logo', 
            'weburl', 
            'cik',
            'last_processing_date', 
            'total_score', 
            'total_grade'], 
            inplace=True
        )

        # Coloca todo o dataset para minúsculo
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].str.lower()         
        
        # Modificando valores da target
        df["total_level"] = df["total_level"].map({
            "medium": "no high",
            "high": "high"
        })

        df.to_parquet(filepath, index=False)
        df.to_sql(
            name="esg_reporting_silver",
            con=engine,
            if_exists='replace',
            index=False
        )
        logging.info("Criação de tabela SILVER concluída.")
        
    except Exception as e:
        logging.error(e)