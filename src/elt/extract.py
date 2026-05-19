import pandas as pd
import logging
import os 

def extract_data(filepath: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(filepath)
        logging.info("Extração de dados de arquivo concluída.")
        return df
        
    except Exception as e:
        logging.error(e)

