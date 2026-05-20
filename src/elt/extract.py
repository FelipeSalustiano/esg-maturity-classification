import pandas as pd
import logging
import os 
import kagglehub

def extract_data() -> pd.DataFrame:

    try:
        path = kagglehub.dataset_download(
            "alistairking/public-company-esg-ratings-dataset"
        )

        df = pd.read_csv(f"{path}/data.csv")
        return df
        
    except Exception as e:
        logging.error(e)