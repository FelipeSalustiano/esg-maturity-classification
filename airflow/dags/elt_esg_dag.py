import sys
from airflow.decorators import dag, task
from datetime import timedelta, datetime

sys.path.append("/opt/airflow")

from src.elt.extract import extract_data
from src.elt.load import load_data 
from src.elt.silver_transform import silver_transformer
from src.elt.gold_transform import gold_transformer

@dag(
    dag_id = "elt_esg_dag",
    default_args={
        "owner": "airflow",
        "depends_on_past": False,
        "retries": 3,
        "retry_delay": timedelta(minutes=5) 
    },
    description="pipeline elt esg",
    schedule="0 */1 * * *",
    start_date=datetime(2026, 5, 13),
    catchup=False
)

def esg_elt_pipline():

    @task
    def extract():
        extract_data()
    
    @task
    def load():
        load_data("data/bronze/esg_reporting_bronze.parquet")
    
    @task
    def silver():
        silver_transformer("data/silver/esg_reporting_silver.parquet")

    @task
    def gold():
        gold_transformer("data/gold/esg_reporting_gold.parquet")

    extract() >> load() >> silver() >> gold()

esg_elt_pipline()