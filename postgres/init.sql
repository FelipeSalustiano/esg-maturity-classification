CREATE USER airflow WITH PASSWORD 'airflow';
CREATE USER mlflow WITH PASSWORD 'mlflow';
CREATE USER metabase WITH PASSWORD 'metabase';

CREATE DATABASE airflow OWNER airflow;
CREATE DATABASE mlflow OWNER mlflow;
CREATE DATABASE metabase OWNER metabase;