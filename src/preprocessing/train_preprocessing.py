from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


def train_encoding(df_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    try:
        if df_train is None or df_train.empty:
            logging.info("Erro ao importar TRAIN DATASET.")

        X = df_train.drop(columns=["target"])
        y = df_train["target"]

        # Label Encoding
        exchange_encoder = LabelEncoder()
        currency_encoder = LabelEncoder()
        X["exchange"] = exchange_encoder.fit_transform(X["exchange"])
        X["currency"] = currency_encoder.fit_transform(X["currency"])

        # Frequency Encoding
        industry_freq = X["industry"].value_counts().to_dict()
        X["industry"] = X["industry"].map(industry_freq)

        # Imputação de valore nulos com mediana para não ser afetado por outliers
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        logging.info("TRAIN DATASET encodado. Retornando X e y")
        return X, y

    except Exception as e:
        logging.info(e)
        return None