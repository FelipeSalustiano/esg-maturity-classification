from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


def test_encoding(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series] | None:

    try:
        if df_train is None or df_train.empty:
            logging.error("Erro ao importar TRAIN DATASET.")
            return None

        if df_test is None or df_test.empty:
            logging.error("Erro ao importar TEST DATASET.")
            return None

        X = df_train.drop(columns=["target"])
        y = df_train["target"]

        X_test = df_test.drop(columns=["target"])
        y_test = df_test["target"]

        # ENCODER NO DATASET DE TREINO

        # OrdinalEncoder para exchange e currency
        exchange_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        currency_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[["exchange"]] = exchange_encoder.fit_transform(X[["exchange"]])
        X[["currency"]] = currency_encoder.fit_transform(X[["currency"]])

        # Frequency Encoding
        industry_freq = X["industry"].value_counts().to_dict()
        X["industry"] = X["industry"].map(industry_freq)

        # Imputação de valores nulos com mediana para não ser afetado por outliers
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # ENCODERS NO TEST DATASET

        # OrdinalEncoder para exchange e currency
        X_test[["exchange"]] = exchange_encoder.transform(X_test[["exchange"]])
        X_test[["currency"]] = currency_encoder.transform(X_test[["currency"]])

        # Frequency Encoding
        X_test["industry"] = X_test["industry"].map(industry_freq).fillna(0)

        # Imputação de valores nulos com mediana para não ser afetado por outliers
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

        logging.info("TEST DATASET encodado. Retornando X_test e y_test.")
        return X_test, y_test

    except Exception as e:
        logging.exception(e)
        return None