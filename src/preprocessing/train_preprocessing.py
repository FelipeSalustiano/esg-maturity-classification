from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.preprocessing.data_division import get_train_test_split
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


def train_encoding(df_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    try:
        if df_train is None or df_train.empty:
            logging.info("Erro ao importar TRAIN DATASET.")

        X = df_train.drop(columns=["target"])
        y = df_train["target"]

        # Ordinal Encoding para grades ESG
        GRADE_ORDER = ['C', 'CCC', 'B', 'BB', 'BBB', 'A', 'AA']
        grade_cols = ['environment_grade', 'social_grade', 'governance_grade']
        oe = OrdinalEncoder(
            categories=[GRADE_ORDER] * len(grade_cols),
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        X[grade_cols] = oe.fit_transform(X[grade_cols])

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