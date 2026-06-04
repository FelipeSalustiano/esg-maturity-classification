from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from src.preprocessing.data_division import get_train_test_split
from src.preprocessing.test_preprocessing import test_encoding
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import logging
import joblib

logging.basicConfig(level=logging.INFO)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("esg-maturity-evaluate")


def evaluate_xgb_model() -> XGBClassifier | None:

    try:
        with mlflow.start_run(run_name="xgboost_evaluate"):

            dfs = get_train_test_split()
            if dfs is None:
                logging.error("Erro ao carregar dados de treino e teste.")
                return None

            df_train, df_test = dfs
            X_test, y_test = test_encoding(df_train, df_test)

            model = joblib.load("model/xgboost_model.pkl")

            y_pred = model.predict(X_test)

            acc         =  accuracy_score(y_test, y_pred)
            recall      =  recall_score(y_test, y_pred)
            precision   =  precision_score(y_test, y_pred)
            f1          =  f1_score(y_test, y_pred)

            mlflow.log_metrics({
                "accuracy": acc,
                "recall": recall,
                "precision": precision,
                "f1": f1
            })

            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig("assets/confusion_matrix.png")
            plt.close()
            mlflow.log_artifact("assets/confusion_matrix.png")

            mlflow.sklearn.log_model(model, "xgb_model")

            logging.info("Previsão realizada! Métricas salvas no MLflow.")
            return model

    except Exception as e:
        logging.exception(f"Erro durante treinamento do XGBoost: {e}")
        return None