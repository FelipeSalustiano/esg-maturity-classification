from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from src.preprocessing.data_division import get_train_test_split
from src.preprocessing.train_preprocessing import train_encoding
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("esg-maturity-train")


def train_xgb_model() -> XGBClassifier | None:

    try:
        with mlflow.start_run(run_name="xgboost"):

            dfs = get_train_test_split()
            if dfs is None:
                logging.error("Erro ao carregar dados de treino e teste.")
                return None

            df_train, _ = dfs
            X, y = train_encoding(df_train)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            model = XGBClassifier(eval_metric="logloss")

            params_grid = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "min_child_weight": [1, 3],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }

            grid = GridSearchCV(
                estimator=model,
                param_grid=params_grid,
                scoring={
                    "accuracy":  "accuracy",
                    "recall":    "recall_weighted",
                    "precision": "precision_weighted",
                    "f1":        "f1_weighted"
                },
                refit="accuracy",
                cv=skf,
                n_jobs=-1,
            )
            grid.fit(X, y)

            best_model = grid.best_estimator_
            best_index = grid.best_index_

            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                "cv_accuracy":  grid.cv_results_["mean_test_accuracy"][best_index],
                "cv_recall":    grid.cv_results_["mean_test_recall"][best_index],
                "cv_precision": grid.cv_results_["mean_test_precision"][best_index],
                "cv_f1":        grid.cv_results_["mean_test_f1"][best_index],
            })

            mlflow.sklearn.log_model(best_model, name="xgb_model")

            logging.info(f"XGBoost logado. Best params: {grid.best_params_}")
            return best_model

    except Exception as e:
        logging.exception(f"Erro durante treinamento do XGBoost: {e}")
        return None