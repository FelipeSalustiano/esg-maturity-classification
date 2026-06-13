from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from src.preprocessing.data_division import get_train_test_split
from src.preprocessing.train_preprocessing import train_encoding
import mlflow
import mlflow.sklearn
import logging
import joblib

logging.basicConfig(level=logging.INFO)


def train_xgb_model() -> XGBClassifier | None:

    try:
        mlflow.set_tracking_uri("http://mlflow:5001")
        mlflow.set_experiment("esg-maturity-train")

        dfs = get_train_test_split()
        if dfs is None:
            logging.error("Erro ao carregar dados de treino e teste.")
            return None

        df_train, _ = dfs
        X, y = train_encoding(df_train)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = XGBClassifier(
            random_state=42,
            eval_metric="logloss", 
            scale_pos_weight=(y==0).sum()/(y==1).sum()
        )

        params_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 4],
            "min_child_weight": [1],
            "subsample": [0.8],
            "colsample_bytree": [0.8]
        }

        grid = GridSearchCV(
            estimator=model,
            param_grid=params_grid,
            scoring={
                "accuracy":  "accuracy",
                "recall":    make_scorer(recall_score,    average="weighted", zero_division=0),
                "precision": make_scorer(precision_score, average="weighted", zero_division=0),
                "f1":        make_scorer(f1_score,        average="weighted", zero_division=0),
            },
            refit="accuracy",
            cv=skf,
            n_jobs=-1,
        )
        grid.fit(X, y)

        best_model = grid.best_estimator_
        best_index = grid.best_index_


        with mlflow.start_run(run_name="xgboost_train"):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                "cv_accuracy":  grid.cv_results_["mean_test_accuracy"][best_index],
                "cv_recall":    grid.cv_results_["mean_test_recall"][best_index],
                "cv_precision": grid.cv_results_["mean_test_precision"][best_index],
                "cv_f1":        grid.cv_results_["mean_test_f1"][best_index],
            })
            mlflow.xgboost.log_model(best_model, "xgb_model")

        joblib.dump(best_model, "model/xgboost_model.pkl")
        logging.info(f"XGBoost logado. Best params: {grid.best_params_}")
        return best_model

    except Exception as e:
        logging.exception(f"Erro durante treinamento do XGBoost: {e}")
        return None