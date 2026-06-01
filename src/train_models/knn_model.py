from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from src.preprocessing.data_division import get_train_test_split
from src.preprocessing.train_preprocessing import train_encoding
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("esg-maturity-train")


def train_knn_model() -> KNeighborsClassifier | None:

    try:
        with mlflow.start_run(run_name="knn"):

            dfs = get_train_test_split()
            if dfs is None:
                logging.error("Erro ao carregar dados de treino e teste.")
                return None

            df_train, _ = dfs
            X, y = train_encoding(df_train)

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            model = KNeighborsClassifier()

            params_grid = {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
                "p": [1, 2]
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

            mlflow.sklearn.log_model(best_model, "knn_model")

            logging.info(f"KNN logado. Best params: {grid.best_params_}")
            return best_model

    except Exception as e:
        logging.exception(f"Erro durante treinamento do KNN: {e}")
        return None