from src.model_training.xgb_train import train_xgb_model
from src.model_evaluation.xgb_evaluate import evaluate_xgb_model
import logging
import mlflow

logging.basicConfig(level=logging.INFO)


def run_pipeline_model(acc_limiar=0.70, recall_limiar=0.70, precision_limiar=0.70):

    try:
        logging.info("Iniciando pipeline.")
        logging.info("Buscando métricas do mlflow.")

        experiment = mlflow.get_experiment_by_name("esg-maturity-evaluate")
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            logging.info("Nenhum run encontrado no mlflow, avaliando modelo do zero.")
            metrics = evaluate_xgb_model()
            if metrics is None:
                logging.error("Falha na avaliação inicial do modelo.")
                return None
        else:
            ultimo_run = runs[0]
            metrics = {
                "accuracy":  ultimo_run.data.metrics.get("accuracy", 0.0),
                "recall":    ultimo_run.data.metrics.get("recall", 0.0),
                "precision": ultimo_run.data.metrics.get("precision", 0.0),
            }
            logging.info(f"Métricas carregadas do mlflow: {metrics}")

        acc       = metrics["accuracy"]
        recall    = metrics["recall"]
        precision = metrics["precision"]
        logging.info(f"Métricas atuais: acc={acc:.2f}, recall={recall:.2f}, precision={precision:.2f}")

        passou = acc >= acc_limiar and recall >= recall_limiar and precision >= precision_limiar

        if passou:
            logging.info("Modelo ok! Métricas dentro do esperado.")
            return metrics

        logging.info("Métricas baixas, retreinando o modelo.")
        train_xgb_model()

        logging.info("Avaliando modelo retreinado.")
        metrics_novo = evaluate_xgb_model()
        if metrics_novo is None:
            logging.error("Falha na avaliação do modelo retreinado.")
            return metrics

        acc_novo       = metrics_novo["accuracy"]
        recall_novo    = metrics_novo["recall"]
        precision_novo = metrics_novo["precision"]
        logging.info(f"Métricas novas: acc={acc_novo:.2f}, recall={recall_novo:.2f}, precision={precision_novo:.2f}")

        media_antiga = (acc + recall + precision) / 3
        media_nova   = (acc_novo + recall_novo + precision_novo) / 3

        if media_nova > media_antiga:
            logging.info("Novo modelo melhorou, mantendo ele.")
            return metrics_novo
        else:
            logging.info("Novo modelo não melhorou, voltando pro modelo anterior.")
            return metrics

    except Exception as e:
        logging.error(e)