from src.model_training.xgb_train import train_xgb_model
from src.model_evaluation.xgb_evaluate import evaluate_xgb_model
import logging
import mlflow

logging.basicConfig(level=logging.INFO)


def run_pipeline_model(acc_limiar=0.70, recall_limiar=0.70, precision_limiar=0.70):

    try:
        logging.info("Iniciando pipeline...")

        # pegar as métricas do último run salvo no mlflow
        logging.info("Buscando métricas do mlflow...")
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=["0"],  
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            logging.info("Nenhum run encontrado no mlflow, avaliando modelo do zero...")
            metrics = evaluate_xgb_model()
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

        # não passou, então vou retreinar
        logging.info("Métricas baixas, retreinando o modelo...")
        train_xgb_model()

        # avaliar de novo depois do treino e logar no mlflow
        logging.info("Avaliando modelo retreinado...")
        with mlflow.start_run():
            metrics_novo = evaluate_xgb_model()
            acc_novo       = metrics_novo["accuracy"]
            recall_novo    = metrics_novo["recall"]
            precision_novo = metrics_novo["precision"]
            logging.info(f"Métricas novas: acc={acc_novo:.2f}, recall={recall_novo:.2f}, precision={precision_novo:.2f}")

            mlflow.log_metrics(metrics_novo)

        # ver se o novo ficou melhor do que o antigo
        media_antiga = (acc + recall + precision) / 3
        media_nova   = (acc_novo + recall_novo + precision_novo) / 3

        if media_nova > media_antiga:
            logging.info("Novo modelo melhorou, mantendo ele.")
            return metrics_novo
        else:
            logging.info("Novo modelo não melhorou, voltando pro modelo anterior.")
            return metrics

    except Exception as e:
        logging.error(f"Deu erro no pipeline: {e}")