import mlflow

client = mlflow.tracking.MlflowClient()
latest = client.get_latest_versions("RandomForest_Churn_Model", stages=["None"])[0]

client.transition_model_version_stage(
    name="RandomForest_Churn_Model",
    version=latest.version,
    stage="Production"
)
