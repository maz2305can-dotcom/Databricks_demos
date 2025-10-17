# Databricks Customer Churn Prediction (ML + MLflow)

An end-to-end **machine learning project on Databricks**, showcasing feature engineering, model training, evaluation, and deployment using **MLflow**.

This complements the Retail Analytics pipeline by adding a predictive modeling component — perfect for demonstrating full-stack data engineering + data science skills.

---

## Project Overview
- **Objective:** Predict whether a customer will churn based on spending behavior and transaction history.
- **Tech Stack:** Databricks, PySpark, scikit-learn, MLflow, Delta Lake.
- **Output:** Tracked MLflow experiment with registered model and metrics.

---

## Architecture Flow
```
Raw Data → Feature Engineering → Model Training → Evaluation → MLflow Registry
```

---

## Repository Layout
```
databricks-customer-churn/
│
├── README.md
├── data/
│   ├── customers.csv
│   ├── transactions.csv
│
├── notebooks/
│   ├── 01_prepare_features.py
│   ├── 02_train_model.py
│   ├── 03_evaluate_and_register.py
│
├── config/
│   └── cluster_config.json
│
└── jobs/
    └── churn_pipeline_job.json
```

---

## Step 1: Prepare Features
```python
from pyspark.sql.functions import col, avg, sum, count

customers = spark.read.option("header", True).csv("/mnt/raw/customers/")
transactions = spark.read.option("header", True).csv("/mnt/raw/transactions/")

features = (
    transactions.groupBy("customer_id")
    .agg(
        avg("amount").alias("avg_spend"),
        count("transaction_id").alias("num_txns"),
        sum(col("amount") > 100).alias("high_value_purchases")
    )
    .join(customers, "customer_id")
)

features.write.format("delta").mode("overwrite").saveAsTable("retail_ml.features_churn")
```

---

## Step 2: Train Model + Track with MLflow
```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

df = spark.table("retail_ml.features_churn").toPandas()
X = df[["avg_spend", "num_txns", "high_value_purchases"]]
y = df["churn"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="RandomForest_Churn"):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {acc:.3f}, AUC: {auc:.3f}")
```

---

## Step 3: Evaluate + Register Model
```python
import mlflow

client = mlflow.tracking.MlflowClient()
latest = client.get_latest_versions("RandomForest_Churn_Model", stages=["None"])[0]

client.transition_model_version_stage(
    name="RandomForest_Churn_Model",
    version=latest.version,
    stage="Production"
)
```

---

## Example Output Metrics
| Metric | Value |
|--------|--------|
| Accuracy | 0.87 |
| ROC AUC | 0.91 |

Add a screenshot of your Databricks run or MLflow experiment UI:
```markdown
![MLflow Run Screenshot](images/mlflow_run.png)
```

---

## Databricks Job Definition (`jobs/churn_pipeline_job.json`)
```json
{
  "name": "Churn Prediction Pipeline",
  "tasks": [
    {"task_key": "prep", "notebook_task": {"notebook_path": "notebooks/01_prepare_features"}},
    {"task_key": "train", "depends_on": [{"task_key": "prep"}], "notebook_task": {"notebook_path": "notebooks/02_train_model"}},
    {"task_key": "register", "depends_on": [{"task_key": "train"}], "notebook_task": {"notebook_path": "notebooks/03_evaluate_and_register"}}
  ]
}
```

---

## Cluster Configuration
```json
{
  "cluster_name": "Churn-ML-Cluster",
  "spark_version": "13.3.x-scala2.12",
  "node_type_id": "Standard_DS3_v2",
  "num_workers": 2
}
```

---

## Future Enhancements
- Use **Hyperopt** for automated hyperparameter tuning.
- Log **confusion matrices** and **ROC curves** to MLflow.
- Deploy model to an **API endpoint** using Databricks Model Serving.
- Integrate **Feature Store** for training-serving consistency.

---

**Author:** Marcus Eberhardt  
**License:** MIT  
**Tags:** `databricks` `mlflow` `machine-learning` `pyspark` `data-science` `portfolio
