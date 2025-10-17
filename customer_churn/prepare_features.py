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
