from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

spark = SparkSession.builder.getOrCreate()

raw_df = spark.read.option("header", True).csv("/mnt/raw/retail_sales/")
bronze_df = raw_df.withColumn("ingestion_timestamp", current_timestamp())

bronze_df.write.format("delta").mode("overwrite").saveAsTable("retail.bronze_sales")
