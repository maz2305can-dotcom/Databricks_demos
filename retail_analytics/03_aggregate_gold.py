from pyspark.sql.functions import sum, month, year

df = spark.table("retail.silver_sales_clean")

agg = (
    df.groupBy("region", year("date").alias("year"), month("date").alias("month"))
      .agg(sum("sales_amount").alias("monthly_sales"))
)

agg.write.format("delta").mode("overwrite").saveAsTable("retail.gold_sales_summary")
