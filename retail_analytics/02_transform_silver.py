from pyspark.sql.functions import col

sales_df = spark.table("retail.bronze_sales")
products_df = spark.table("reference.products")
stores_df = spark.table("reference.stores")

silver_df = (
    sales_df.join(products_df, "product_id", "left")
             .join(stores_df, "store_id", "left")
             .filter(col("quantity") > 0)
             .withColumn("sales_amount", col("quantity") * col("unit_price"))
)

silver_df.write.format("delta").mode("overwrite").saveAsTable("retail.silver_sales_clean")
