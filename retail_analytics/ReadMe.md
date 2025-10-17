# Databricks Retail Sales Analytics Pipeline

A modern **Medallion Architecture** project built on **Databricks + Delta Lake**, showcasing end‑to‑end data engineering and analytics. This project is ideal for demonstrating Databricks proficiency in your GitHub portfolio.

---

## Project Overview
The pipeline processes retail sales data through three layers — **Bronze**, **Silver**, and **Gold** — to produce business insights ready for BI tools or dashboards.

| Layer | Description |
|-------|--------------|
| **Bronze** | Ingest raw CSV data into Delta tables with metadata (timestamps, schema enforcement). |
| **Silver** | Clean and enrich data by joining with product and store dimensions. |
| **Gold** | Aggregate KPIs such as monthly revenue, top products, and regional trends. |

---

## Architecture Diagram
```
Source CSVs → Bronze (raw ingestion) → Silver (cleansed + enriched) → Gold (aggregated analytics)
```

---

## Tech Stack
- **Databricks (PySpark + SQL)**  
- **Delta Lake** for ACID and schema evolution  
- **Azure Data Lake / DBFS** storage  
- **Databricks Jobs** for orchestration  

---

## How to Run
1. Import the repository or notebook into your Databricks workspace.
2. Attach it to a cluster (e.g., `13.3.x-scala2.12`).
3. Upload sample data (CSV files) to `/mnt/raw/retail_sales/`.
4. Run the notebook sequentially or schedule it via Databricks Jobs.

---

## 🧩 Example PySpark Pipeline
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, current_timestamp, month, year

spark = SparkSession.builder.getOrCreate()

# Bronze – Ingest
bronze_df = (spark.read.option("header", True).csv("/mnt/raw/retail_sales/")
             .withColumn("ingestion_timestamp", current_timestamp()))
bronze_df.write.format("delta").mode("overwrite").saveAsTable("retail.bronze_sales")

# Silver – Clean + Enrich
sales_df = spark.table("retail.bronze_sales")
products_df = spark.table("reference.products")
stores_df = spark.table("reference.stores")

silver_df = (sales_df.join(products_df, "product_id", "left")
             .join(stores_df, "store_id", "left")
             .filter(col("quantity") > 0)
             .withColumn("sales_amount", col("quantity") * col("unit_price")))

silver_df.write.format("delta").mode("overwrite").saveAsTable("retail.silver_sales_clean")

# Gold – Aggregate
agg = (silver_df.groupBy("region", year("date").alias("year"), month("date").alias("month"))
               .agg(sum("sales_amount").alias("monthly_sales")))
agg.write.format("delta").mode("overwrite").saveAsTable("retail.gold_sales_summary")
```

---

## Example SQL Queries
```sql
-- Top 10 Products by Revenue
SELECT product_name, SUM(sales_amount) AS total_sales
FROM retail.gold_sales_summary
GROUP BY product_name
ORDER BY total_sales DESC
LIMIT 10;

-- Regional Sales Trends
SELECT region, year, month, SUM(monthly_sales) AS revenue
FROM retail.gold_sales_summary
GROUP BY region, year, month
ORDER BY year, month;
```

---

## Repository Layout
```
notebooks/
└── retail_pipeline_full.py   # Full pipeline (Bronze → Silver → Gold)
config/
└── cluster_config.json       # Optional Databricks cluster config
data/
├── raw/                      # Example sales CSVs
└── reference/                # Product and store lookup tables
```

---

## Example Cluster Config
```json
{
  "cluster_name": "Retail-Sales-Cluster",
  "spark_version": "13.3.x-scala2.12",
  "node_type_id": "Standard_DS3_v2",
  "num_workers": 2
}
```

---

## Results Preview
| Region | Year | Month | Monthly Sales |
|---------|------|-------|----------------|
| West | 2025 | 7 | 1,234,560.45 |
| Central | 2025 | 7 | 842,120.33 |
| East | 2025 | 7 | 1,054,992.18 |

Add a screenshot of your Databricks data preview or dashboard:
```markdown
![Gold Layer Preview](images/gold_table.png)
```

---

## Future Enhancements
- Implement **Auto Loader** for incremental ingestion.
- Add **Great Expectations** for data validation.
- Integrate **Power BI** or **Tableau** dashboards.
- Replace job JSONs with **Delta Live Tables** orchestration.


**Author:** Marcus Eberhardt  
**License:** MIT  
**Tags:** `databricks` `delta-lake` `pyspark` `data-engineering` `portfolio`
