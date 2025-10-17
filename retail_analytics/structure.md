databricks-retail-analytics/
│
├── README.md
├── data/
│   ├── raw/                # CSV source files (Bronze)
│   ├── reference/          # Dim tables (Products, Stores)
│
├── notebooks/
│   ├── 01_ingest_bronze.py
│   ├── 02_transform_silver.py
│   ├── 03_aggregate_gold.py
│   └── 04_dashboard_queries.sql
│
├── config/
│   └── cluster_config.json
│
└── jobs/
    └── sales_pipeline_job.json
