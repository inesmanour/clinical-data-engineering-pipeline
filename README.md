# Clinical Data Engineering Pipeline

An end-to-end clinical data pipeline implementing the **Medallion Architecture (Bronze → Silver → Gold)** for cardiovascular patient data.

This project demonstrates common Data Engineering practices including raw data ingestion, schema validation, data quality controls, deduplication, feature engineering and preparation of analysis-ready datasets.

---

# Project Architecture

```
                    Heart Disease CSV
                            │
                            ▼
                    Bronze Layer
             - Raw data ingestion
             - Metadata tracking
             - Parquet conversion
                            │
                            ▼
                    Silver Layer
             - Schema validation
             - Data type conversion
             - Duplicate removal
             - Data quality checks
             - Missing value detection
                            │
                            ▼
                     Gold Layer
             - Analysis-ready dataset
             - Feature engineering
             - Dataset summary
```

---

# Project Structure

```
clinical-data-engineering-pipeline/

├── data/
│   └── heart_disease_data.csv
│
├── src/
│   ├── bronze.py
│   ├── silver.py
│   ├── gold.py
│   └── spark_pipeline.py
│
├── output/
│   ├── bronze/
│   ├── silver/
│   ├── gold/
│   └── spark/
│       ├── bronze/
│       ├── silver/
│       └── gold/
├── notebooks/
│
├── images/
│
├── run_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

The project uses the publicly available **Heart Disease Dataset**.

The dataset contains demographic and clinical information including:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise induced angina
- ST depression
- Number of major vessels
- Thalassemia
- Heart disease diagnosis (target)

---

# Bronze Layer

Purpose:

- Ingest the raw CSV dataset
- Preserve original records
- Add ingestion metadata
- Convert CSV into Parquet

Added metadata:

- source filename
- ingestion timestamp
- original row number

No clinical values are modified in this layer.

---

# Silver Layer

Purpose:

- Validate dataset schema
- Convert variables into appropriate data types
- Detect missing values
- Remove duplicated observations
- Apply quality rules

Quality controls include:

- schema validation
- allowed categorical values
- numerical range validation
- duplicate detection

Pipeline results:

- 606 raw observations
- 304 duplicated observations removed
- 302 validated patient records
- 0 rejected records

---

# Gold Layer

Purpose:

Prepare an analysis-ready dataset.

Additional engineered variables include:

- age_group
- blood_pressure_category
- cholesterol_category
- low_max_heart_rate
- risk_factor_count

A summary report is automatically generated.


---

# Pipeline Results

| Layer | Records | Description |
|-------|---------:|------------|
| Bronze | 606 | Raw data successfully ingested |
| Silver | 302 | Cleaned and validated records |
| Gold | 302 | Analysis-ready dataset |

Data Quality Summary

- Raw observations: **606**
- Duplicate records removed: **304**
- Missing values detected: **0**
- Invalid records rejected: **0**
- Final observations: **302**
---

# Pipeline Execution

Run the complete pipeline using:

```bash
python run_pipeline.py
```

Or execute each layer individually:

```bash
python src/bronze.py
python src/silver.py
python src/gold.py
```


---

# PySpark Implementation

The project includes an additional PySpark implementation of the complete Medallion pipeline.

It reproduces the Bronze, Silver and Gold transformations using Spark DataFrames and writes partitioned Parquet datasets.

The Spark implementation performs:

- CSV ingestion with an explicit schema
- Technical metadata creation
- Missing-value detection
- Clinical range validation
- Categorical-value validation
- Duplicate removal
- Feature engineering
- Partitioned Parquet output
- Aggregated Gold-layer metrics

Run the Spark pipeline using:

```bash
python src/spark_pipeline.py
```

Spark outputs are written to:

```text
output/spark/
├── bronze/
├── silver/
└── gold/
```

Unlike the Pandas implementation, Spark writes Parquet datasets as directories containing one or more partition files.

## Spark Pipeline Results

| Layer | Records | Columns |
|-------|--------:|--------:|
| Bronze | 606 | 17 |
| Silver | 302 | 18 |
| Gold | 302 | 21 |

The Spark implementation produced the same main analytical results as the Pandas implementation:

- Final validated records: **302**
- Positive target observations: **54.3%**
- Average age: **54.42**
- Average cholesterol: **246.50**
- Average resting blood pressure: **131.60**

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/clinical-data-engineering-pipeline.git
```

Create a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Technologies

- Python
- Pandas
- PyArrow
- Parquet
- Data Validation
- Medallion Architecture
- Git
- PySpark
- Apache Spark
- Java 17

---


# PySpark Version

A PySpark implementation of the pipeline is also available.

It reproduces the Bronze, Silver and Gold transformations using Spark DataFrames and writes partitioned Parquet datasets.

Run it using:

```bash
python src/spark_pipeline.py
```

This implementation demonstrates how the same Medallion Architecture can scale to distributed data processing.



---

# Technology Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Data Processing | Pandas, PySpark |
| Storage | Parquet |
| Architecture | Medallion Architecture |
| Data Validation | Custom Quality Rules |
| Version Control | Git |

---

# Learning Objectives

This project demonstrates the implementation of a production-inspired Medallion Architecture pipeline for structured clinical data. including:

- ETL pipelines
- Medallion Architecture
- Data quality validation
- Clinical data processing
- Feature engineering
- Reproducible data pipelines

An end-to-end clinical data pipeline implementing the Medallion Architecture with both Pandas and PySpark.