from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StructField,
    StructType,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "heart_disease_data.csv"

SPARK_OUTPUT_ROOT = PROJECT_ROOT / "output" / "spark"

BRONZE_OUTPUT = SPARK_OUTPUT_ROOT / "bronze"
SILVER_OUTPUT = SPARK_OUTPUT_ROOT / "silver"
GOLD_OUTPUT = SPARK_OUTPUT_ROOT / "gold"


CLINICAL_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


SCHEMA = StructType(
    [
        StructField("age", IntegerType(), True),
        StructField("sex", IntegerType(), True),
        StructField("cp", IntegerType(), True),
        StructField("trestbps", IntegerType(), True),
        StructField("chol", IntegerType(), True),
        StructField("fbs", IntegerType(), True),
        StructField("restecg", IntegerType(), True),
        StructField("thalach", IntegerType(), True),
        StructField("exang", IntegerType(), True),
        StructField("oldpeak", DoubleType(), True),
        StructField("slope", IntegerType(), True),
        StructField("ca", IntegerType(), True),
        StructField("thal", IntegerType(), True),
        StructField("target", IntegerType(), True),
    ]
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("ClinicalDataEngineeringPipeline")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def build_bronze(spark: SparkSession) -> DataFrame:
    logging.info("Lecture du CSV avec Spark : %s", INPUT_FILE)

    bronze_df = (
        spark.read
        .option("header", True)
        .schema(SCHEMA)
        .csv(str(INPUT_FILE))
    )

    bronze_df = (
        bronze_df
        .withColumn(
            "_source_file",
            F.lit(INPUT_FILE.name),
        )
        .withColumn(
            "_ingestion_timestamp_utc",
            F.current_timestamp(),
        )
        .withColumn(
            "_source_row_number",
            F.monotonically_increasing_id() + 1,
        )
    )

    logging.info(
        "Bronze Spark : %d lignes et %d colonnes",
        bronze_df.count(),
        len(bronze_df.columns),
    )

    (
        bronze_df.write
        .mode("overwrite")
        .parquet(str(BRONZE_OUTPUT))
    )

    logging.info("Bronze Spark enregistré : %s", BRONZE_OUTPUT)

    return bronze_df


def build_silver(bronze_df: DataFrame) -> DataFrame:
    logging.info("Construction de la couche Silver Spark")

    missing_condition = None

    for column in CLINICAL_COLUMNS:
        condition = F.col(column).isNull()

        if missing_condition is None:
            missing_condition = condition
        else:
            missing_condition = missing_condition | condition

    silver_df = (
        bronze_df
        .dropDuplicates(CLINICAL_COLUMNS)
        .filter(~missing_condition)
        .filter(F.col("age").between(18, 100))
        .filter(F.col("trestbps").between(50, 250))
        .filter(F.col("chol").between(50, 700))
        .filter(F.col("thalach").between(40, 250))
        .filter(F.col("oldpeak").between(0, 10))
        .filter(F.col("sex").isin(0, 1))
        .filter(F.col("cp").isin(0, 1, 2, 3))
        .filter(F.col("fbs").isin(0, 1))
        .filter(F.col("restecg").isin(0, 1, 2))
        .filter(F.col("exang").isin(0, 1))
        .filter(F.col("slope").isin(0, 1, 2))
        .filter(F.col("ca").isin(0, 1, 2, 3, 4))
        .filter(F.col("thal").isin(0, 1, 2, 3))
        .filter(F.col("target").isin(0, 1))
        .withColumn(
            "_silver_processed_timestamp_utc",
            F.current_timestamp(),
        )
    )

    logging.info(
        "Silver Spark : %d lignes et %d colonnes",
        silver_df.count(),
        len(silver_df.columns),
    )

    (
        silver_df.write
        .mode("overwrite")
        .parquet(str(SILVER_OUTPUT))
    )

    logging.info("Silver Spark enregistré : %s", SILVER_OUTPUT)

    return silver_df


def build_gold(silver_df: DataFrame) -> DataFrame:
    logging.info("Construction de la couche Gold Spark")

    gold_df = (
        silver_df
        .select(*CLINICAL_COLUMNS)
        .withColumn(
            "patient_record_id",
            F.monotonically_increasing_id() + 1,
        )
        .withColumn(
            "age_group",
            F.when(F.col("age") < 40, "under_40")
            .when(F.col("age") < 50, "40_49")
            .when(F.col("age") < 60, "50_59")
            .when(F.col("age") < 70, "60_69")
            .otherwise("70_plus"),
        )
        .withColumn(
            "blood_pressure_category",
            F.when(F.col("trestbps") < 120, "normal")
            .when(F.col("trestbps") < 130, "elevated")
            .when(
                F.col("trestbps") < 140,
                "hypertension_stage_1",
            )
            .when(
                F.col("trestbps") < 180,
                "hypertension_stage_2",
            )
            .otherwise("hypertensive_crisis"),
        )
        .withColumn(
            "cholesterol_category",
            F.when(F.col("chol") < 200, "desirable")
            .when(F.col("chol") < 240, "borderline_high")
            .otherwise("high"),
        )
        .withColumn(
            "low_max_heart_rate",
            F.when(F.col("thalach") < 100, 1).otherwise(0),
        )
        .withColumn(
            "risk_factor_count",
            (
                F.when(F.col("trestbps") >= 140, 1).otherwise(0)
                + F.when(F.col("chol") >= 240, 1).otherwise(0)
                + F.when(F.col("fbs") == 1, 1).otherwise(0)
                + F.when(F.col("exang") == 1, 1).otherwise(0)
            ),
        )
        .withColumn(
            "_gold_processed_timestamp_utc",
            F.current_timestamp(),
        )
    )

    logging.info(
        "Gold Spark : %d lignes et %d colonnes",
        gold_df.count(),
        len(gold_df.columns),
    )

    (
        gold_df.write
        .mode("overwrite")
        .parquet(str(GOLD_OUTPUT))
    )

    logging.info("Gold Spark enregistré : %s", GOLD_OUTPUT)

    return gold_df


def show_summary(gold_df: DataFrame) -> None:
    logging.info("Résumé Gold Spark")

    gold_df.select(
        F.count("*").alias("number_of_records"),
        F.round(F.avg("age"), 2).alias("average_age"),
        F.round(F.avg("chol"), 2).alias("average_cholesterol"),
        F.round(
            F.avg("trestbps"),
            2,
        ).alias("average_resting_blood_pressure"),
        F.round(
            F.avg("target") * 100,
            2,
        ).alias("target_1_percentage"),
    ).show(truncate=False)


def main() -> None:
    configure_logging()

    spark = None

    try:
        spark = create_spark_session()

        bronze_df = build_bronze(spark)
        silver_df = build_silver(bronze_df)
        gold_df = build_gold(silver_df)

        show_summary(gold_df)

        gold_df.show(5, truncate=False)

        logging.info("Pipeline Spark terminé avec succès.")

    except Exception as error:
        logging.exception("Échec du pipeline Spark : %s", error)
        sys.exit(1)

    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()