from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# CHEMINS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "output"
    / "silver"
    / "heart_silver.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "gold"

OUTPUT_FILE = OUTPUT_DIR / "heart_gold.parquet"
SUMMARY_FILE = OUTPUT_DIR / "heart_gold_summary.csv"


# ============================================================
# COLONNES
# ============================================================

FEATURE_COLUMNS = [
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
]

TARGET_COLUMN = "target"


# ============================================================
# LOGGING
# ============================================================

def configure_logging() -> None:
    """Configure les logs affichés dans le terminal."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ============================================================
# CHARGEMENT SILVER
# ============================================================

def load_silver_data(file_path: Path) -> pd.DataFrame:
    """Charge les données nettoyées de la couche Silver."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier Silver introuvable : {file_path}\n"
            "Lance d'abord src/silver.py."
        )

    logging.info("Lecture de la couche Silver : %s", file_path)

    dataframe = pd.read_parquet(file_path)

    if dataframe.empty:
        raise ValueError("La couche Silver ne contient aucune ligne.")

    logging.info(
        "Couche Silver chargée : %d lignes et %d colonnes",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    return dataframe


# ============================================================
# VALIDATION
# ============================================================

def validate_gold_columns(dataframe: pd.DataFrame) -> None:
    """Vérifie que les variables nécessaires sont présentes."""

    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in expected_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes nécessaires absentes : {missing_columns}"
        )

    logging.info("Colonnes nécessaires à la couche Gold validées.")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_gold_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Crée quelques variables simples utilisables pour l'analyse
    et le machine learning.
    """

    gold_df = dataframe[
        FEATURE_COLUMNS + [TARGET_COLUMN]
    ].copy()

    # Groupes d'âge
    gold_df["age_group"] = pd.cut(
        gold_df["age"],
        bins=[0, 39, 49, 59, 69, 120],
        labels=[
            "under_40",
            "40_49",
            "50_59",
            "60_69",
            "70_plus",
        ],
        include_lowest=True,
    )

    # Catégorie de pression artérielle
    gold_df["blood_pressure_category"] = pd.cut(
        gold_df["trestbps"],
        bins=[0, 119, 129, 139, 179, 300],
        labels=[
            "normal",
            "elevated",
            "hypertension_stage_1",
            "hypertension_stage_2",
            "hypertensive_crisis",
        ],
        include_lowest=True,
    )

    # Catégorie simple du cholestérol
    gold_df["cholesterol_category"] = pd.cut(
        gold_df["chol"],
        bins=[0, 199, 239, 1000],
        labels=[
            "desirable",
            "borderline_high",
            "high",
        ],
        include_lowest=True,
    )

    # Indicateur de fréquence cardiaque maximale faible
    gold_df["low_max_heart_rate"] = (
        gold_df["thalach"] < 100
    ).astype("int8")

    # Nombre de facteurs de risque simples
    gold_df["risk_factor_count"] = (
        (gold_df["trestbps"] >= 140).astype("int8")
        + (gold_df["chol"] >= 240).astype("int8")
        + (gold_df["fbs"] == 1).astype("int8")
        + (gold_df["exang"] == 1).astype("int8")
    )

    # Identifiant technique stable pour les lignes Gold
    gold_df.insert(
        0,
        "patient_record_id",
        range(1, len(gold_df) + 1),
    )

    gold_df["_gold_processed_timestamp_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    return gold_df


# ============================================================
# CONTRÔLES GOLD
# ============================================================

def validate_gold_data(dataframe: pd.DataFrame) -> None:
    """Effectue les derniers contrôles avant sauvegarde."""

    if dataframe.isna().any().any():
        missing_counts = dataframe.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]

        raise ValueError(
            "Valeurs manquantes détectées dans la couche Gold : "
            f"{missing_counts.to_dict()}"
        )

    if dataframe["patient_record_id"].duplicated().any():
        raise ValueError(
            "Des identifiants patient_record_id sont dupliqués."
        )

    if not dataframe[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(
            "La variable target doit uniquement contenir 0 ou 1."
        )

    logging.info("Contrôles qualité Gold validés.")


# ============================================================
# RÉSUMÉ DU DATASET
# ============================================================

def create_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Crée un petit résumé utile pour le README et le contrôle qualité."""

    summary_df = pd.DataFrame(
        {
            "metric": [
                "number_of_records",
                "number_of_features",
                "target_0_count",
                "target_1_count",
                "target_1_percentage",
                "average_age",
                "average_cholesterol",
                "average_resting_blood_pressure",
            ],
            "value": [
                len(dataframe),
                len(FEATURE_COLUMNS),
                int((dataframe[TARGET_COLUMN] == 0).sum()),
                int((dataframe[TARGET_COLUMN] == 1).sum()),
                round(
                    dataframe[TARGET_COLUMN].mean() * 100,
                    2,
                ),
                round(dataframe["age"].mean(), 2),
                round(dataframe["chol"].mean(), 2),
                round(dataframe["trestbps"].mean(), 2),
            ],
        }
    )

    return summary_df


# ============================================================
# SAUVEGARDE
# ============================================================

def save_gold_data(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """Enregistre la table Gold en Parquet."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_parquet(
        output_file,
        index=False,
        engine="pyarrow",
    )

    logging.info(
        "Couche Gold enregistrée : %s (%d lignes)",
        output_file,
        len(dataframe),
    )


def save_summary(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """Enregistre le résumé du dataset au format CSV."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(
        output_file,
        index=False,
    )

    logging.info("Résumé Gold enregistré : %s", output_file)


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main() -> None:
    configure_logging()

    try:
        silver_df = load_silver_data(INPUT_FILE)

        validate_gold_columns(silver_df)

        gold_df = create_gold_features(silver_df)

        validate_gold_data(gold_df)

        summary_df = create_summary(gold_df)

        save_gold_data(
            dataframe=gold_df,
            output_file=OUTPUT_FILE,
        )

        save_summary(
            dataframe=summary_df,
            output_file=SUMMARY_FILE,
        )

        logging.info("Aperçu de la couche Gold :")
        print(gold_df.head())

        logging.info("Résumé du dataset :")
        print(summary_df)

        logging.info(
            "Pipeline Gold terminé avec succès : %d lignes et %d colonnes",
            gold_df.shape[0],
            gold_df.shape[1],
        )

    except Exception as error:
        logging.exception("Échec du pipeline Gold : %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()