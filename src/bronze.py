from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# Racine du projet : clinical-lakehouse-pipeline/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "heart_disease_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "bronze"
OUTPUT_FILE = OUTPUT_DIR / "heart_bronze.parquet"


def configure_logging() -> None:
    """Configure l'affichage des logs dans le terminal."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def detect_separator(file_path: Path) -> str:
    """
    Détecte simplement si le CSV utilise une virgule ou un point-virgule.
    """
    first_line = file_path.read_text(encoding="utf-8-sig").splitlines()[0]

    if first_line.count(";") > first_line.count(","):
        return ";"

    return ","


def ingest_raw_data(file_path: Path) -> pd.DataFrame:
    """
    Charge les données brutes sans effectuer de nettoyage métier.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {file_path}\n"
            "Vérifie que ton CSV est bien placé dans data/heart.csv."
        )

    separator = detect_separator(file_path)

    logging.info("Lecture du fichier : %s", file_path)
    logging.info("Séparateur détecté : %r", separator)

    dataframe = pd.read_csv(
        file_path,
        sep=separator,
        encoding="utf-8-sig",
    )

    if dataframe.empty:
        raise ValueError("Le fichier CSV ne contient aucune ligne.")

    return dataframe


def add_ingestion_metadata(
    dataframe: pd.DataFrame,
    source_file: Path,
) -> pd.DataFrame:
    """
    Ajoute des métadonnées techniques utiles pour la traçabilité.
    """
    bronze_df = dataframe.copy()

    bronze_df["_source_file"] = source_file.name
    bronze_df["_ingestion_timestamp_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    bronze_df["_source_row_number"] = range(1, len(bronze_df) + 1)

    return bronze_df


def save_bronze_data(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Enregistre la couche Bronze au format Parquet.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_parquet(
        output_file,
        index=False,
        engine="pyarrow",
    )

    logging.info("Couche Bronze enregistrée : %s", output_file)


def main() -> None:
    configure_logging()

    try:
        raw_df = ingest_raw_data(INPUT_FILE)

        logging.info(
            "Données chargées : %d lignes et %d colonnes",
            raw_df.shape[0],
            raw_df.shape[1],
        )

        logging.info("Colonnes détectées : %s", list(raw_df.columns))

        bronze_df = add_ingestion_metadata(
            dataframe=raw_df,
            source_file=INPUT_FILE,
        )

        save_bronze_data(
            dataframe=bronze_df,
            output_file=OUTPUT_FILE,
        )

        logging.info("Aperçu de la couche Bronze :")
        print(bronze_df.head())

        logging.info("Pipeline Bronze terminé avec succès.")

    except Exception as error:
        logging.exception("Échec du pipeline Bronze : %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()