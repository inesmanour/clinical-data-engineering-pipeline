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
    / "bronze"
    / "heart_bronze.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "silver"

OUTPUT_FILE = OUTPUT_DIR / "heart_silver.parquet"
REJECTED_FILE = OUTPUT_DIR / "heart_rejected.parquet"


# ============================================================
# CONFIGURATION DU DATASET
# ============================================================

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

INTEGER_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "slope",
    "ca",
    "thal",
    "target",
]

FLOAT_COLUMNS = [
    "oldpeak",
]


# Valeurs autorisées pour les variables catégorielles encodées
ALLOWED_VALUES = {
    "sex": {0, 1},
    "cp": {0, 1, 2, 3},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {0, 1, 2},
    "ca": {0, 1, 2, 3, 4},
    "thal": {0, 1, 2, 3},
    "target": {0, 1},
}


# Plages simples utilisées pour le contrôle qualité
NUMERIC_RANGES = {
    "age": (18, 100),
    "trestbps": (50, 250),
    "chol": (50, 700),
    "thalach": (40, 250),
    "oldpeak": (0, 10),
}


# ============================================================
# LOGGING
# ============================================================

def configure_logging() -> None:
    """Configure les messages affichés dans le terminal."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ============================================================
# LECTURE DE LA COUCHE BRONZE
# ============================================================

def load_bronze_data(file_path: Path) -> pd.DataFrame:
    """Lit les données provenant de la couche Bronze."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Fichier Bronze introuvable : {file_path}\n"
            "Lance d'abord le script src/bronze.py."
        )

    logging.info("Lecture de la couche Bronze : %s", file_path)

    dataframe = pd.read_parquet(file_path)

    if dataframe.empty:
        raise ValueError("La couche Bronze ne contient aucune ligne.")

    logging.info(
        "Couche Bronze chargée : %d lignes et %d colonnes",
        dataframe.shape[0],
        dataframe.shape[1],
    )

    return dataframe


# ============================================================
# VALIDATION DU SCHÉMA
# ============================================================

def validate_schema(dataframe: pd.DataFrame) -> None:
    """Vérifie que toutes les colonnes cliniques attendues existent."""

    missing_columns = [
        column
        for column in CLINICAL_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes obligatoires absentes : "
            f"{missing_columns}"
        )

    logging.info("Schéma validé : toutes les colonnes attendues sont présentes.")


# ============================================================
# NETTOYAGE DES NOMS DE COLONNES
# ============================================================

def clean_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Uniformise les noms de colonnes."""

    cleaned_df = dataframe.copy()

    cleaned_df.columns = [
        column.strip().lower().replace(" ", "_")
        for column in cleaned_df.columns
    ]

    return cleaned_df


# ============================================================
# CONVERSION DES TYPES
# ============================================================

def convert_data_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les variables cliniques en valeurs numériques.

    errors='coerce' transforme les valeurs impossibles à convertir en NaN.
    """

    cleaned_df = dataframe.copy()

    for column in INTEGER_COLUMNS:
        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column],
            errors="coerce",
        ).astype("Int64")

    for column in FLOAT_COLUMNS:
        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column],
            errors="coerce",
        ).astype("Float64")

    return cleaned_df


# ============================================================
# SUPPRESSION DES DOUBLONS
# ============================================================

def remove_duplicates(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Supprime les doublons en comparant uniquement les colonnes cliniques.

    Les métadonnées Bronze ne sont pas prises en compte, car deux lignes
    identiques peuvent avoir des numéros de ligne source différents.
    """

    duplicated_mask = dataframe.duplicated(
        subset=CLINICAL_COLUMNS,
        keep="first",
    )

    duplicate_count = int(duplicated_mask.sum())

    cleaned_df = dataframe.loc[~duplicated_mask].copy()

    logging.info("Doublons détectés et supprimés : %d", duplicate_count)

    return cleaned_df, duplicate_count


# ============================================================
# CONTRÔLES QUALITÉ
# ============================================================

def add_quality_rules(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une raison de rejet à chaque ligne invalide.

    Une ligne valide possède une chaîne vide dans _rejection_reason.
    """

    checked_df = dataframe.copy()

    checked_df["_rejection_reason"] = ""

    # Valeurs manquantes
    for column in CLINICAL_COLUMNS:
        missing_mask = checked_df[column].isna()

        checked_df.loc[
            missing_mask,
            "_rejection_reason",
        ] += f"missing_{column};"

    # Variables catégorielles
    for column, allowed_values in ALLOWED_VALUES.items():
        invalid_mask = (
            checked_df[column].notna()
            & ~checked_df[column].isin(allowed_values)
        )

        checked_df.loc[
            invalid_mask,
            "_rejection_reason",
        ] += f"invalid_{column};"

    # Variables numériques
    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        invalid_mask = (
            checked_df[column].notna()
            & ~checked_df[column].between(minimum, maximum)
        )

        checked_df.loc[
            invalid_mask,
            "_rejection_reason",
        ] += f"out_of_range_{column};"

    return checked_df


# ============================================================
# SÉPARATION VALIDES / REJETÉES
# ============================================================

def split_valid_and_rejected(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare les données valides des données rejetées."""

    valid_mask = dataframe["_rejection_reason"].eq("")

    valid_df = dataframe.loc[valid_mask].copy()
    rejected_df = dataframe.loc[~valid_mask].copy()

    valid_df = valid_df.drop(columns="_rejection_reason")

    return valid_df, rejected_df


# ============================================================
# MÉTADONNÉES SILVER
# ============================================================

def add_silver_metadata(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la date de traitement Silver."""

    silver_df = dataframe.copy()

    silver_df["_silver_processed_timestamp_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    return silver_df


# ============================================================
# SAUVEGARDE
# ============================================================

def save_parquet(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """Enregistre un DataFrame au format Parquet."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_parquet(
        output_file,
        index=False,
        engine="pyarrow",
    )

    logging.info(
        "Fichier enregistré : %s (%d lignes)",
        output_file,
        len(dataframe),
    )


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main() -> None:
    configure_logging()

    try:
        bronze_df = load_bronze_data(INPUT_FILE)

        bronze_df = clean_column_names(bronze_df)

        validate_schema(bronze_df)

        typed_df = convert_data_types(bronze_df)

        missing_values = typed_df[
            CLINICAL_COLUMNS
        ].isna().sum()

        logging.info(
            "Nombre total de valeurs manquantes : %d",
            int(missing_values.sum()),
        )

        deduplicated_df, duplicate_count = remove_duplicates(
            typed_df
        )

        checked_df = add_quality_rules(deduplicated_df)

        valid_df, rejected_df = split_valid_and_rejected(
            checked_df
        )

        silver_df = add_silver_metadata(valid_df)

        save_parquet(
            dataframe=silver_df,
            output_file=OUTPUT_FILE,
        )

        # On crée le fichier rejeté seulement s'il existe des lignes invalides
        if not rejected_df.empty:
            save_parquet(
                dataframe=rejected_df,
                output_file=REJECTED_FILE,
            )
        else:
            logging.info("Aucune ligne rejetée.")

        logging.info(
            "Résumé Silver | Bronze : %d | Doublons : %d | "
            "Valides : %d | Rejetées : %d",
            len(bronze_df),
            duplicate_count,
            len(silver_df),
            len(rejected_df),
        )

        logging.info("Aperçu de la couche Silver :")
        print(silver_df.head())

        logging.info("Types des colonnes :")
        print(silver_df.dtypes)

        logging.info("Pipeline Silver terminé avec succès.")

    except Exception as error:
        logging.exception("Échec du pipeline Silver : %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()