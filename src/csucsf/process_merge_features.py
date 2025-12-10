import pandas as pd
from csucsf.features import DOMAIN_FEATURES, PSYCHOLINGUISTIC_FEATURES
from csucsf.config import DEFAULT_DATASET_DIR, DEFAULT_SPEECHMETRYFLOW_DIR, DEFAULT_DATABASE_PATH

def calculate_proportion(df):

    """Calculate relative frequencies of some features, dividing by the "Nombre_de_mots" column.
    New columns are named <column_name>_relatif."""

    cols_to_divide = [
        "Nombre_de_pauses_remplies",
        "Nombre_de_repetitions_mots",
        "Nombre_de_fragments",
        "Nombre_de_pronoms_deictiques_spatiaux",
        "Nombre_de_pronoms_deictiques_personnels",
        "UCSF_disfluency_filled_pauses",
        "UCSF_disfluency_multiple_repetitions",
        "UCSF_disfluency_partial_word_false_start",
        "UCSF_disfluency_repeated_phrase",
        "UCSF_disfluency_restart_rephrase",
        "UCSF_disfluency_silent_pauses",
        "UCSF_disfluency_single_repetition",
        "UCSF_disfluency_spoonerism",
    ]

    for col in cols_to_divide:
        df.loc[:, f"{col}_relatif"] = df[col] / df["Nombre_de_mots"]

    return df


def select_features(df, pheno_df, ucsf_disfluencies=False):
    """Select certain linguistic features for project (and retain pheno columns).
    Optionally include disfluencies that were manually added to some samples by UCSF team (default=False)"""

    # Get columns from pheno_df, to retain
    pheno_cols = [col for col in df.columns if col in pheno_df.columns]

    # Linguistic features for current project
    linguistic_cols = DOMAIN_FEATURES + PSYCHOLINGUISTIC_FEATURES

    # Optionally include disfluencies
    if ucsf_disfluencies:
        linguistic_cols += [col for col in df.columns if col.startswith("UCSF_disfluency")]

    # Columns to retain
    cols_to_retain = ["transcript_type"] + pheno_cols + linguistic_cols

    # Filter df
    filtered_df = df[cols_to_retain].copy()

    # Add '_L' suffix to all linguistic columns, for easier selection later
    rename_dict = {col: f"{col}_L" for col in linguistic_cols}
    filtered_df.rename(columns=rename_dict, inplace=True)

    return filtered_df

def process_features(df):
    df = df.copy()

    # Calculate some features as a proprtion of total words
    df = calculate_proportion(df)

    # Recode ICU True/False to 1/0
    icu_columns = [col for col in df.columns if col.startswith("ICU ")]
    df[icu_columns] = df[icu_columns].astype(int)

    # Drop counts and absolute values (which have correpsonding relative values)
    df = df.loc[
        :, ~(df.columns.str.endswith("_count") | df.columns.str.endswith("_absolu"))
    ]

    # Drop columns we don't need
    #df = df.drop(columns=["Langue", "SpaCy_Model"])

    # One-hot encode non-numeric columns
    df = pd.get_dummies(df, columns=["Emotion", "Sentiment-valence"], drop_first=False, dtype=int)

    # Fill NaNs with zeros. I think this is fine since they are mostly depth and percentage features
    df = df.fillna(0)

    return df

def main():
    # Load data
    pheno_df = pd.read_csv(DEFAULT_DATASET_DIR / "participants.tsv",sep="\t", index_col="participant_id")
    text_df = pd.read_csv(DEFAULT_SPEECHMETRYFLOW_DIR / "Statistics/LINGUALAB_TEXT/population_lingualab_text.csv", index_col="participant_id")

    # Extract transcript type
    text_df["transcript_type"] = text_df["filename"].str.extract(r"transcript-(.*?)-ver")

    # Do some preprocessing of speech features (see function for details)
    processed_df = process_features(text_df)

    # Merge phenotype data with linguistic features
    merged_df = pheno_df.merge(processed_df, left_index=True, right_index=True)

    # Output data
    merged_df.to_csv(DEFAULT_DATABASE_PATH, index=True)
    print(f"Merged data and saved to {DEFAULT_SPEECHMETRYFLOW_DIR}")
