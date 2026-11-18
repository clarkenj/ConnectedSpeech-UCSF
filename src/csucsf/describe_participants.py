import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math

from pathlib import Path

from csucsf.config import DEFAULT_DERIVATIVES_DIR, DEFAULT_TRANSCRIPTS_DIR

def plot_continuous_dist(df, variable, results_dir):
    figures_dir = results_dir / "demographic_plots"
    figures_dir.mkdir(exist_ok=True)

    diagnoses = df["Diagnosis"].unique()
    n = len(diagnoses)

    # Get overall min and max for axes
    x_min = df[variable].min()
    x_max = df[variable].max()
    y_max = 0
    for diag in diagnoses:
        data = df[df["Diagnosis"] == diag][variable].dropna()
        counts, _ = np.histogram(data, bins=15)
        y_max = max(y_max, counts.max())

    # Create 2x2 grid
    n_rows = 2
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4), sharey=True)

    axes = axes.flatten()

    for i, diag in enumerate(diagnoses):
        sns.histplot(
            data=df[df["Diagnosis"] == diag],
            x=variable,
            bins=15,
            ax=axes[i],
            kde=True,
            color="skyblue"
        )
        axes[i].set_title(diag)
        axes[i].set_xlabel(variable)
        axes[i].set_ylabel("Count")
        axes[i].set_xlim(x_min, x_max)
        axes[i].set_ylim(0, y_max)

    # Hide any unused subplots
    for j in range(i+1, n_rows*n_cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(figures_dir / f"{variable}_by_diagnosis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{variable} plot saved to {figures_dir}")

def plot_categorical_dist(df, variable, results_dir):
    figures_dir = results_dir / "demographic_plots"
    figures_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    counts = df.dropna(subset=[variable]).groupby(["Diagnosis", variable]).size().unstack(fill_value=0)
    counts.plot(kind='bar', stacked=True)
    plt.title(f"{variable} distribution by diagnosis")
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / f"{variable}_by_diagnosis.png", dpi=300, bbox_inches="tight")

    print(f"{variable} plot saved to {figures_dir}")


def main():
    ####################
    # Define result directory
    ####################
    date = datetime.date.today().isoformat()
    results_dir = DEFAULT_DERIVATIVES_DIR / f"describe_participants_{date}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load UCSF data
    df_ucsf = pd.read_excel(
        DEFAULT_TRANSCRIPTS_DIR / "LAVA-Merged_simona_natasha_cases_re-run_06-19-2025_backup.xlsx",
        index_col=0
    )

    df_ucsf = df_ucsf.drop(columns=['Sex'])

    # Rename columns
    column_mapping = {
        "Age at Picnic": "Age",
        "DX": "Diagnosis",
        "Hand (Labeled)": "Handedness",
        "Sex (Labeled)": "Sex",
        "MMSE Total": "MMSE",
    }
    df = df_ucsf.rename(columns=column_mapping)

    df = df[[
        "Diagnosis", "Age", "Sex", "Education", "Handedness", "MMSE",
        "CDR Total", "CDR Sum of Boxes", "FTLD-CDR Total", "FTLD-CDR Sum of Boxes"
    ]]

    # Drop duplicates
    df = df[~df.index.duplicated(keep="first")]

    # Ensure numeric
    numeric_cols = [
        "Age", "Education", "MMSE", "CDR Total", "CDR Sum of Boxes",
        "FTLD-CDR Total", "FTLD-CDR Sum of Boxes"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Summary stats
    summary_table = df.groupby("Diagnosis").agg(
        Age_mean=("Age", "mean"),
        Age_sd=("Age", "std"),
        Age_min=("Age", "min"),
        Age_max=("Age", "max"),
        Education_mean=("Education", "mean"),
        Education_sd=("Education", "std"),
        Education_min=("Education", "min"),
        Education_max=("Education", "max"),
        MMSE_mean=("MMSE", "mean"),
        MMSE_sd=("MMSE", "std"),
        MMSE_min=("MMSE", "min"),
        MMSE_max=("MMSE", "max"),
        CDR_mean=("CDR Sum of Boxes", "mean"),
        CDR_sd=("CDR Sum of Boxes", "std"),
        CDR_min=("CDR Sum of Boxes", "min"),
        CDR_max=("CDR Sum of Boxes", "max")
    ).round(2)

    # Add counts
    counts = df.groupby("Diagnosis").size()
    summary_table = summary_table.join(counts.rename("N"))

    female_counts = df.groupby("Diagnosis")["Sex"].apply(lambda x: (x == "Female").sum())
    female_counts.name = "N Female"

    rh_counts = df.groupby("Diagnosis")["Handedness"].apply(lambda x: (x == "RIGHT").sum())
    rh_counts.name = "N R handed"

    # Merge into summary table
    summary_table = summary_table.join(female_counts)
    summary_table = summary_table.join(rh_counts)

    # Create mean ± sd (range) columns
    for col in ["Age", "Education", "MMSE", "CDR"]:
        summary_table[f"{col}_mean_sd_range"] = (
            summary_table[f"{col}_mean"].astype(str) 
            + " ± " 
            + summary_table[f"{col}_sd"].astype(str) 
            + " (" 
            + summary_table[f"{col}_min"].astype(str) 
            + "–" 
            + summary_table[f"{col}_max"].astype(str) 
            + ")"
        )

    # Save
    summary_table.to_csv(results_dir / "participants_summary.tsv", sep="\t")

    # Generate distribution plots
    plt.style.use('default')
    sns.set_palette("colorblind")

    for col in ["Age", "Education", "MMSE", "CDR Total", "FTLD-CDR Total", "CDR Sum of Boxes","FTLD-CDR Sum of Boxes"] :
        plot_continuous_dist(df, col, results_dir)

    for col in ["Sex", "Handedness"]:
        print(df[col].unique())
        plot_categorical_dist(df, col, results_dir)
