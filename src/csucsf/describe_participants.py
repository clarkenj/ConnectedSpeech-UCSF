import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from scipy import stats

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

    ####################
    # Compute Group Comparison Statistics (ANOVA / Chi-Square)
    ####################
    stat_results = []

    # 1. Categorical: Sex (Chi-Square)
    contingency_sex = pd.crosstab(df["Diagnosis"], df["Sex"])
    chi2_sex, p_sex, dof_sex, _ = stats.chi2_contingency(contingency_sex)
    stat_results.append({
        "Variable": "Sex (Female proportion)",
        "Test": "Chi-Square (Chi2)",
        "Statistic": chi2_sex,
        "df": dof_sex,
        "N": int(contingency_sex.values.sum()),
        "p_value": p_sex,
    })

    # 2. Continuous variables: One-Way ANOVA & Kruskal-Wallis
    test_vars = {
        "Age": "Age",
        "Education": "Education",
        "MMSE": "MMSE",
        "CDR Sum of Boxes": "CDR Sum of Boxes",
    }

    for label, col in test_vars.items():
        sub_df = df[["Diagnosis", col]].dropna()
        groups = [group[col].values for _, group in sub_df.groupby("Diagnosis")]
        n = len(sub_df)
        k = len(groups)

        # One-Way ANOVA
        f_val, p_anova = stats.f_oneway(*groups)
        # Non-parametric alternative (Kruskal-Wallis)
        h_val, p_kw = stats.kruskal(*groups)

        stat_results.append({
            "Variable": label,
            "Test": "One-Way ANOVA (F)",
            "Statistic": f_val,
            "df": f"{k - 1}, {n - k}",
            "N": n,
            "p_value": p_anova,
        })
        stat_results.append({
            "Variable": label,
            "Test": "Kruskal-Wallis (H)",
            "Statistic": h_val,
            "df": k - 1,
            "N": n,
            "p_value": p_kw,
        })

    df_stats = pd.DataFrame(stat_results)
    df_stats.to_csv(
        results_dir / "group_comparison_stats.tsv", sep="\t", index=False
    )

    print(df[["Age", "Education", "MMSE", "CDR Sum of Boxes"]].isna().groupby(df["Diagnosis"]).sum())
    
    ####################
    # Post hoc pairwise comparisons
    ####################
    posthoc_results = []

    # Age: Tukey HSD (parametric; group SDs are similar)
    sub = df[["Diagnosis", "Age"]].dropna()
    tukey = pairwise_tukeyhsd(endog=sub["Age"], groups=sub["Diagnosis"], alpha=0.05)
    tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    for _, row in tukey_df.iterrows():
        posthoc_results.append({
            "Variable": "Age",
            "Test": "Tukey HSD",
            "Group 1": row["group1"],
            "Group 2": row["group2"],
            "Mean diff (Group 2 - Group 1)": row["meandiff"],
            "p_adj": row["p-adj"],
            "N": len(sub),
        })

    # MMSE and CDR Sum of Boxes: pairwise Mann-Whitney U, Benjamini-Hochberg corrected
    # (non-parametric, because HC have no variance in CDR and Kruskal-Wallis was used)
    for col in ["MMSE", "CDR Sum of Boxes"]:
        sub = df[["Diagnosis", col]].dropna()
        pairs = list(combinations(sorted(sub["Diagnosis"].unique()), 2))
        raw_p, rows = [], []
        for g1, g2 in pairs:
            a = sub.loc[sub["Diagnosis"] == g1, col]
            b = sub.loc[sub["Diagnosis"] == g2, col]
            u_val, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
            raw_p.append(p_val)
            rows.append((g1, g2, len(a), len(b), u_val))
        p_adj = multipletests(raw_p, method="fdr_bh")[1]
        for (g1, g2, n1, n2, u_val), p_raw, p_corr in zip(rows, raw_p, p_adj):
            posthoc_results.append({
                "Variable": col,
                "Test": "Mann-Whitney U (BH-corrected)",
                "Group 1": g1,
                "Group 2": g2,
                "U": u_val,
                "p_raw": p_raw,
                "p_adj": p_corr,
                "N": f"{n1} vs {n2}",
            })

    df_posthoc = pd.DataFrame(posthoc_results)
    df_posthoc.to_csv(results_dir / "group_comparison_posthoc.tsv", sep="\t", index=False)
    print(df_posthoc.to_string())

    # Generate distribution plots
    plt.style.use('default')
    sns.set_palette("colorblind")

    for col in ["Age", "Education", "MMSE", "CDR Total", "FTLD-CDR Total", "CDR Sum of Boxes","FTLD-CDR Sum of Boxes"] :
        plot_continuous_dist(df, col, results_dir)

    for col in ["Sex", "Handedness"]:
        print(df[col].unique())
        plot_categorical_dist(df, col, results_dir)


