import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from csucsf.config import DEFAULT_DERIVATIVES_DIR

def create_summary_df(cv_df, boot_df, metrics):
    """
    Create a summary table combining CV mean±SD and bootstrap CI.
    """
    rows = []
    for task in cv_df['task'].unique():
        for method in cv_df['method'].unique():
            cv_subset = cv_df[(cv_df['task'] == task) & (cv_df['method'] == method)]
            boot_subset = boot_df[(boot_df['task'] == task) & (boot_df['method'] == method)]

            row = {'Task': task, 'Transcript type': method}
            for metric in metrics:
                # CV mean ± SD
                cv_values = cv_subset[metric].dropna()
                mean_cv = cv_values.mean()
                std_cv = cv_values.std()

                # Bootstrap CI
                boot_values = boot_subset[metric].dropna()
                ci_lower = np.percentile(boot_values, 2.5)
                ci_upper = np.percentile(boot_values, 97.5)

                row[metric] = f"{mean_cv:.2f} ± {std_cv:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"

            rows.append(row)
    return pd.DataFrame(rows)

def pairwise_stats(cv_df, metrics, alpha=0.05):
    """
    Perform Wilcoxon signed-rank tests between methods for each task and ICC condition.
    Applies FDR correction across all tests.
    """
    results = []    
    methods_to_compare = cv_df["method"].unique()

    for task in cv_df["task"].unique():
        for icc_status in cv_df["icc_filtered"].unique():
            task_df = cv_df[(cv_df["task"] == task) & (cv_df["icc_filtered"] == icc_status)]

            for m1, m2 in itertools.combinations(methods_to_compare, 2):
                for metric in metrics:
                    s1 = task_df[task_df["method"] == m1][metric].dropna()
                    s2 = task_df[task_df["method"] == m2][metric].dropna()

                    # Ensure paired samples
                    n = min(len(s1), len(s2))
                    if n < 2:  # too few folds to compare
                        continue

                    s1, s2 = s1.iloc[:n], s2.iloc[:n]

                    try:
                        stat, p = wilcoxon(s1, s2, alternative="two-sided")
                    except ValueError:
                        continue

                    diff = s1.mean() - s2.mean()
                    results.append({
                        "Task": task,
                        "ICC_filtered": icc_status,
                        "Metric": metric,
                        "Method1": m1,
                        "Method2": m2,
                        "Mean_diff": diff,
                        "p_value": p,
                        "n_folds": n
                    })

    df = pd.DataFrame(results)

    # FDR correction
    reject, p_corrected, _, _ = multipletests(df["p_value"], alpha=alpha, method="fdr_bh")
    df["p_value_corrected"] = p_corrected
    df["sig_uncorrected"] = df["p_value"] < alpha
    df["sig_corrected"] = reject

    return df

def plot_cv_summary(cv_df, figure_p, method_order, metrics=["auc"], icc=False):
    """
    Plot CV results as bars with mean ± SD per task and transcript method.
    """
    df_plot = cv_df.copy()
    if icc:
        df_plot = df_plot[df_plot["icc_filtered"] == True]
    else:
        df_plot = df_plot[df_plot["icc_filtered"] == False]

    for metric in metrics:
        # Compute mean and SD per task and method
        summary = (
            df_plot.groupby(["task", "method"])[metric]
                   .agg(["mean", "std"])
                   .reset_index()
        )

        # Determine x-axis order based on manual method
        task_order = (
            summary[summary["method"] == "salt-edit"]
            .sort_values("mean", ascending=False)["task"]
            .unique()
        )

        plt.figure(figsize=(10, 6))
        
        # Use seaborn's pointplot which automatically handles error bar positioning
        ax = sns.barplot(
            data=summary,
            x="task",
            y="mean",
            hue="method",
            hue_order=method_order,
            order=task_order,
            palette="colorblind",
            ci=None,  # We'll add error bars manually
            alpha=0.7
        )
        
        # Add error bars using pointplot (automatically positioned correctly)
        sns.pointplot(
            data=summary,
            x="task",
            y="mean",
            hue="method",
            hue_order=method_order,
            order=task_order,
            dodge=0.5,  # This matches the barplot dodge
            join=False,
            markers='',
            scale=0.8,
            errwidth=1.5,
            capsize=0.05,
            color='black',
            ax=ax
        )

        plt.ylabel(f"{metric.upper()} (mean ± SD)")
        plt.xlabel("Classification task")
        title_suffix = "ICC filtered" if icc else "No ICC filtering"
        plt.title(f"{title_suffix} {metric.upper()} by transcript method")
        plt.ylim(0.4, 1)
        plt.legend(title="Transcript method")
        plt.tight_layout()

        icc_flag = "_icc" if icc else ""
        plt.savefig(figure_p / f"cv_{metric}{icc_flag}.png", dpi=300)
        plt.close()
            
    print(f"Plots saved to {figure_p}")

def box_swarm(df, method_order, figure_p):
    plt.figure(figsize=(10,6))
    
    # Add shaded regions for alternating tasks
    tasks = df["task"].unique()
    for i, task in enumerate(tasks):
        plt.axvspan(
            i - 0.5, i + 0.5, 
            facecolor="gray", 
            alpha=0.15 if i % 2 == 0 else 0.05,
            zorder=-1
        )

    sns.boxplot(
        data=df, 
        x="task", 
        y="auc", 
        hue="method",
        palette="Set2",
        hue_order=method_order
    )

    # Overlay individual bootstrap points
    sns.stripplot(
        data=df,
        x="task",
        y="auc",
        hue="method",
        dodge=True,
        alpha=0.3,
        jitter=0.2,
        hue_order=method_order,
        palette="Set2",
        legend=False,
    ) 

    plt.title("Bootstrap AUC by classification task and transcript method")
    plt.ylabel("AUC")
    plt.xlabel("Classification task")
    plt.legend(title="Transcript method", loc="lower left")

    plt.savefig(figure_p / "auc_box_swarm.png")
    plt.show()

def box_swarm_icc(df, figure_p):
    """
    Combined box + swarm plot comparing ICC-filtered vs non-filtered
    results for both Whisper-raw and Whisper-edit, per task.
    """

    # Create the figure and axes using subplots
    fig, axes = plt.subplots(1, len(df["task"].unique()), figsize=(12, 6), sharey=True)
    
    # Keep only Whisper methods
    df = df[df["method"].isin(["whisper-raw", "whisper-edit"])].copy()
    order = ["whisper-raw", "whisper-edit"]
    
    # Convert icc to string to ensure consistent handling
    df["icc_filtered"] = df["icc_filtered"].astype(str)
    icc_order = ["True", "False"]
    
    for i, task in enumerate(df["task"].unique()):
        task_data = df[df["task"] == task]
        
        # Create boxplot
        sns.boxplot(
            data=task_data,
            x="method",
            y="auc",
            hue="icc_filtered",
            order=order,
            hue_order=icc_order,
            palette="Set2",
            ax=axes[i]
        )
        
        # Overlay stripplot
        sns.stripplot(
            data=task_data,
            x="method",
            y="auc",
            hue="icc_filtered",
            order=order,
            hue_order=icc_order,
            dodge=True,
            alpha=0.3,
            jitter=0.2,
            palette="Set2",
            legend=False,
            ax=axes[i]
        )

        axes[i].set_title(f"{task}")
        axes[i].set_xlabel("Transcript method")
        axes[i].set_ylabel("AUC" if i == 0 else "")  # Only show y-label for first subplot
    
    # Set overall figure title
    fig.suptitle("Effect of filtering for robust features", fontsize=16)
    
    # Create a unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="ICC filtering", 
               bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # Remove individual subplot legends
    for ax in axes:
        ax.get_legend().remove()
    
    plt.tight_layout()
    plt.savefig(figure_p / "whisper_icc_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()



def main():
    input_p = list(sorted(DEFAULT_DERIVATIVES_DIR.glob("*classification*")))[-1]  # most recent classification folder
    figure_p = input_p / "figures"
    figure_p.mkdir(parents=True, exist_ok=True)

    #classifier = "SVC"
    metrics = ["auc", "balanced_accuracy", "f1", "sensitivity", "specificity"]
    method_order = ["salt-edit", "whisper-raw", "whisper-edit"]

    # Load results
    boot_df = pd.read_csv(input_p / "bootstrap_results.csv")
    cv_df = pd.read_csv(input_p / "cv_results.csv")

    # Compare iccc filtered and unfileterd results
    box_swarm_icc(boot_df, figure_p)

    # generate AUC plot with no ICC filtering
    box_swarm(boot_df[boot_df['icc_filtered'] == False], method_order, figure_p)

    # bar plot of CV results
    plot_cv_summary(cv_df, figure_p, method_order, metrics,  icc=False)
    plot_cv_summary(cv_df, figure_p, method_order, metrics, icc=True)

    # generate summarys in a nice looking table
    no_icc_summary_df = create_summary_df(cv_df[cv_df['icc_filtered'] == False], boot_df[boot_df['icc_filtered'] == False], metrics)
    icc_summary_df = create_summary_df(cv_df[cv_df['icc_filtered'] == True], boot_df[boot_df['icc_filtered'] == True], metrics)

    # output
    no_icc_summary_df.to_excel(input_p / "no_icc_summary.xlsx", index=False)
    icc_summary_df.to_excel(input_p / "icc_summary.xlsx", index=False)

    # generate Wilcoxon pairwise comparisons
    stats_df = pairwise_stats(cv_df, metrics)
    stats_df.to_excel(input_p / "pairwise_wilcoxon.xlsx", index=False)
    

