import copy
import datetime
import jiwer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy import stats
from statannotations.Annotator import Annotator
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from csucsf.config import DEFAULT_DATABASE_PATH, DEFAULT_DERIVATIVES_DIR
from csucsf.util import read_transcript, show_comparison, create_html_from_text_files


def load_data(participant_id: str):
    """
    Load transcript data for a specific participant across all transcript types.
    
    This function retrieves transcript files and their content for a given participant
    across three transcript types: manual, whisper, and whisperQC. It organizes the 
    data into a dictionary containing the participant ID, filenames, and content for
    each transcript type.
    
    Parameters:
        participant_id (str): Unique identifier for the participant whose data to load
        
    Returns:
        dict: A dictionary containing:
            - "participant_id" (str): The participant ID
            - "filename_manual" (str): Name of the manual transcript file
            - "manual" (str): Content of the manual transcript
            - "filename_whisper" (str): Name of the whisper transcript file
            - "whisper" (str): Content of the whisper transcript
            - "filename_whisperQC" (str): Name of the whisperQC transcript file
            - "whisperQC" (str): Content of the whisperQC transcript
            
    Raises:
        Any exceptions raised by read_transcript() function
    """
    results = {}
    results["participant_id"] = participant_id

    # Process each transcript type
    for transcript in ["manual", "whisper", "whisperQC"]:
        filename, content = read_transcript(participant_id, transcript_type=transcript)
        results[f"filename_{transcript}"] = filename.name
        results[transcript] = content

    return results


def compute_jiwer(manual, whisper, whisperQC, metric=jiwer.wer):
    """
    Compute Word Error Rates (WER) between different transcript types.
    
    This function calculates WER scores comparing manual transcripts against both raw and 
    quality-controlled Whisper transcripts.
    
    Parameters:
        manual (str or list(str)): Reference transcript (manual transcription)
        whisper (str or list(str)): Hypothesis transcript (raw Whisper transcription)
        whisperQC (str or list(str)): Hypothesis transcript (quality-controlled Whisper transcription)
        
    Returns:
        dict: A dictionary containing four WER scores:
            - "manual vs. whisper"
            - "manual vs. whisperQC"
            
    Notes:
        - WER (Word Error Rate) measures the difference between reference and hypothesis texts
        - Lower WER values indicate better transcription accuracy
    """
    wers = {}

    wers["manual vs. whisper"] = metric(
        reference=manual,
        hypothesis=whisper,
        reference_transform=jiwer.wer_standardize,
        hypothesis_transform=jiwer.wer_standardize,
    )

    wers["manual vs. whisperQC"] = metric(
        reference=manual,
        hypothesis=whisperQC,
        reference_transform=jiwer.wer_standardize,
        hypothesis_transform=jiwer.wer_standardize,
    )

    return wers


def main():
    METRIC = jiwer.wer
    METRIC_NAME = "WER"
    ####################
    # Define result directory
    ####################
    date = datetime.date.today().isoformat()
    results_dir = DEFAULT_DERIVATIVES_DIR / f"{METRIC_NAME}_{date}"
    results_dir.mkdir(parents=True, exist_ok=True)

    ####################
    # Load UCSF data
    ####################
    df_ucsf = pd.read_csv(DEFAULT_DATABASE_PATH)
    df_ucsf = df_ucsf.drop_duplicates(subset=['participant_id'])
    # print(list(df_ucsf.shape))

    ####################
    # Compute WER for each participant
    ####################
    results_wer = []
    for participant_id in df_ucsf["participant_id"]:
        transcripts = load_data(participant_id)
        transcripts.update(compute_jiwer(
            transcripts["manual"],
            transcripts["whisper"],
            transcripts["whisperQC"],
            METRIC,
            ))
        results_wer.append(transcripts)
    
    df_wers = pd.merge(df_ucsf, pd.DataFrame.from_dict(results_wer), on="participant_id")

    ####################
    # Compute General WER
    ####################
    general_wers = {}
    general_wers["all"] = compute_jiwer(
        df_wers["manual"].to_list(),
        df_wers["whisper"].to_list(),
        df_wers["whisperQC"].to_list(),
        METRIC,
    )

    for _, sub_df in df_wers.groupby("diagnosis"):
        sub_diagnosis = sub_df["diagnosis"].unique()[0]
        general_wers[sub_diagnosis] = compute_jiwer(
            sub_df["manual"].to_list(),
            sub_df["whisper"].to_list(),
            sub_df["whisperQC"].to_list(),
            METRIC,
        )

    df_general_wers = pd.DataFrame.from_dict(general_wers)
    df_general_wers.to_csv(results_dir / f"general_{METRIC_NAME}.tsv", sep="\t")

    ####################
    # Save WER in a melted format
    ####################
    wers_types = df_general_wers.index
    df_wers_records = df_wers.melt(
        value_vars=wers_types,
        id_vars=["participant_id", "diagnosis", "cdr_sum", "cdr_age"],
    )
    df_wers_records.to_csv(results_dir / f"participants_{METRIC_NAME}.tsv", sep="\t", index=False)
    # print(df_wers_records)

    ####################
    # Normality test and find outliers with IQR
    ####################
    def find_outliers_IQR(df):
        q1=df.quantile(0.25)
        q3=df.quantile(0.75)
        IQR=q3-q1
        outliers = (df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR))
        return outliers
    
    dfs_outliers = []
    for wer_type in df_wers_records["variable"].unique():
        df_temp = df_wers_records[df_wers_records["variable"] == wer_type]
        stat, p = stats.normaltest(df_temp["value"])
        print(f"Statistics={stat:.3f}, p={p:.3f}, {wer_type}")
        df_outliers = df_temp[find_outliers_IQR(df_temp["value"])]
        dfs_outliers.append(df_outliers)
    df_outliers = pd.concat(dfs_outliers)
    df_outliers.to_csv(results_dir / f"outliers_{METRIC_NAME}.tsv", sep="\t", index=False)

    df_outliers.groupby(["variable", "diagnosis"]).describe().to_csv(results_dir / f"outliers_{METRIC_NAME}_describe.tsv", sep="\t")

    ####################
    # Studying outliers
    ####################
    diff_paths = []
    for pid in df_outliers["participant_id"].unique():
        df_pid = df_wers[df_wers["participant_id"] == pid]
        manual_txt = df_pid.iloc[0]["manual"]
        whisper_txt = df_pid.iloc[0]["whisper"]
        whisperQC_txt = df_pid.iloc[0]["whisperQC"]
        whisper_wer = df_pid.iloc[0]["manual vs. whisper"]
        whisperQC_wer = df_pid.iloc[0]["manual vs. whisperQC"]

        diff_output_whisper = results_dir / f"{METRIC_NAME}-{whisper_wer}_{pid}_whisper.txt"
        diff_output_whisperQC = results_dir / f"{METRIC_NAME}-{whisperQC_wer}_{pid}_whisperQC.txt"
        show_comparison(manual_txt, whisper_txt, diff_output_whisper)
        show_comparison(manual_txt, whisperQC_txt, diff_output_whisperQC)

        diff_paths += [diff_output_whisper, diff_output_whisperQC]

    create_html_from_text_files(diff_paths, results_dir / f"outliers_{METRIC_NAME}.html")
    [_.unlink() for _ in diff_paths]

    ####################
    # manual vs. whisper plot
    ####################
    def plot_wer_manual_whisper(variables, tag):
        df_manual_whisper = df_wers_records[df_wers_records["variable"].isin(variables.keys())]
        df_manual_whisper = df_manual_whisper.replace(variables)

        ax = sns.boxplot(
            data=df_manual_whisper,
            x="variable",
            y="value",
        )
        ax.set(
            xlabel="",
            ylabel=METRIC_NAME,
        )
        plt.savefig(results_dir / f'{METRIC_NAME}_{tag}_with_outliers.png', dpi=300, bbox_inches='tight')
        plt.close()

        # removing outliers
        outliers = df_outliers[df_outliers["variable"].isin(variables.keys())]["participant_id"]
        df_manual_whisper = df_manual_whisper[~df_manual_whisper["participant_id"].isin(outliers)]

        ax = sns.boxplot(
            data=df_manual_whisper,
            x="variable",
            y="value",
        )
        ax.set(
            xlabel="",
            ylabel=METRIC_NAME,
        )
        plt.savefig(results_dir / f'{METRIC_NAME}_{tag}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # diagnosis
        md = ols("value ~ diagnosis + variable + diagnosis:variable", data=df_manual_whisper).fit()
        table = sm.stats.anova_lm(md, typ=2)
        table.to_csv(results_dir / f'{METRIC_NAME}_{tag}_diagnosis_anova.tsv', sep="\t")

        tukey_oneway = pairwise_tukeyhsd(endog=df_manual_whisper["value"], groups=df_manual_whisper["variable"])
        with (results_dir / f'{METRIC_NAME}_{tag}_diagnosis_transcript.csv').open("w") as summary_file:
            summary_file.write(tukey_oneway.summary().as_csv())

        tukey_oneway = pairwise_tukeyhsd(endog=df_manual_whisper["value"], groups=df_manual_whisper["diagnosis"])
        with (results_dir / f'{METRIC_NAME}_{tag}_diagnosis_diagnosis.csv').open("w") as summary_file:
            summary_file.write(tukey_oneway.summary().as_csv())
        
        ax = sns.boxplot(
            data=df_manual_whisper,
            x="diagnosis",
            y="value",
            hue="variable",
            order=["HC", "svPPA", "lvPPA", "nfvPPA"]
        )
        ylim = ax.get_ylim()
        new_ylim = (0, ylim[1])
        ax.set(
            xlabel="diagnosis",
            ylabel=METRIC_NAME,
            ylim=new_ylim,
            # title="Manual vs. Whisper",
        )
        ax.get_legend().set_title("")

        # pairs=[
        #     (("svPPA", "manual vs. whisper"), ("HC", "manual vs. whisper")),
        #     (("lvPPA", "manual vs. whisper"), ("HC", "manual vs. whisper")),
        #     (("nfvPPA", "manual vs. whisper"), ("HC", "manual vs. whisper")),
        #     (("lvPPA", "manual vs. whisper"), ("svPPA", "manual vs. whisper")),
        #     (("nfvPPA", "manual vs. whisper"), ("svPPA", "manual vs. whisper")),
        # ]
        # annotator = Annotator(
        #     ax,
        #     pairs,
        #     data=df_manual_whisper,
        #     x="diagnosis",
        #     y="value",
        #     hue="variable",
        # )
        # annotator.configure(
        #     test='Mann-Whitney',
        #     loc='outside',
        # )
        # annotator.apply_and_annotate()

        plt.savefig(results_dir / f'{METRIC_NAME}_{tag}_diagnosis.png', dpi=300, bbox_inches='tight')
        plt.close()

    variables_type = {
        "plot": {
            "manual vs. whisper": "whisper",
            "manual vs. whisperQC": "whisperQC",
        },
    }
    for tag, variables in variables_type.items():
        plot_wer_manual_whisper(variables, tag)

    ####################
    # Regression between WER and CDR
    ####################
    for wer_type in df_wers_records["variable"].unique():
        df_graph = df_wers_records[df_wers_records["variable"] == wer_type].dropna()
        outliers = df_outliers[df_outliers["variable"] == wer_type]["participant_id"]
        df_graph = df_graph[~df_graph["participant_id"].isin(outliers)]
        # wer_type_clean = " ".join(wer_type.split(" ")[:3])
        wer_type_clean = wer_type.split(" ")[-1]
        df_graph = df_graph.replace({wer_type: wer_type_clean})
        df_graph = df_graph[df_graph["diagnosis"].isin(["svPPA", "lvPPA", "nfvPPA"])].sort_values("diagnosis")

        pearson_results = []
        for diagnosis in ["svPPA", "lvPPA", "nfvPPA"]:
            data = df_graph[df_graph["diagnosis"] == diagnosis]
            res = stats.pearsonr(data["cdr_sum"], data["value"])
            res.type = diagnosis
            pearson_results.append(res)

        g = sns.lmplot(
            data=df_graph,
            x="value",
            y="cdr_sum",
            hue="diagnosis",
        )

        g.set(
            xlabel=METRIC_NAME,
            ylabel="CDR Sum",
            title=wer_type_clean,
        )

        ax = g.axes[0][0]
        ax.set(
            xlim=(0, 0.63),
            ylim=(-0.4, 12),
        )

        plt.savefig(results_dir / f'regression_{METRIC_NAME}_CDR_{wer_type.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

        df_pearson = pd.DataFrame({
            'type': [res.type for res in pearson_results],
            'statistic': [res.statistic for res in pearson_results],
            'pvalue': [res.pvalue for res in pearson_results],
        })
        df_pearson.to_csv(results_dir / f'pearson_{METRIC_NAME}_CDR_{wer_type.replace(" ", "_")}.tsv', sep="\t", index=False)

    ####################
    # Extra analysis
    ####################
    # Load data
    all_wer = df_wers_records
    outliers = df_outliers

    # Filter for transcription type
    transcription_type = "manual vs. whisperQC" # manual vs. whisperQC
    all_wer = all_wer[all_wer['variable'] == transcription_type]
    outliers = outliers[outliers['variable'] == transcription_type]

    print(f"There are {len(all_wer)} participants")
    print(f"There are {len(outliers)} outliers for the {transcription_type}")

    # Print means & sds per group to add to paper
    group_stats = all_wer.groupby("diagnosis")["value"].agg(["mean", "std"])
    print(group_stats.round(2))

    # Compare age and CDR for outliers and non-outliers
    # Identify outlier and non-outlier participant IDs
    outlier_ids = outliers.index
    non_outlier_ids = all_wer.index.difference(outlier_ids)

    print(f"There are {len(non_outlier_ids)} non-outliers")

    # Create separate dataframes
    outlier_df = all_wer.loc[outlier_ids]
    non_outlier_df = all_wer.loc[non_outlier_ids]

    def run_demographic_stats(outlier_df, non_outlier_df, col_variable):
        # Drop NaNs
        outlier_df = outlier_df.dropna(subset=[col_variable])
        non_outlier_df = non_outlier_df.dropna(subset=[col_variable])

        # Compute mean ± SD for variable
        df_stats = pd.DataFrame({
            "outliers": [outlier_df[col_variable].mean(), outlier_df[col_variable].std()],
            "non_outliers": [non_outlier_df[col_variable].mean(), non_outlier_df[col_variable].std()]
        }, index=["mean", "std"])

        print(f"{col_variable} stats:\n", df_stats)

        # Run independent t-tests
        t, p = stats.ttest_ind(outlier_df[col_variable], non_outlier_df[col_variable], equal_var=False)

        print(f"\n{col_variable}: t = {t:.2f}, p = {p:.3f}")

    run_demographic_stats(outlier_df, non_outlier_df, "cdr_age")
    run_demographic_stats(outlier_df, non_outlier_df, "cdr_sum")
