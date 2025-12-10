"""
ICC analysis

- ICC1: each of the subjects has been measured by only a sub-set of the raters, and it is NOT the same sub-set of raters that measured each subject
- ICC2: each of the subjects has been measured by only a sub-set of the raters, but it IS the same sub-set of raters that measured each subject
- ICC3: each of the subjects has been measured by the entire population of raters (so we don't have to take inter-rater variability into account)
- ICC1k, ICC2k and ICC3k: the reliability of the k raters when working as a group (whereas ICC1, ICC2 and ICC3 represent the reliability of the raters as individuals). These values will always be higher (in the example they are ~0.91 as opposed to the first three which are ~0.73) because multiple raters working together will always give a statistically more reliable result.

https://rowannicholls.github.io/python/statistics/agreement/intraclass_correlation.html

We are using ICC3 in this analysis
"""


import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy.stats import chi2_contingency

from csucsf.config import DEFAULT_DATABASE_PATH, DEFAULT_DERIVATIVES_DIR
from csucsf.features import FEATURE_2_DOMAIN


def interpret_icc_cicchetti(icc):
    """
    Interpret the inter-rater agreement.
    
    Cicchetti D. Guidelines, criteria, and rules of thumb for evaluating
    normed and standardized assessment instruments in psychology.
    Psychological Assessment. 1994;6(4):284-290
    """
    if icc < 0.4:
        return 'poor'
    elif icc < 0.6:
        return 'fair'
    elif icc < 0.75:
        return 'good'
    elif icc <= 1:
        return 'excellent'
    else:
        raise ValueError(f'Invalid value for the ICC: {icc}')


def interpret_icc_koo_li(icc):
    """
    Interpret the inter-rater agreement.
    
    Koo T, Li M. A Guideline of Selecting and Reporting Intraclass
    Correlation Coefficients for Reliability Research.
    Journal of Chiropractic Medicine. 2016;15(2):155-163
    """
    if icc < 0.5:
        return 'poor'
    elif icc < 0.75:
        return 'moderate'
    elif icc < 0.9:
        return 'good'
    elif icc <= 1:
        return 'excellent'
    else:
        raise ValueError(f'Invalid value for the ICC: {icc}')
    

def icc_by_group(data, feature, targets="participant_id", raters="transcript"):
    """
    Compute intraclass correlation for all participants and inside each DX groups

    Return `stats` DataFrame from `pingouin.intraclass_corr` with columns added:
    - `group`: "all" or name of the DX group
    - `cicchetti`: Cicchetti interpretation of the icc score
    - `koo_li`: Koo & Li interpretation of the icc score
    
    https://pingouin-stats.org/build/html/generated/pingouin.intraclass_corr.html
    """
    dfs_icc = []
    # Intraclass correlation for all participants and inside each groups define in "diagnosis" column
    for dx in ["all"] + list(data["diagnosis"].unique()):
        if dx == "all":
            df = data
        else:
            df = data[data["diagnosis"] == dx]

        # print(dx, df)
        result_icc = pg.intraclass_corr(data=df, targets=targets, raters=raters, ratings=feature)
        result_icc["group"] = dx
        result_icc["feature"] = feature
        result_icc["domain"] = FEATURE_2_DOMAIN[feature]
        dfs_icc.append(result_icc)

    icc_stat = pd.concat(dfs_icc)
    icc_stat["cicchetti"] = icc_stat["ICC"].apply(interpret_icc_cicchetti)
    icc_stat["koo_li"] = icc_stat["ICC"].apply(interpret_icc_koo_li)
    return icc_stat


def plot_icc_feature(data, feature, iccs, output_dir):
    data_plot = data.pivot_table(values=feature, columns="transcript", index=["participant_id", "diagnosis"])
    y = data_plot.columns[-1]
    fig_joint = sns.jointplot(data=data_plot, x="manual", y=y, hue="diagnosis")
    
    fig_joint.ax_joint.set_title(feature, pad=70)
    
    icc3 = iccs[iccs["Type"] == "ICC3"].drop(columns=["Type", "Description", "df1", "df2"]).set_index("group")
    for col in ["ICC", "F", "pval"]:
        icc3[col] = icc3[col].apply("{:.4g}".format)
    
    pd.plotting.table(
        fig_joint.ax_joint,
        icc3,
        bbox=(-.1, -0.4, 1.3, 0.25)
    )

    rating_name = feature.replace(" ", "_").replace("/", "_over_")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_joint.savefig(output_dir / f"{rating_name}_{y}.png")
    plt.close()


def plot_icc_domain(data, output, title, y="ICC"):
        sns.catplot(
            data=data, kind="bar",
            x="domain", y=y, hue="group",
            # errorbar="sd",
            aspect=20/9,
        )
        if y == "ICC":
            plt.axhline(y=.5, ls='--', c="grey")
            plt.axhline(y=.75, ls='--', c="grey")
            plt.axhline(y=.9, ls='--', c="grey")
            plt.text(-.5, .3, "poor", rotation=90)
            plt.text(-.5, .55, "moderate", rotation=90)
            plt.text(-.5, .78, "good", rotation=90)
            plt.text(-.5, .92, "excellent", rotation=90)
        plt.title(title)
        plt.savefig(output, dpi=300, bbox_inches='tight')
        plt.close()


def main_analysis(data, results_dir, tag):
    ####################
    # ICC computation
    ####################
    ICCs = {}
    for feature in FEATURE_2_DOMAIN.keys():
        ICCs[feature] = icc_by_group(data, feature)

    ####################
    # ICC Plots
    ####################
    for feature, icc_stat in ICCs.items():
        plot_icc_feature(data, feature, icc_stat, results_dir / f"icc_plots_manual_{tag}")

    ####################
    # Koo & Li Agreements
    ####################
    def agreements_color(value):
        cmap = mpl.colormaps["PRGn"]
        value_to_color = {
            "poor": cmap(0.),
            "moderate": cmap(1/4),
            "good": cmap(3/4),
            "excellent": cmap(1.),
        }
        color = mpl.colors.rgb2hex(value_to_color[value])
        return f'background-color: {color}'

    def agreements_to_excel(ICC_stats, output):
        all_stats = []
        for feature, stats in ICC_stats.items():
            stats["feature"] = feature
            all_stats.append(stats)
        
        df_stats = pd.concat(all_stats)
        df_stats = df_stats[df_stats["Type"] == "ICC3"]
        agreements = df_stats.pivot(index="feature", columns="group", values=["koo_li"])
        agreements.columns = agreements.columns.get_level_values(1)

        # Reordering
        agreements = agreements[["all", "HC", "svPPA", "lvPPA", "nfvPPA"]]
        
        agreements.style.map(agreements_color).to_excel(output)
        return agreements
    
    agreements = agreements_to_excel(ICCs, results_dir / f"koo_li_agreements_manual_{tag}.xlsx")
    agreements_to_value = {
        "poor": 0,
        "moderate": 1,
        "good": 2,
        "excellent": 3,
    }
    agreements = agreements.replace(agreements_to_value).infer_objects(copy=False)

    ####################
    # 
    ####################
    df_iccs = pd.concat(ICCs.values())
    df_iccs = df_iccs[df_iccs["Type"] == "ICC3"]

    koo_li_count = df_iccs.groupby(["group", "koo_li", "domain"])["Type"].count().unstack("domain").fillna(0).astype(int)
    koo_li_count.to_excel(results_dir / f"koo_li_categories_all_participant_{tag}.xlsx")

    plot_icc_domain(df_iccs, results_dir / f"icc_domain_plot_{tag}.png", f"ICC manual vs. {tag}")

    return df_iccs


def main():
    ####################
    # Define result directory
    ####################
    date = datetime.date.today().isoformat()
    results_dir = DEFAULT_DERIVATIVES_DIR / f"ICC_{date}"
    results_dir.mkdir(parents=True, exist_ok=True)

    ####################
    # Load UCSF data
    ####################
    transcript_tag = {
        "transcript-salt-edit-ver-1": "manual",
        "transcript-whisper-edit-ver-largev3-v20240930": "whisperQC",
        "transcript-whisper-raw-ver-largev3-v20240930": "whisper",
    }

    df_ucsf = pd.read_csv(DEFAULT_DATABASE_PATH)
    df_ucsf["transcript"] = df_ucsf["filename"].str.split("_").str[-1].replace(transcript_tag)
    df_ucsf = df_ucsf[df_ucsf["transcript"].isin(['manual', 'whisper', 'whisperQC'])]

    df_ucsf_whisper = df_ucsf[df_ucsf["transcript"].isin(["manual", "whisper"])]
    df_ucsf_whisperQC = df_ucsf[df_ucsf["transcript"].isin(["manual", "whisperQC"])]

    df_ICCs_whisper = main_analysis(df_ucsf_whisper, results_dir, "whisper")     
    df_ICCs_whisperQC = main_analysis(df_ucsf_whisperQC, results_dir, "whisperQC")
    
    df_ICCs_whisper["method"] = df_ICCs_whisperQC["ICC"] - df_ICCs_whisper["ICC"]
    
    plot_icc_domain(
        df_ICCs_whisper,
        results_dir / "icc_domain_plot_whisper_whisperQC.png",
        "ICC manual vs. whisper",
        "method",
    )

    ####################
    # Extra analysis
    ####################
    # Load data
    whisper_df = pd.read_excel(
        results_dir / "koo_li_agreements_manual_whisper.xlsx",
        index_col=0
    )
    QC_df = pd.read_excel(
        results_dir / "koo_li_agreements_manual_whisperQC.xlsx",
        index_col=0
    )

    reliable = ["good", "excellent"]
    groups = ["all", "HC" , "svPPA", "lvPPA", "nfvPPA"]

    summary = []
    for group in groups:
        # boolean masks
        whisper_mask = whisper_df[group].isin(reliable)
        QC_mask = QC_df[group].isin(reliable)

        # count reliable / not reliable
        whisper_reliable = whisper_mask.sum()
        whisper_unreliable = len(whisper_mask) - whisper_reliable
        QC_reliable = QC_mask.sum()
        QC_unreliable = len(QC_mask) - QC_reliable

        # percentage reliable
        whisper_percent = 100 * whisper_mask.sum() / len(whisper_mask)
        QC_percent = 100 * QC_mask.sum() / len(QC_mask)

        # 2x2 contingency table
        contingency = [[whisper_reliable, whisper_unreliable],
                       [QC_reliable, QC_unreliable]]

        # chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency)

        summary.append({
            "group": group,
            "whisper_reliable_%": whisper_percent,
            "QC_reliable_%": QC_percent,
            "chi2": round(chi2, 3),
            "p_value": round(p, 3)
        })

    # save summary
    pd.DataFrame(summary).to_csv(results_dir / "icc_whisper_vs_QC_summary.csv", index=False)


