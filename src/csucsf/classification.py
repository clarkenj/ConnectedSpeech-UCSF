import datetime
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix
)

from csucsf.config import DEFAULT_DATASET_DIR, DEFAULT_DERIVATIVES_DIR, DEFAULT_DATABASE_PATH
from csucsf.features import DOMAIN_FEATURES, PSYCHOLINGUISTIC_FEATURES

def compute_metric(y_true, y_pred, y_proba, metric):
    """Compute a single performance metric."""
    if metric == 'auc':
        return roc_auc_score(y_true, y_proba)
    elif metric == 'balanced_accuracy':
        return balanced_accuracy_score(y_true, y_pred)
    elif metric == 'f1':
        return f1_score(y_true, y_pred)
    elif metric == 'sensitivity':
        return recall_score(y_true, y_pred, pos_label=1)
    elif metric == 'specificity':
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def classification_metrics(y_true, y_pred, y_proba):
    """Return multiple classification metrics as a dict."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        'auc': roc_auc_score(y_true, y_proba),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def cv_classifier(df_task, features, methods, diag_map, cv_splitter, model):
    """
    Run CV classification for multiple methods and return metrics per fold,
    plus the out-of-fold predictions for each method.
    """
    all_results = []
    oof_predictions = {}

    for method in methods:
        df_method = df_task[df_task["transcript_type"] == method]
        X = df_method[features]
        y = df_method["diagnosis"].map(diag_map)

        oof_pred = np.zeros(len(y), dtype=int)
        oof_proba = np.zeros(len(y))

        method_results = []
        for train_idx, test_idx in cv_splitter.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            oof_pred[test_idx] = y_pred
            oof_proba[test_idx] = y_proba

            # Collect metrics for this fold
            fold_metrics = classification_metrics(y_test, y_pred, y_proba)
            fold_metrics['method'] = method
            method_results.append(fold_metrics)

        oof_predictions[method] = pd.DataFrame(
            {"y_true": y.to_numpy(), "y_pred": oof_pred, "y_proba": oof_proba},
            index=df_method.index
        )
        all_results.extend(method_results)

    return pd.DataFrame(all_results), oof_predictions

def bootstrap_classifier(oof_predictions, methods, n_bootstrap=1000, random_state=None):
    """
    Bootstrap participants (stratified by class) and recompute the metrics from the
    out-of-fold predictions. The same resamples are used for all methods.
    """
    rng = np.random.default_rng(random_state)
    for m in methods:
        assert oof_predictions[m].index.is_unique

    ids = oof_predictions[methods[0]].index
    oof = {m: oof_predictions[m].loc[ids] for m in methods}
    y = oof[methods[0]]["y_true"].to_numpy()
    for m in methods:
        assert np.array_equal(oof[m]["y_true"].to_numpy(), y)

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    all_results = {m: [] for m in methods}

    for b in range(n_bootstrap):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True)
        ])

        for method in methods:
            o = oof[method].iloc[idx]
            metrics = classification_metrics(o["y_true"], o["y_pred"], o["y_proba"])
            metrics["bootstrap"] = b
            all_results[method].append(metrics)

    return all_results

def run_transcription_ml(df, features, diag_pairs, methods, clf, cv_splitter, filter_by_icc, icc_results=None,
                         n_bootstrap=10, random_state=None, feature_selection="none"):
    """
    Run both CV and bootstrap analysis for all tasks and methods.
    """
    all_cv_results = []
    all_bootstrap_results = []

    for diag1, diag2 in diag_pairs:
        task = f"{diag1}-vs-{diag2}"
        print(f"Running {task}")
        df_task = df[df["diagnosis"].isin([diag1, diag2])]
        diag_map = {diag1: 0, diag2: 1}

        filtered_features = features.copy()
        # Run ICC filtering
        if filter_by_icc:
            if icc_results is None:
                raise ValueError("icc_results must be provided when filter_by_icc=True")

            # Keep only features that are 'excellent' or 'good' for both diagnoses
            mask = icc_results[diag1].isin(["excellent", "good"]) & icc_results[diag2].isin(["excellent", "good"])
            retained_features = icc_results.loc[mask, "feature"].tolist()
            filtered_features = [f for f in filtered_features if f in retained_features]

            print(f"{len(filtered_features)} features retained after ICC filtering.")

        else:
            print(f"Using all {len(filtered_features)} features")

        n_features = len(filtered_features)
        if "selectkbest" in clf.named_steps:
            n_features = min(clf.named_steps["selectkbest"].k, n_features)

        print("Running CV for all methods...")
        cv_results, oof_predictions = cv_classifier(df_task, filtered_features, methods, diag_map, cv_splitter, clf)
        cv_results["task"] = task
        cv_results["icc_filtered"] = filter_by_icc
        cv_results["feature_selection"] = feature_selection
        cv_results["n_features"] = n_features
        all_cv_results.append(cv_results)

        print("Bootstrapping all methods...")
        boot_results = bootstrap_classifier(oof_predictions, methods, n_bootstrap, random_state)

        # Convert to DataFrame
        for method_name, boot_list in boot_results.items():
            df_boot = pd.DataFrame(boot_list)
            df_boot["task"] = task
            df_boot["method"] = method_name
            df_boot["icc_filtered"] = filter_by_icc
            df_boot["feature_selection"] = feature_selection
            df_boot["n_features"] = n_features
            all_bootstrap_results.append(df_boot)

    # Combine all results
    all_cv_df = pd.concat(all_cv_results, ignore_index=True)
    all_boot_df = pd.concat(all_bootstrap_results, ignore_index=True)

    return all_cv_df, all_boot_df

def main():
    ####################
    # Define result directory
    ####################
    date = datetime.date.today().isoformat()
    results_dir = DEFAULT_DERIVATIVES_DIR / f"classification_{date}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(DEFAULT_DATABASE_PATH, index_col=0, low_memory=False,)
    most_recent_ICC_folder = list(sorted(DEFAULT_DERIVATIVES_DIR.glob("*ICC*")))[-1]
    icc_results = pd.read_excel(most_recent_ICC_folder / "koo_li_agreements_manual_whisper.xlsx")

    # -----------------------------
    # Define ML setup
    # -----------------------------
    SEED = 42
    K = 20
    cv_splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    clf = SVC(probability=True, kernel="linear", random_state=SEED, class_weight='balanced')
    model = make_pipeline(StandardScaler(), clf)
    model_k = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=K), clone(clf))

    features = DOMAIN_FEATURES + PSYCHOLINGUISTIC_FEATURES
    metrics = ["auc", "balanced_accuracy", "f1", "sensitivity", "specificity"]
    methods = ["salt-edit", "whisper-raw", "whisper-edit"]
    diag_pairs = [
        ("HC", "svPPA"),
        ("HC", "lvPPA"),
        ("HC", "nfvPPA"),
        ("lvPPA", "svPPA")
    ]

    # -----------------------------
    # Run ML with and without ICC filtering, and with feature selection
    # -----------------------------
    print("Running without ICC filtering...")
    cv_no_icc, boot_no_icc = run_transcription_ml(
        df=df, features=features, diag_pairs=diag_pairs, methods=methods,
        clf=model, cv_splitter=cv_splitter, filter_by_icc=False,
        n_bootstrap=1000, random_state=SEED
    )

    print("Running with ICC filtering...")
    cv_icc, boot_icc = run_transcription_ml(
        df=df, features=features, diag_pairs=diag_pairs, methods=methods,
        clf=model, cv_splitter=cv_splitter, filter_by_icc=True,
        icc_results=icc_results, n_bootstrap=1000, random_state=SEED
    )

    print(f"Running with feature selection (k={K})...")
    cv_k, boot_k = run_transcription_ml(
        df=df, features=features, diag_pairs=diag_pairs, methods=methods,
        clf=model_k, cv_splitter=cv_splitter, filter_by_icc=False,
        n_bootstrap=1000, random_state=SEED, feature_selection=f"kbest{K}"
    )

    # Combine results
    cv_df = pd.concat([cv_no_icc, cv_icc, cv_k], ignore_index=True)
    boot_df = pd.concat([boot_no_icc, boot_icc, boot_k], ignore_index=True)

    # Output results
    cv_df.to_csv(results_dir / "cv_results.csv", index=False)
    boot_df.to_csv(results_dir / "bootstrap_results.csv", index=False)

    print(f"CV results: {len(cv_df)} fold-level results")
    print(f"Bootstrap results: {len(boot_df)} bootstrap results")
    print(f"All results saved to {results_dir}")
