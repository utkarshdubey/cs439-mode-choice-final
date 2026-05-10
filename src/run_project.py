"""
This script loads the Statsmodels Travel Mode Choice dataset, performs leakage safe
preprocessing, fits unsupervised traveler segments on the training set only, trains
several supervised models, evaluates row-level and choice-set metrics, and writes
all figures/tables used in the final report.

Run from the repository root with:
    python src/run_project.py
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

SEED = 439
MODE_MAP = {1: "air", 2: "train", 3: "bus", 4: "car"}
MODE_ORDER = ["air", "train", "bus", "car"]
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"


def load_data() -> pd.DataFrame:
    """Load and validate the Travel Mode Choice data."""
    df = sm.datasets.modechoice.load_pandas().data.copy()
    df["individual"] = df["individual"].astype(int)
    df["mode"] = df["mode"].astype(int)
    df["choice"] = df["choice"].astype(int)
    df["mode_name"] = df["mode"].map(MODE_MAP)

    # Basic reproducibility/sanity checks that catch data leakage and corruption.
    assert df.isna().sum().sum() == 0, "Unexpected missing values."
    assert df.duplicated().sum() == 0, "Unexpected duplicate rows."
    assert (df.groupby("individual")["choice"].sum() == 1).all(), (
        "Each individual should choose exactly one travel mode."
    )
    assert set(df.groupby("individual").size().unique()) == {4}, (
        "Each individual should have exactly four alternatives."
    )
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create utility-inspired features using only observed covariates."""
    out = df.copy()
    out["total_time"] = out["ttme"] + out["invt"]
    out["cost_per_income"] = out["gc"] / (out["hinc"] + 1e-6)
    out["party_adjusted_cost"] = out["gc"] / out["psize"]

    g = out.groupby("individual")
    out["gc_rel_min"] = out["gc"] - g["gc"].transform("min")
    out["time_rel_min"] = out["total_time"] - g["total_time"].transform("min")
    out["is_lowest_gc"] = (out["gc"] == g["gc"].transform("min")).astype(int)
    out["is_fastest"] = (out["total_time"] == g["total_time"].transform("min")).astype(int)
    out["gc_vs_mean"] = out["gc"] / (g["gc"].transform("mean") + 1e-6)
    out["time_vs_mean"] = out["total_time"] / (g["total_time"].transform("mean") + 1e-6)
    return out


def traveler_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build one leakage-safe row per traveler for K-Means segmentation."""
    rows: List[Dict[str, float]] = []
    for individual, sub in df.groupby("individual"):
        sub = sub.sort_values("mode")
        row: Dict[str, float] = {
            "individual": individual,
            "hinc": float(sub["hinc"].iloc[0]),
            "psize": float(sub["psize"].iloc[0]),
        }
        for col in ["gc", "ttme", "invc", "invt", "total_time"]:
            vals = sub[col].to_numpy(dtype=float)
            row[f"{col}_min"] = float(vals.min())
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_max"] = float(vals.max())
            row[f"{col}_range"] = float(vals.max() - vals.min())
            for mode, val in zip(sub["mode_name"], vals):
                row[f"{mode}_{col}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows).set_index("individual")


def add_segments(
    df: pd.DataFrame, train_ids: set[int], n_clusters: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame, PCA, np.ndarray]:
    """Fit K-Means on training travelers only and assign labels to all rows."""
    train_matrix = traveler_matrix(df[df["individual"].isin(train_ids)])
    all_matrix = traveler_matrix(df)

    scaler = StandardScaler().fit(train_matrix)
    x_train = scaler.transform(train_matrix)
    x_all = scaler.transform(all_matrix)

    k_rows = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(x_train)
        k_rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(x_train, km.labels_),
                "davies_bouldin": davies_bouldin_score(x_train, km.labels_),
            }
        )
    k_stats = pd.DataFrame(k_rows)

    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10).fit(x_train)
    segments = pd.Series(kmeans.predict(x_all), index=all_matrix.index, name="segment").astype(str)

    out = df.copy()
    out["segment"] = out["individual"].map(segments)

    pca = PCA(n_components=2, random_state=SEED).fit(x_train)
    pca_all = pca.transform(x_all)
    pca_df = pd.DataFrame(
        {
            "individual": all_matrix.index,
            "pc1": pca_all[:, 0],
            "pc2": pca_all[:, 1],
            "segment": segments.values,
        }
    )
    return out, k_stats, pca, pca_df


def build_pipeline(model, numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def choice_set_metrics(
    test_df: pd.DataFrame, scores: np.ndarray
) -> Tuple[float, float, np.ndarray, pd.Series, pd.Series]:
    """Evaluate by choosing the highest-scoring alternative for each traveler."""
    tmp = test_df[["individual", "mode_name", "choice"]].copy()
    tmp["score"] = scores
    pred_idx = tmp.groupby("individual")["score"].idxmax()
    pred_modes = tmp.loc[pred_idx].set_index("individual")["mode_name"]
    true_modes = tmp[tmp["choice"] == 1].set_index("individual")["mode_name"]

    top1 = float((pred_modes == true_modes).mean())
    top2_hits = []
    for _, sub in tmp.groupby("individual"):
        top_two = sub.nlargest(2, "score")["mode_name"].tolist()
        true_mode = sub.loc[sub["choice"] == 1, "mode_name"].iloc[0]
        top2_hits.append(true_mode in top_two)
    top2 = float(np.mean(top2_hits))
    cm = confusion_matrix(true_modes, pred_modes, labels=MODE_ORDER)
    return top1, top2, cm, pred_modes, true_modes


def evaluate_model(name: str, pipe: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, test_df: pd.DataFrame):
    prob = pipe.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    top1, top2, cm, pred_modes, true_modes = choice_set_metrics(test_df, prob)
    metrics = {
        "model": name,
        "row_accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
        "log_loss": log_loss(y_test, prob),
        "choice_set_accuracy": top1,
        "top2_accuracy": top2,
    }
    return metrics, prob, cm, pred_modes, true_modes


def clean_feature_name(name: str) -> str:
    return (
        name.replace("num__", "")
        .replace("cat__", "")
        .replace("mode_name_", "mode=")
        .replace("segment_", "segment=")
        .replace("_", " ")
    )


def save_figures(
    k_stats: pd.DataFrame,
    pca_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    test_df: pd.DataFrame,
    probas: Dict[str, np.ndarray],
    rf_cm: np.ndarray,
    shap_importance: pd.DataFrame,
):
    FIG_DIR.mkdir(exist_ok=True)

    # K selection: silhouette.
    plt.figure(figsize=(5.5, 3.6))
    plt.plot(k_stats["k"], k_stats["silhouette"], marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("K-Means cluster selection")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kmeans_silhouette.pdf")
    plt.savefig(FIG_DIR / "kmeans_silhouette.png", dpi=200)
    plt.close()

    # PCA visualization.
    plt.figure(figsize=(5.4, 4.2))
    for seg, sub in pca_df.groupby("segment"):
        plt.scatter(sub["pc1"], sub["pc2"], s=28, alpha=0.8, label=f"Segment {seg}")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title("Traveler segments (PCA projection)")
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pca_segments.pdf")
    plt.savefig(FIG_DIR / "pca_segments.png", dpi=200)
    plt.close()

    # Choice-set accuracy including rules.
    plot_df = pd.concat(
        [
            baseline_df[["model", "choice_set_accuracy"]],
            metrics_df[["model", "choice_set_accuracy"]],
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(8.0, 3.8))
    x = np.arange(len(plot_df))
    plt.bar(x, plot_df["choice_set_accuracy"])
    plt.xticks(x, plot_df["model"], rotation=38, ha="right")
    plt.ylabel("Top-1 choice-set accuracy")
    plt.ylim(0, 1.05)
    plt.title("Operational accuracy by method")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "choice_set_accuracy.pdf")
    plt.savefig(FIG_DIR / "choice_set_accuracy.png", dpi=200)
    plt.close()

    # ROC curves.
    y_test = test_df["choice"].to_numpy()
    plt.figure(figsize=(5.5, 4.2))
    for name, prob in probas.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on held-out travelers")
    plt.legend(frameon=False, fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curves.pdf")
    plt.savefig(FIG_DIR / "roc_curves.png", dpi=200)
    plt.close()

    # Confusion matrix for the best operational model.
    plt.figure(figsize=(4.5, 4.0))
    plt.imshow(rf_cm)
    plt.xticks(np.arange(len(MODE_ORDER)), MODE_ORDER)
    plt.yticks(np.arange(len(MODE_ORDER)), MODE_ORDER)
    plt.xlabel("Predicted mode")
    plt.ylabel("True mode")
    plt.title("Random Forest choice-set confusion matrix")
    for i in range(rf_cm.shape[0]):
        for j in range(rf_cm.shape[1]):
            plt.text(j, i, str(rf_cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rf_confusion_matrix.pdf")
    plt.savefig(FIG_DIR / "rf_confusion_matrix.png", dpi=200)
    plt.close()

    # SHAP/TreeSHAP top features from XGBoost.
    top = shap_importance.sort_values("mean_abs_shap", ascending=True).tail(10)
    plt.figure(figsize=(5.5, 4.0))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("XGBoost feature attribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_top_features.pdf")
    plt.savefig(FIG_DIR / "shap_top_features.png", dpi=200)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_data()
    df = add_engineered_features(raw)
    df.to_csv(DATA_DIR / "modechoice_statsmodels.csv", index=False)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, df["choice"], groups=df["individual"]))
    train_ids = set(df.iloc[train_idx]["individual"].unique())

    df, k_stats, pca_model, pca_df = add_segments(df, train_ids, n_clusters=2)
    train_df = df[df["individual"].isin(train_ids)].copy()
    test_df = df[~df["individual"].isin(train_ids)].copy()

    numeric_basic = ["ttme", "invc", "invt", "gc", "hinc", "psize"]
    numeric_engineered = [
        "ttme",
        "invc",
        "invt",
        "gc",
        "hinc",
        "psize",
        "total_time",
        "cost_per_income",
        "party_adjusted_cost",
        "gc_rel_min",
        "time_rel_min",
        "is_lowest_gc",
        "is_fastest",
        "gc_vs_mean",
        "time_vs_mean",
    ]

    x_train = train_df.drop(columns=["choice"])
    y_train = train_df["choice"]
    x_test = test_df.drop(columns=["choice"])
    y_test = test_df["choice"]

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)

    models: List[Tuple[str, Pipeline]] = [
        (
            "Logit-basic",
            build_pipeline(
                LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=SEED),
                numeric_basic,
                ["mode_name"],
            ),
        ),
        (
            "Logit-engineered",
            build_pipeline(
                LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced", random_state=SEED),
                numeric_engineered,
                ["mode_name"],
            ),
        ),
        (
            "Logit+segments",
            build_pipeline(
                LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced", random_state=SEED),
                numeric_engineered,
                ["mode_name", "segment"],
            ),
        ),
        (
            "RandomForest+segments",
            build_pipeline(
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=1,
                ),
                numeric_engineered,
                ["mode_name", "segment"],
            ),
        ),
        (
            "ExtraTrees+segments",
            build_pipeline(
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=1,
                ),
                numeric_engineered,
                ["mode_name", "segment"],
            ),
        ),
        (
            "XGBoost+segments",
            build_pipeline(
                XGBClassifier(
                    n_estimators=100,
                    max_depth=2,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=SEED,
                    eval_metric="logloss",
                    objective="binary:logistic",
                    n_jobs=1,
                    scale_pos_weight=neg / pos,
                ),
                numeric_engineered,
                ["mode_name", "segment"],
            ),
        ),
    ]

    cv = GroupKFold(n_splits=5)
    metric_rows = []
    probas: Dict[str, np.ndarray] = {}
    confusion_matrices: Dict[str, np.ndarray] = {}
    fitted: Dict[str, Pipeline] = {}

    for name, pipe in models:
        cv_scores = cross_val_score(
            pipe,
            x_train,
            y_train,
            groups=train_df["individual"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
        )
        pipe.fit(x_train, y_train)
        metrics, prob, cm, _, _ = evaluate_model(name, pipe, x_test, y_test, test_df)
        metrics["cv_roc_auc_mean"] = float(cv_scores.mean())
        metrics["cv_roc_auc_sd"] = float(cv_scores.std())
        metric_rows.append(metrics)
        probas[name] = prob
        confusion_matrices[name] = cm
        fitted[name] = pipe

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RES_DIR / "model_metrics.csv", index=False)

    # Baselines that choose a single mode within each traveler choice set.
    baseline_rows = []
    for label, scores in [
        ("Rule: lowest GC", -test_df["gc"].to_numpy()),
        ("Rule: fastest total time", -test_df["total_time"].to_numpy()),
    ]:
        top1, top2, _, _, _ = choice_set_metrics(test_df, scores)
        baseline_rows.append({"model": label, "choice_set_accuracy": top1, "top2_accuracy": top2})
    mode_counts = train_df[train_df["choice"] == 1]["mode_name"].value_counts()
    mode_score = {mode: len(mode_counts) - i for i, mode in enumerate(mode_counts.index)}
    top1, top2, _, _, _ = choice_set_metrics(test_df, test_df["mode_name"].map(mode_score).to_numpy())
    baseline_rows.append({"model": "Rule: train popularity", "choice_set_accuracy": top1, "top2_accuracy": top2})
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(RES_DIR / "baseline_metrics.csv", index=False)

    # Segment summary for interpretation.
    segment_summary = (
        df.groupby("segment")
        .agg(
            individuals=("individual", "nunique"),
            rows=("choice", "size"),
            choice_rate=("choice", "mean"),
            mean_income=("hinc", "mean"),
            mean_party_size=("psize", "mean"),
            mean_gc=("gc", "mean"),
            mean_total_time=("total_time", "mean"),
        )
        .reset_index()
    )
    segment_summary.to_csv(RES_DIR / "segment_summary.csv", index=False)
    k_stats.to_csv(RES_DIR / "kmeans_selection.csv", index=False)

    # TreeSHAP for XGBoost interpretability.
    xgb_pipe = fitted["XGBoost+segments"]
    transformed = xgb_pipe.named_steps["preprocess"].transform(x_test)
    names = [clean_feature_name(n) for n in xgb_pipe.named_steps["preprocess"].get_feature_names_out()]
    explainer = shap.TreeExplainer(xgb_pipe.named_steps["model"])
    shap_values = explainer.shap_values(transformed)
    shap_importance = pd.DataFrame(
        {"feature": names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(RES_DIR / "xgboost_shap_importance.csv", index=False)

    # Summary metadata.
    summary = pd.DataFrame(
        [
            {
                "seed": SEED,
                "train_individuals": train_df["individual"].nunique(),
                "test_individuals": test_df["individual"].nunique(),
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "positive_rate_train": float(y_train.mean()),
                "positive_rate_test": float(y_test.mean()),
            }
        ]
    )
    summary.to_csv(RES_DIR / "run_summary.csv", index=False)

    save_figures(
        k_stats=k_stats,
        pca_df=pca_df,
        metrics_df=metrics_df,
        baseline_df=baseline_df,
        test_df=test_df,
        probas=probas,
        rf_cm=confusion_matrices["RandomForest+segments"],
        shap_importance=shap_importance,
    )

    print("Run complete. Results written to:")
    print(f"  {RES_DIR}")
    print(f"  {FIG_DIR}")


if __name__ == "__main__":
    main()
