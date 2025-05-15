import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

__all__ = [
    "run_notebook_validation",
]


def run_notebook_validation(
    *,
    model,
    X_train,
    y_train,
    X_train_oversampled,
    y_train_oversampled,
    X_test,
    y_test,
    top_pdp: int = 3,
):
    """Replicate notebook‑style validation: ROC curves, prob histogram, feature importance, SHAP, PDP, permutation importance.

    Parameters
    ----------
    model : fitted XGBClassifier
    X_train, y_train : original train split before oversampling
    X_train_oversampled, y_train_oversampled : resampled train matrix used for fitting
    X_test, y_test : held‑out test split
    top_pdp : int, default 3 – number of top SHAP features to plot PDP
    """
    # ---------------- ROC curves ----------------
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_proba_train = model.predict_proba(X_train_oversampled)[:, 1]
    y_proba_train_orig = model.predict_proba(X_train)[:, 1]

    auc_test = roc_auc_score(y_test, y_proba_test)
    auc_train = roc_auc_score(y_train_oversampled, y_proba_train)
    auc_train_orig = roc_auc_score(y_train, y_proba_train_orig)

    fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)
    fpr_train, tpr_train, _ = roc_curve(y_train_oversampled, y_proba_train)
    fpr_train_orig, tpr_train_orig, _ = roc_curve(y_train, y_proba_train_orig)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_test, tpr_test, label=f"Test AUC = {auc_test:.2f}")
    plt.plot(fpr_train, tpr_train, label=f"Train AUC = {auc_train:.2f}", linestyle="--")
    plt.plot(fpr_train_orig, tpr_train_orig, label=f"Train AUC (orig) = {auc_train_orig:.2f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Train vs Test)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------------- Probability histogram ----------------
    _plot_predicted_proba_by_class(y_test, y_proba_test)

    # ---------------- Feature importances (gain) ----------------
    fi = pd.Series(model.feature_importances_, index=X_train.columns)
    top10 = fi.sort_values(ascending=False).head(10)
    top10.plot(kind="barh", title="Top 10 Feature Importances")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

    # ---------------- SHAP summary ----------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)

    shap_imp = np.abs(shap_values.values).mean(axis=0)
    top_features = X_train.columns[np.argsort(shap_imp)[-top_pdp:][::-1]].tolist()
    print("Top SHAP features:", top_features)

    # ---------------- PDP/ICE for top SHAP features ----------------
    for f in top_features:
        PartialDependenceDisplay.from_estimator(model, X_train, [f], random_state=42)
        plt.title(f"Partial Dependence: {f}")
        plt.tight_layout()
        plt.show()

    # ---------------- Permutation importance ----------------
    perm = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42)
    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values(by="importance_mean", ascending=False)

    print("Permutation importance (top 10):")
    display(perm_df.head(10))


# helper histogram plot

def _plot_predicted_proba_by_class(y_true, y_pred_proba, bins: int = 50):
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_true == 0], bins=bins, alpha=0.6, label="y = 0", color="blue", density=True)
    plt.hist(y_pred_proba[y_true == 1], bins=bins, alpha=0.6, label="y = 1", color="red", density=True)
    plt.title("Histogram of Predicted Probabilities by True Class")
    plt.xlabel("Predicted Probability (class 1)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
