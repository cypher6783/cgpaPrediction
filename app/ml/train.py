import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from app.ml.preprocess import gpa_to_category


def build_hybrid_model():
    """Build the Voting Ensemble hybrid model with tuned hyperparameters."""
    dt = DecisionTreeRegressor(
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
    )
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
    )

    hybrid_model = VotingRegressor(
        estimators=[("decision_tree", dt), ("random_forest", rf)],
        weights=[0.35, 0.65],
    )
    return hybrid_model


def train_model(X_train, y_train):
    """Train the hybrid model."""
    model = build_hybrid_model()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)

    y_test_cat = [gpa_to_category(g) for g in y_test]
    y_pred_cat = [gpa_to_category(g) for g in y_pred]

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test_cat, y_pred_cat),
        "precision": precision_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0),
        "recall": recall_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0),
        "f1": f1_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0),
        "y_pred": y_pred.tolist(),
        "y_test": y_test.tolist(),
        "y_test_cat": y_test_cat,
        "y_pred_cat": y_pred_cat,
    }

    # Cross-validation score
    cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring="r2")
    metrics["cv_r2_mean"] = float(np.mean(cv_scores))
    metrics["cv_r2_std"] = float(np.std(cv_scores))

    return metrics


def compare_models(X_train, y_train, X_test, y_test):
    """Compare individual models vs hybrid model."""
    dt = DecisionTreeRegressor(
        max_depth=8, min_samples_split=10, min_samples_leaf=5,
        max_features="sqrt", random_state=42
    )
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_split=5,
        min_samples_leaf=3, max_features="sqrt", random_state=42
    )
    hybrid = build_hybrid_model()

    results = {}
    for name, model in [("Decision Tree", dt), ("Random Forest", rf), ("Hybrid (Voting)", hybrid)]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_test_cat = [gpa_to_category(g) for g in y_test]
        y_pred_cat = [gpa_to_category(g) for g in y_pred]

        results[name] = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test_cat, y_pred_cat),
            "precision": precision_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0),
            "recall": recall_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0),
            "f1": f1_score(y_test_cat, y_pred_cat, average="weighted", zero_division=0),
        }
    return results


def generate_confusion_matrix(y_test_cat, y_pred_cat, save_path):
    """Generate and save confusion matrix plot."""
    labels = ["Excellent", "Good", "Average", "Poor"]
    cm = confusion_matrix(y_test_cat, y_pred_cat, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - Hybrid Model")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_comparison_chart(results, save_path):
    """Generate model comparison bar chart."""
    models = list(results.keys())
    metrics = ["accuracy", "f1", "r2"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in models]
        colors = ["#3498db", "#2ecc71", "#e74c3c"]
        axes[i].bar(models, values, color=colors)
        axes[i].set_title(metric.upper())
        axes[i].set_ylim(0, 1.1)
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_roc_chart(y_test_cat, y_pred_cat, save_path):
    """Generate precision-recall chart by category."""
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import precision_recall_curve, average_precision_score

    labels = ["Excellent", "Good", "Average", "Poor"]
    present_labels = [l for l in labels if l in y_test_cat or l in y_pred_cat]

    if len(present_labels) < 2:
        return

    n_classes = len(present_labels)
    y_true_bin = label_binarize(y_test_cat, classes=present_labels)
    y_pred_bin = label_binarize(y_pred_cat, classes=present_labels)

    # label_binarize returns 1D for 2 classes — reshape
    if n_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
        y_pred_bin = np.column_stack([1 - y_pred_bin, y_pred_bin])

    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    for idx, label in enumerate(present_labels):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], y_pred_bin[:, idx])
        ap = average_precision_score(y_true_bin[:, idx], y_pred_bin[:, idx])
        axes[idx].plot(recall, precision, linewidth=2)
        axes[idx].set_title(f"{label} (AP={ap:.3f})")
        axes[idx].set_xlabel("Recall")
        axes[idx].set_ylabel("Precision")
        axes[idx].set_xlim([0, 1])
        axes[idx].set_ylim([0, 1.1])

    plt.suptitle("Precision-Recall by Category", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_model(model, scaler, label_encoders, feature_names, model_dir):
    """Save trained model and preprocessing objects."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "hybrid_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.pkl"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.pkl"))


def load_model(model_dir):
    """Load trained model and preprocessing objects."""
    model = joblib.load(os.path.join(model_dir, "hybrid_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.pkl"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
    return model, scaler, label_encoders, feature_names
