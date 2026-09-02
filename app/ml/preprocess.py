import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_uci_dataset(filepath):
    """Load the UCI Student Performance dataset."""
    try:
        df = pd.read_csv(filepath, sep=";")
        return df
    except Exception:
        return None


def engineer_features(df):
    """Add interaction and derived features that improve prediction."""
    df = df.copy()

    # Parent education average
    if "Medu" in df.columns and "Fedu" in df.columns:
        df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Study effort score
    if "studytime" in df.columns and "failures" in df.columns:
        df["study_effort"] = df["studytime"] - df["failures"] * 0.5

    # Alcohol index
    if "Dalc" in df.columns and "Walc" in df.columns:
        df["alcohol_index"] = df["Dalc"] * 0.4 + df["Walc"] * 0.6

    # Social vs study balance
    if "goout" in df.columns and "studytime" in df.columns:
        df["social_study_ratio"] = df["goout"] / (df["studytime"] + 0.1)

    # Absence rate impact
    if "absences" in df.columns:
        df["absence_impact"] = df["absences"].apply(
            lambda x: 0 if x <= 3 else (1 if x <= 10 else (2 if x <= 20 else 3))
        )

    # Health-study interaction
    if "health" in df.columns and "studytime" in df.columns:
        df["health_study"] = df["health"] * df["studytime"]

    return df


def preprocess_student_data(df, target_col="G3"):
    """
    Preprocess student data with feature engineering.
    Returns: X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names
    """
    df = df.copy()

    # Convert G3 (0-20) to GPA (0.0-4.0)
    if target_col in df.columns and df[target_col].max() <= 20:
        df["GPA"] = df[target_col] / 5.0
        y = df["GPA"].values
    else:
        y = df[target_col].values

    # Drop target and intermediate grades (G1, G2 are not available at prediction time)
    drop_cols = [c for c in [target_col, "G1", "G2", "GPA"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Feature engineering
    X = engineer_features(X)

    # Encode categorical columns
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))

    # Remove low-variance features
    from sklearn.feature_selection import VarianceThreshold
    var_selector = VarianceThreshold(threshold=0.01)
    X_array = var_selector.fit_transform(X.values)
    kept_mask = var_selector.get_support()
    kept_columns = X.columns[kept_mask].tolist()
    X = pd.DataFrame(X_array, columns=kept_columns)

    # Correlation-based feature selection — drop highly correlated pairs
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_correlated = [col for col in upper.columns if any(upper[col] > 0.92)]
    X = X.drop(columns=drop_correlated, errors="ignore")

    feature_names = X.columns.tolist()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names


def preprocess_single_student(data_dict, scaler, label_encoders, feature_names):
    """Preprocess a single student's data for prediction."""
    df = pd.DataFrame([data_dict])

    # Encode categorical columns
    for col, le in label_encoders.items():
        if col in df.columns:
            val = df[col].iloc[0]
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                df[col] = 0

    # Apply same feature engineering
    df = engineer_features(df)

    # Ensure correct column order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]
    df = df.fillna(0)

    X_scaled = scaler.transform(df)
    return X_scaled


def gpa_to_category(gpa):
    """Convert continuous GPA to performance category."""
    if gpa >= 3.5:
        return "Excellent"
    elif gpa >= 3.0:
        return "Good"
    elif gpa >= 2.0:
        return "Average"
    else:
        return "Poor"
