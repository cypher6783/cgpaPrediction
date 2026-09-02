import pandas as pd
import numpy as np

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
except ImportError:
    class StandardScaler:
        def fit_transform(self, X):
            X_arr = np.asarray(X, dtype=float)
            self.mean_ = np.mean(X_arr, axis=0)
            self.scale_ = np.std(X_arr, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
            return (X_arr - self.mean_) / self.scale_

        def transform(self, X):
            X_arr = np.asarray(X, dtype=float)
            mean = getattr(self, "mean_", np.mean(X_arr, axis=0))
            scale = getattr(self, "scale_", np.std(X_arr, axis=0))
            scale = np.where(scale == 0, 1.0, scale)
            return (X_arr - mean) / scale

    def train_test_split(X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        n = len(X)
        shuffled_indices = np.random.permutation(n)
        test_set_size = int(n * test_size)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    class LabelEncoder:
        def fit_transform(self, y):
            self.classes_ = np.unique(y)
            return np.searchsorted(self.classes_, y)

        def transform(self, y):
            return np.searchsorted(getattr(self, "classes_", np.unique(y)), y)


def load_uci_dataset(filepath):
    """Load the UCI Student Performance dataset."""
    try:
        df = pd.read_csv(filepath, sep=";")
        return df
    except Exception:
        return None


def engineer_features(df):
    """Add interaction features for post-1st year academic performance."""
    df = df.copy()

    # Calculate 1st Year CGPA (5.0 scale) if missing but G1 and G2 exist
    if "first_year_cgpa" not in df.columns:
        if "G1" in df.columns and "G2" in df.columns:
            df["first_year_cgpa"] = ((df["G1"] + df["G2"]) / 40.0) * 5.0
        elif "previous_gpa" in df.columns:
            df["first_year_cgpa"] = df["previous_gpa"]
        else:
            df["first_year_cgpa"] = 3.50

    # Study effort score
    if "studytime" in df.columns and "failures" in df.columns:
        df["study_effort"] = df["studytime"] - df["failures"] * 0.5
    else:
        df["study_effort"] = df.get("studytime", 2)

    return df


def preprocess_student_data(df, target_col="G3"):
    """
    Preprocess student data using selected academic features (Post-1st Year).
    Selected Features: 1st Year CGPA (5.0 scale), Study Time, Absences, Past Failures, Study Effort.
    Returns: X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names
    """
    df = df.copy()

    # Target variable: Final GPA (0.00 - 5.00 scale)
    if target_col in df.columns and df[target_col].max() <= 20:
        df["GPA"] = (df[target_col] / 20.0) * 5.0
        y = df["GPA"].values
    elif "GPA" in df.columns:
        y = df["GPA"].values
    else:
        y = df[target_col].values

    # Feature engineering
    df = engineer_features(df)

    # Restrict strictly to selected academic feature set
    selected_academic_cols = [
        "first_year_cgpa",
        "studytime",
        "absences",
        "failures",
        "study_effort",
    ]

    for col in selected_academic_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[selected_academic_cols].copy()

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))

    feature_names = X.columns.tolist()
    label_encoders = {}  # All selected academic features are numeric

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
    """Convert continuous GPA (5.0 scale) to official Class of Degree."""
    if gpa >= 4.50:
        return "First Class"
    elif gpa >= 3.50:
        return "Second Class Upper (2:1)"
    elif gpa >= 2.40:
        return "Second Class Lower (2:2)"
    elif gpa >= 1.50:
        return "Third Class"
    else:
        return "Pass / Fail"
