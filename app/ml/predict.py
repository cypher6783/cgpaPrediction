import os
from app.ml.train import load_model
from app.ml.preprocess import preprocess_single_student, gpa_to_category
from app.config import Config


def predict_student_performance(student_data):
    """
    Predict academic performance for a single student.

    student_data: dict with keys matching the feature names
    Returns: dict with predicted GPA and category
    """
    model_dir = Config.MODEL_DIR
    model_path = os.path.join(model_dir, "hybrid_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not trained yet. Please train the model first.")

    model, scaler, label_encoders, feature_names = load_model(model_dir)
    X = preprocess_single_student(student_data, scaler, label_encoders, feature_names)
    predicted_gpa = model.predict(X)[0]

    # Clamp GPA to valid 5.0 range
    predicted_gpa = max(0.0, min(5.0, predicted_gpa))
    category = gpa_to_category(predicted_gpa)

    return {
        "predicted_gpa": round(float(predicted_gpa), 2),
        "category": category,
    }


def predict_batch(students_df):
    """
    Predict performance for multiple students.

    students_df: DataFrame with student features
    Returns: DataFrame with predictions added
    """
    model_dir = Config.MODEL_DIR
    model_path = os.path.join(model_dir, "hybrid_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not trained yet.")

    model, scaler, label_encoders, feature_names = load_model(model_dir)

    import pandas as pd
    df = students_df.copy()

    # Preprocess each row
    from app.ml.preprocess import preprocess_single_student
    predictions = []
    for _, row in df.iterrows():
        data_dict = row.to_dict()
        X = preprocess_single_student(data_dict, scaler, label_encoders, feature_names)
        gpa = model.predict(X)[0]
        gpa = max(0.0, min(5.0, gpa))
        predictions.append({
            "predicted_gpa": round(gpa, 2),
            "category": gpa_to_category(gpa),
        })

    pred_df = pd.DataFrame(predictions)
    result = pd.concat([df, pred_df], axis=1)
    return result
