import os
import io
import csv
import pandas as pd
from flask import Blueprint, render_template, request, flash, redirect, url_for, send_file, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import Student, PredictionLog
from app.ml.predict import predict_student_performance, predict_batch
from app.ml.train import (
    train_model, evaluate_model, compare_models,
    generate_confusion_matrix, generate_comparison_chart, generate_roc_chart,
    save_model, load_model,
)
from app.ml.preprocess import load_uci_dataset, preprocess_student_data
from app.config import Config

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        data = {
            "first_year_cgpa": float(request.form.get("first_year_cgpa", 3.5)),
            "studytime": float(request.form.get("studytime", 2)),
            "absences": float(request.form.get("absences", 0)),
            "failures": float(request.form.get("failures", 0)),
        }

        try:
            result = predict_student_performance(data)
        except FileNotFoundError:
            flash("Model not trained yet. Please train the model first.", "warning")
            return redirect(url_for("prediction.train_model_view"))

        student_name = request.form.get("student_name", "Unknown").strip().title()
        student_id = request.form.get("student_id_number", "N/A").strip().upper()

        student = Student.query.filter_by(student_id=student_id).first()
        if student:
            student.full_name = student_name
            student.attendance_rate = max(0, 100 - data["absences"] * 2)
            student.study_hours_per_week = data["studytime"] * 5
            student.previous_gpa = data["first_year_cgpa"]
            student.predicted_gpa = float(result["predicted_gpa"])
            student.predicted_category = result["category"]
        else:
            student = Student(
                full_name=student_name,
                student_id=student_id,
                age=20,
                gender="N/A",
                attendance_rate=max(0, 100 - data["absences"] * 2),
                study_hours_per_week=data["studytime"] * 5,
                previous_gpa=data["first_year_cgpa"],
                assignment_score=data["first_year_cgpa"] * 20,
                midterm_score=data["first_year_cgpa"] * 20,
                predicted_gpa=float(result["predicted_gpa"]),
                predicted_category=result["category"],
            )
            if current_user.is_student():
                student.user_id = current_user.id
            db.session.add(student)
        db.session.commit()

        log = PredictionLog(
            student_id=student.id,
            predicted_gpa=float(result["predicted_gpa"]),
            predicted_category=result["category"],
            model_version="v1.0",
        )
        db.session.add(log)
        db.session.commit()

        return render_template("prediction_result.html", result=result, student=student)

    return render_template("predict.html")


@prediction_bp.route("/predict/upload", methods=["POST"])
@login_required
def upload_csv():
    if "file" not in request.files:
        flash("No file uploaded.", "danger")
        return redirect(url_for("prediction.predict"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.", "danger")
        return redirect(url_for("prediction.predict"))

    if not file.filename.endswith(".csv"):
        flash("Please upload a CSV file.", "danger")
        return redirect(url_for("prediction.predict"))

    try:
        df = pd.read_csv(file)
        result_df = predict_batch(df)

        output = io.BytesIO()
        result_df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name="predictions.csv",
        )
    except FileNotFoundError:
        flash("Model not trained yet. Please train the model first.", "warning")
        return redirect(url_for("prediction.train_model_view"))
    except Exception as e:
        flash(f"Error processing file: {str(e)}", "danger")
        return redirect(url_for("prediction.predict"))


@prediction_bp.route("/train", methods=["GET", "POST"])
@login_required
def train_model_view():
    if not current_user.is_admin():
        flash("Only admins can train models.", "danger")
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        dataset_path = os.path.join(Config.DATA_DIR, "student-mat.csv")

        if not os.path.exists(dataset_path):
            flash("Dataset not found. Please place student-mat.csv in the data/ folder.", "danger")
            return render_template("train.html")

        df = load_uci_dataset(dataset_path)
        if df is None:
            flash("Error loading dataset.", "danger")
            return render_template("train.html")

        X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = preprocess_student_data(df)
        model = train_model(X_train, y_train)
        save_model(model, scaler, label_encoders, feature_names, Config.MODEL_DIR)

        metrics = evaluate_model(model, X_test, y_test)
        comparison = compare_models(X_train, y_train, X_test, y_test)

        chart_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "charts")
        os.makedirs(chart_dir, exist_ok=True)

        generate_confusion_matrix(metrics["y_test_cat"], metrics["y_pred_cat"], os.path.join(chart_dir, "confusion_matrix.png"))
        generate_comparison_chart(comparison, os.path.join(chart_dir, "model_comparison.png"))
        generate_roc_chart(metrics["y_test_cat"], metrics["y_pred_cat"], os.path.join(chart_dir, "roc_chart.png"))

        return render_template(
            "train_result.html",
            metrics=metrics,
            comparison=comparison,
            charts_dir="charts",
        )

    return render_template("train.html")


@prediction_bp.route("/history")
@login_required
def history():
    if current_user.is_student():
        students = Student.query.filter_by(user_id=current_user.id).order_by(Student.created_at.desc()).all()
    else:
        students = Student.query.filter(Student.predicted_gpa.isnot(None)).order_by(Student.created_at.desc()).all()
    return render_template("history.html", students=students)


@prediction_bp.route("/export/csv")
@login_required
def export_csv():
    if current_user.is_student():
        students = Student.query.filter_by(user_id=current_user.id).all()
    else:
        students = Student.query.filter(Student.predicted_gpa.isnot(None)).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Name", "Predicted GPA", "Category", "Date"])
    for s in students:
        writer.writerow([s.student_id, s.full_name, s.predicted_gpa, s.predicted_category, s.created_at])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv",
    )
