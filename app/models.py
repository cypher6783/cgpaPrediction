from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db, login_manager


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="student")  # admin, lecturer, student
    full_name = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    students = db.relationship("Student", backref="owner", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_admin(self):
        return self.role == "admin"

    def is_lecturer(self):
        return self.role == "lecturer"

    def is_student(self):
        return self.role == "student"


class Student(db.Model):
    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False)
    full_name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    attendance_rate = db.Column(db.Float, nullable=True)  # percentage 0-100
    study_hours_per_week = db.Column(db.Float, nullable=True)
    previous_gpa = db.Column(db.Float, nullable=True)  # 0.0 - 4.0
    assignment_score = db.Column(db.Float, nullable=True)  # percentage 0-100
    midterm_score = db.Column(db.Float, nullable=True)  # percentage 0-100
    final_score = db.Column(db.Float, nullable=True)  # percentage 0-100
    participation_score = db.Column(db.Float, nullable=True)  # percentage 0-100
    predicted_gpa = db.Column(db.Float, nullable=True)
    predicted_category = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_performance_category(self, gpa):
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


class PredictionLog(db.Model):
    __tablename__ = "prediction_logs"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False)
    predicted_gpa = db.Column(db.Float, nullable=False)
    predicted_category = db.Column(db.String(50), nullable=False)
    model_version = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    student = db.relationship("Student", backref="predictions")


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
