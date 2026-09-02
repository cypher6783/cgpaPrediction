from flask import Blueprint, render_template
from flask_login import login_required, current_user
from app.models import Student, User

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
@login_required
def index():
    stats = {}
    if current_user.is_admin():
        stats["total_students"] = Student.query.count()
        stats["total_users"] = User.query.count()
        stats["predictions_made"] = Student.query.filter(Student.predicted_gpa.isnot(None)).count()
        students = Student.query.order_by(Student.created_at.desc()).limit(10).all()
    elif current_user.is_lecturer():
        stats["total_students"] = Student.query.count()
        stats["predictions_made"] = Student.query.filter(Student.predicted_gpa.isnot(None)).count()
        students = Student.query.order_by(Student.created_at.desc()).limit(10).all()
    else:
        stats["total_students"] = 1
        stats["predictions_made"] = 1 if current_user.students else 0
        students = current_user.students[:10] if current_user.students else []

    return render_template("dashboard.html", stats=stats, students=students)
