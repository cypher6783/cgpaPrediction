from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from app import db
from app.models import User, Student

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/admin/users")
@login_required
def manage_users():
    if not current_user.is_admin():
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard.index"))
    users = User.query.all()
    return render_template("admin_users.html", users=users)


@admin_bp.route("/admin/users/<int:user_id>/role", methods=["POST"])
@login_required
def change_role(user_id):
    if not current_user.is_admin():
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard.index"))

    user = User.query.get_or_404(user_id)
    new_role = request.form.get("role")
    if new_role in ["admin", "lecturer", "student"]:
        user.role = new_role
        db.session.commit()
        flash(f"Updated {user.username}'s role to {new_role}.", "success")
    return redirect(url_for("admin.manage_users"))


@admin_bp.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@login_required
def delete_user(user_id):
    if not current_user.is_admin():
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard.index"))

    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash("Cannot delete your own account.", "danger")
        return redirect(url_for("admin.manage_users"))

    db.session.delete(user)
    db.session.commit()
    flash(f"User {user.username} deleted.", "success")
    return redirect(url_for("admin.manage_users"))


@admin_bp.route("/admin/students")
@login_required
def manage_students():
    if not current_user.is_admin():
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard.index"))
    students = Student.query.all()
    return render_template("admin_students.html", students=students)
