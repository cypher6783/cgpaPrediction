import os
import sys

for path in [
    r"C:\Program Files\PostgreSQL\17\pgAdmin 4\python\Lib\site-packages",
    r"C:\Program Files\PostgreSQL\18\pgAdmin 4\python\Lib\site-packages",
]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

try:
    from flask import Flask
    from flask_sqlalchemy import SQLAlchemy
    from flask_login import LoginManager
    from flask_wtf.csrf import CSRFProtect

    db = SQLAlchemy()
    login_manager = LoginManager()
    csrf = CSRFProtect()
    HAS_FLASK = True
except ImportError:
    db = None
    login_manager = None
    csrf = None
    HAS_FLASK = False


def create_app(config_class=None):
    app = Flask(__name__)

    if config_class:
        app.config.from_object(config_class)
    else:
        from app.config import Config
        app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "info"

    from app.routes.auth import auth_bp
    from app.routes.dashboard import dashboard_bp
    from app.routes.prediction import prediction_bp
    from app.routes.admin import admin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(admin_bp)

    with app.app_context():
        db.create_all()

    return app
