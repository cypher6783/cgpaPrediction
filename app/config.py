import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "postgresql://postgres:asphalt6@localhost:5432/msendoo_prediction"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
