import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_super_secret_key_here'
    basedir = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'database.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
