from ..extensions import db
from flask_login import UserMixin

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='student')  # 'student' or 'teacher'

    quizzes = db.relationship('Quiz', backref='author', lazy=True)
    quiz_attempts = db.relationship('QuizAttempt', backref='student', lazy=True)
    homeworks = db.relationship('Homework', backref='author', lazy=True)
    summaries = db.relationship('Summary', backref='author', lazy=True)
