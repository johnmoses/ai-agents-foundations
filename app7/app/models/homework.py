from ..extensions import db
from datetime import datetime

class Homework(db.Model):
    __tablename__ = 'homeworks'
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    explanation = db.Column(db.Text)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
