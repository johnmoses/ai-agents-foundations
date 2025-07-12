from flask import Flask
from flask_migrate import Migrate
from .extensions import db, bcrypt, login_manager
from .routes.main import main_bp
from .routes.auth import auth_bp
from .routes.quiz import quiz_bp
from .routes.homework import homework_bp
from .routes.summarizer import summarizer_bp
from .routes.chat import chat_bp
from .routes.admin import admin_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    migrate = Migrate(app, db)

    # Import models to register tables
    from .models.user import User
    from .models.chat import ChatMessage
    from .models.quiz import Quiz, QuizAttempt, QuizAnswer
    from .models.homework import Homework
    from .models.summary import Summary

    with app.app_context():
        db.create_all()

    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(quiz_bp, url_prefix='/quiz')
    app.register_blueprint(homework_bp, url_prefix='/homework')
    app.register_blueprint(summarizer_bp, url_prefix='/summarize')
    app.register_blueprint(chat_bp, url_prefix='/chat')
    app.register_blueprint(admin_bp, url_prefix='/admin')

    return app
