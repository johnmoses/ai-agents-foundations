from flask import Blueprint, render_template, request
from flask_login import login_required, current_user
from ..models.homework import Homework
from ..extensions import db
from ..services.service import generate_homework_explanation

homework_bp = Blueprint('homework', __name__, template_folder='../templates')

@homework_bp.route('/help', methods=['GET', 'POST'])
@login_required
def help_homework():
    if request.method == 'POST':
        question = request.form['question']
        explanation = generate_homework_explanation(question)
        hw = Homework(question=question, explanation=explanation, user_id=current_user.id)
        db.session.add(hw)
        db.session.commit()
        return render_template('homework_result.html', question=question, explanation=explanation)
    return render_template('homework_help.html')
