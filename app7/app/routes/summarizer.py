from flask import Blueprint, render_template, request
from flask_login import login_required, current_user
from ..models.summary import Summary
from ..extensions import db
from ..services.service import generate_summary

summarizer_bp = Blueprint('summarizer', __name__, template_folder='../templates')

@summarizer_bp.route('/content', methods=['GET', 'POST'])
@login_required
def summarize_content():
    if request.method == 'POST':
        content = request.form['content']
        summary_text = generate_summary(content)
        summary = Summary(content=content, summary=summary_text, user_id=current_user.id)
        db.session.add(summary)
        db.session.commit()
        return render_template('summary_result.html', summary=summary_text)
    return render_template('summarize.html')
