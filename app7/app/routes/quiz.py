from flask import Blueprint, render_template, request, redirect, url_for, flash, abort
from flask_login import login_required, current_user
from ..models.quiz import Quiz, QuizAttempt, QuizAnswer
from ..extensions import db
from ..services.service import generate_quiz_text
import json

quiz_bp = Blueprint('quiz', __name__, template_folder='../templates')

@quiz_bp.route('/generate', methods=['GET', 'POST'])
@login_required
def generate():
    if current_user.role != 'teacher':
        abort(403)
    if request.method == 'POST':
        topic = request.form.get('topic', '').strip()
        num_questions = request.form.get('num_questions', '5').strip()

        if not topic:
            flash("Please enter a quiz topic.", "warning")
            return redirect(url_for('quiz.generate'))
        try:
            num_questions = int(num_questions)
            if num_questions < 1 or num_questions > 50:
                flash("Number of questions must be between 1 and 50.", "warning")
                return redirect(url_for('quiz.generate'))
        except ValueError:
            flash("Invalid number of questions.", "danger")
            return redirect(url_for('quiz.generate'))

        try:
            questions = generate_quiz_text(topic, num_questions)
            questions_json = json.dumps(questions)
        except Exception as e:
            flash(f"Failed to generate quiz: {e}", "danger")
            return redirect(url_for('quiz.generate'))

        quiz = Quiz(topic=topic, questions_json=questions_json, user_id=current_user.id)
        db.session.add(quiz)
        db.session.commit()

        flash("Quiz generated successfully!", "success")
        return redirect(url_for('quiz.quiz_list'))

    return render_template('generate_quiz.html')

@quiz_bp.route('/list')
@login_required
def list_quizzes():
    if current_user.role != 'teacher':
        abort(403)

    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '', type=str)

    query = Quiz.query
    if search:
        query = query.filter(Quiz.topic.ilike(f'%{search}%'))

    quizzes = query.order_by(Quiz.topic).paginate(page=page, per_page=10)

    return render_template('quiz_list.html', quizzes=quizzes, search=search)

@quiz_bp.route('/my_quizzes')
@login_required
def my_quizzes():
    if current_user.role != 'teacher':
        abort(403)  # Only teachers allowed

    # Get all quizzes authored by the current teacher
    quizzes = Quiz.query.filter_by(user_id=current_user.id).all()

    # Parse questions JSON for each quiz (assuming questions_json field stores JSON string)
    for quiz in quizzes:
        try:
            quiz.questions = quiz.get_questions()  # Implement get_questions() to return parsed JSON
        except Exception:
            quiz.questions = []

    return render_template('my_quizzes.html', quizzes=quizzes)

@quiz_bp.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'teacher':
        attempts = QuizAttempt.query.order_by(QuizAttempt.quiz_id.desc()).all()
        return render_template('dashboard.html', attempts=attempts, role='teacher')
    elif current_user.role == 'student':
        attempts = QuizAttempt.query.filter_by(student_id=current_user.id).order_by(QuizAttempt.quiz_id.desc()).all()
        return render_template('dashboard.html', attempts=attempts, role='student')
    else:
        abort(403)

@quiz_bp.route('/take/<int:quiz_id>', methods=['GET', 'POST'])
@login_required
def take_quiz(quiz_id):
    if current_user.role != 'student':
        abort(403)

    quiz = Quiz.query.get_or_404(quiz_id)
    questions = json.loads(quiz.questions_json)

    if request.method == 'POST':
        user_answers = {}
        for i in range(len(questions)):
            ans = request.form.get(f'q{i}')
            if not ans:
                flash("Please answer all questions.", "warning")
                return render_template('take_quiz.html', quiz=quiz, questions=questions)
            user_answers[i] = ans

        score = 0
        attempt = QuizAttempt(
            quiz_id=quiz.id,
            student_id=current_user.id,
            score=0,
            total_questions=len(questions)
        )
        db.session.add(attempt)
        db.session.flush()

        for i, question in enumerate(questions):
            correct = question['correct_answer']
            selected = user_answers[i]
            is_correct = (selected == correct)
            if is_correct:
                score += 1

            answer_record = QuizAnswer(
                attempt_id=attempt.id,
                question_index=i,
                question_text=question['question'],
                selected_answer=selected,
                correct_answer=correct,
                is_correct=is_correct
            )
            db.session.add(answer_record)

        attempt.score = score
        db.session.commit()

        return redirect(url_for('quiz.quiz_result', attempt_id=attempt.id))

    return render_template('take_quiz.html', quiz=quiz, questions=questions)

@quiz_bp.route('/result/<int:attempt_id>')
@login_required
def quiz_result(attempt_id):
    attempt = QuizAttempt.query.get_or_404(attempt_id)
    if current_user.id != attempt.student_id:
        abort(403)

    return render_template('quiz_result.html', attempt=attempt)

@quiz_bp.route('/submit/<int:quiz_id>', methods=['POST'])
@login_required
def submit_quiz(quiz_id):
    if current_user.role != 'teacher':
        abort(403)
    quiz = Quiz.query.get_or_404(quiz_id)
    questions = quiz.get_questions()
    user_answers = request.form

    score = 0
    total = len(questions)

    for idx, q in enumerate(questions):
        qid_str = str(idx)
        user_answer = user_answers.get(qid_str)
        if user_answer and user_answer == q['correct_answer']:
            score += 1

    attempt = QuizAttempt(
        user_id=current_user.id,
        quiz_id=quiz.id,
        score=score,
        total_questions=total
    )
    db.session.add(attempt)
    db.session.commit()

    grade_percentage = (score / total) * 100
    if grade_percentage >= 90:
        grade = 'A'
    elif grade_percentage >= 80:
        grade = 'B'
    elif grade_percentage >= 70:
        grade = 'C'
    elif grade_percentage >= 60:
        grade = 'D'
    else:
        grade = 'F'

    return render_template('quiz_result.html', score=score, total=total, grade=grade, quiz=quiz)

    
