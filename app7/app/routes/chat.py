from flask import Blueprint, request, jsonify, render_template, session
from flask_login import login_required, current_user
from ..models.chat import ChatMessage
from ..extensions import db
from ..services.service import rag, call_llm
from ..services.multi_agent import AgentOrchestrator, RagAgent, CalculatorAgent, QuizAgent

chat_bp = Blueprint("chat", __name__, template_folder="../templates")

# Initialize agents and orchestrator
rag_agent = RagAgent(rag, call_llm)
calculator_agent = CalculatorAgent()
quiz_agent = QuizAgent()

orchestrator = AgentOrchestrator({
    'rag': rag_agent,
    'calculator': calculator_agent,
    'quiz': quiz_agent,
})

def get_chat_history():
    if "chat_history" not in session:
        session["chat_history"] = []
    return session["chat_history"]


def save_chat_turn(user_msg, ai_msg):
    history = get_chat_history()
    history.append({"user": user_msg, "ai": ai_msg})
    if len(history) > 10:
        history.pop(0)
    session["chat_history"] = history


def format_history_for_rag(history):
    """
    Convert [{'user': ..., 'ai': ...}, ...] to
    [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}, ...]
    """
    formatted = []
    for turn in history:
        formatted.append({"role": "user", "content": turn["user"]})
        formatted.append({"role": "assistant", "content": turn["ai"]})
    return formatted


@chat_bp.route("/", methods=["GET"])
@login_required
def chat_page():
    # Load last 20 chat messages for current user, ordered oldest first
    messages = (
        ChatMessage.query.filter_by(user_id=current_user.id)
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )
    return render_template("chat.html", messages=messages)


@chat_bp.route("/message", methods=["POST"])
@login_required
def chat_message():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please enter a message."})

    # Save user message in DB
    user_msg_db = ChatMessage(user_id=current_user.id, message=user_message, is_ai=False)
    db.session.add(user_msg_db)
    db.session.commit()

    # Retrieve recent conversation history for context (last 20 messages)
    recent_msgs = (
        ChatMessage.query.filter_by(user_id=current_user.id)
        .order_by(ChatMessage.timestamp.asc())
        .limit(20)
        .all()
    )

    # Format history for multi-agent system: list of {role, content}
    formatted_history = []
    for msg in recent_msgs:
        role = "assistant" if msg.is_ai else "user"
        formatted_history.append({"role": role, "content": msg.message})

    # Use orchestrator to handle query with multi-agent system
    ai_response = orchestrator.handle_query(user_message, formatted_history)

    # Save AI response in DB
    ai_msg_db = ChatMessage(user_id=current_user.id, message=ai_response, is_ai=True)
    db.session.add(ai_msg_db)
    db.session.commit()

    return jsonify({"response": ai_response})