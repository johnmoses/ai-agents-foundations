from flask import Blueprint, render_template
from flask_login import login_required, current_user
from ..models.user import User
from flask import abort

admin_bp = Blueprint('admin', __name__, template_folder='../templates')

@admin_bp.route('/users')
@login_required
def list_users():
    # Only allow teachers or admins to view user list
    if current_user.role != 'teacher':
        abort(403)
    users = User.query.order_by(User.username).all()
    return render_template('users_list.html', users=users)
