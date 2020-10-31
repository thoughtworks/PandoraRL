import os

from flask import Blueprint, Response, send_file, render_template

controller = Blueprint('artifact', __name__)


@controller.route('/<path:path>')
def artifact(path):
    file_path = f'/{path}'
    if not os.path.exists(file_path):
        return render_template("404.html")

    return send_file(file_path)


