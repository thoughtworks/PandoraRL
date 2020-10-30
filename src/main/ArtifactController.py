import os

from flask import Blueprint, Response, send_file

controller = Blueprint('artifact', __name__)


@controller.route('/<path:path>')
def artifact(path):
    file_path = f'/{path}'
    if not os.path.exists(file_path):
        return Response(status=404)

    return send_file(file_path)


