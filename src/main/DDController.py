import json

from flask import Blueprint, Response, request

controller = Blueprint('dd_controller', __name__)


@controller.route('/', methods=['POST'])
def test_method():
    try:
        file = request.files['proteinFile']
        file = request.files['ligandFile']
    except Exception as e:
        raise Exception(getattr(e, 'message', repr(e)))
    return Response(status=202, content_type='application/json')
