import json

from flask import Blueprint, Response, request

from src.main.DDService import DDService

controller = Blueprint('dd_controller', __name__)
dd_service = DDService()


@controller.route('/', methods=['POST'])
def test_method():
    try:
        protein_file = request.files['proteinFile']
        ligand_file = request.files['ligandFile']
        dd_service.triggerRLAgent(protein_file, ligand_file)
    except Exception as e:
        raise Exception(getattr(e, 'message', repr(e)))
    return Response(status=202, content_type='application/json')
