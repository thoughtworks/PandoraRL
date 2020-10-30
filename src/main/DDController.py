import json

from flask import Blueprint, Response, request

from src.main.DDService import DDService

controller = Blueprint('dd_controller', __name__)
dd_service = DDService()


@controller.route('/jobs', methods=['GET'])
def init():
    metadata = dd_service.init()
    return Response(response=json.dumps(metadata), status=202, content_type='application/json')


@controller.route('/', methods=['POST'])
def trigger_agent():
    try:
        protein_file = request.files['proteinFile']
        string_format = request.form['smiles_string']
        if string_format == 'true':
            ligand_file = request.form['ligandInput']
        else:
            ligand_file = request.files['ligandFile']
        jobid= dd_service.triggerRLAgent(protein_file, ligand_file, string_format)
    except Exception as e:
        raise Exception()
    return Response(response=json.dumps(jobid), status=202, content_type='application/json')


@controller.route('/logs', methods=['GET'])
def get_log():
    seek_from = int(request.args.get('seekFrom', 0))
    response = dd_service.get_log(seek_from)
    return Response(response=json.dumps(response), mimetype="application/json", status=200)
