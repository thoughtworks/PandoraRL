import hashlib
import json
import os
import subprocess
import time

from src.main.Path import Path


class DDService:

    def __init__(self):
        Path.create_artifact_path()
        Path.create_log_path()

    def init(self):
        if not os.path.exists(Path.METDATA_PATH):
            with open(Path.METDATA_PATH, 'w') as metadata_file:
                init = {"next_count": 1, "details": {}}
                json.dump(init, metadata_file)
        with open(Path.METDATA_PATH, 'r') as metadata_file:
            data= json.load(metadata_file)
        return data["details"]

    def triggerRLAgent(self, protein_file, ligand_file, string_format):
        self.init()
        job_details, job_id = self.record_job(protein_file, ligand_file, string_format)
        protein_file.save(job_details["protein_file_path"])
        if not string_format == "true":
            ligand_file.save(job_details["ligand_file_path"])
        testing_process_id = self.run_testing_script(job_details)
        return {"job_id":job_id}

    def record_job(self, protein_file, ligand_file, string_format):
        time_str = time.strftime('%H_%M_%S__%Y_%m_%d')
        with open(Path.METDATA_PATH, 'r') as metadata_file:
            job_details = json.load(metadata_file)
        job_id = job_details["next_count"]
        details = job_details["details"]
        # ligand_name_pose.pdb
        if string_format == "true":
            temp = {"ligand_input_type": "smiles_string", "ligand_file_name": None, "ligand_input": ligand_file,
                    "output_path": os.path.abspath(f"./Results/pose_{Path.MAX_STEPS}_{time_str}.pdb")}
        else:
            ligand_file_name = ligand_file.filename
            ligand_file_name_wo_ext = ligand_file_name.split(".")[0]
            temp = {"ligand_input_type": "file", "ligand_file_name": ligand_file_name,
                    "ligand_file_path": f"{Path.ARTIFACT_FOLDER_PATH}/protein_{time_str}_{ligand_file.filename}",
                    "output_path": os.path.abspath(f"./Results/{ligand_file_name_wo_ext}_pose_{Path.MAX_STEPS}_{time_str}.pdbqt")}
        job = {"protein_file_name": protein_file.filename,
               "protein_file_path": f"{Path.ARTIFACT_FOLDER_PATH}/protein_{time_str}_{protein_file.filename}",
               "log_path": f'{Path.LOG_FOLDER_PATH}/testing_logfile{time_str}.log',
               }
        job.update(temp)
        details[job_id] = job
        updated = {"next_count": job_id + 1, "details": details}
        with open(Path.METDATA_PATH, 'w') as metadata_file:
            json.dump(updated, metadata_file)
        return job, job_id

    def run_testing_script(self, job_details):
        output_fd = os.open(job_details["log_path"], os.O_RDWR | os.O_APPEND | os.O_CREAT)
        process = subprocess.Popen(['python', 'testing_script.py', str(job_details)], stdout=output_fd, stderr=output_fd)
        return process.pid

    def get_log(self, seek_from):
        try:
            with open(Path.TESTING_LOG_FILE_PATH) as log_file:
                log_file.seek(seek_from)
                data = log_file.read()
                seek_from = log_file.tell()
        except FileNotFoundError:
            raise Exception('Logs not yet present')
        return {'data': data, 'last_read_byte': seek_from}