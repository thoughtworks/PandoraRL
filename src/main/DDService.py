import os
import subprocess

from src.main.Path import Path


class DDService:

    def __init__(self):
        Path.create_artifact_path()
        Path.create_log_path()

    def triggerRLAgent(self, protein_file, ligand_file):
        protein_file.save(Path.PROTEIN_FILE_PATH)
        ligand_file.save(Path.LIGAND_FILE_PATH)
        testing_process_id = self.run_testing_script()

    def run_testing_script(self):
        output_fd = os.open(Path.TESTING_LOG_FILE_PATH, os.O_RDWR | os.O_APPEND | os.O_CREAT)
        process = subprocess.Popen(['python', 'testing_script.py'], stdout=output_fd, stderr=output_fd)
        return process.pid