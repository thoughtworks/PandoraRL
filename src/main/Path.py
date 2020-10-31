import os
import time


class Path:
    MAX_STEPS = 20
    ARTIFACT_FOLDER_PATH = "./artifacts"
    PROTEIN_FILE_PATH = ARTIFACT_FOLDER_PATH + "/proteinFile"
    LIGAND_FILE_PATH = ARTIFACT_FOLDER_PATH + "/ligandFile"
    LOG_FOLDER_PATH = "./log"
    TESTING_LOG_FILE_PATH = LOG_FOLDER_PATH+"/testing_logfile.log"
    METDATA_PATH = "./metadata.json"

    @staticmethod
    def create_artifact_path():
        if not os.path.exists(Path.ARTIFACT_FOLDER_PATH):
            os.makedirs(Path.ARTIFACT_FOLDER_PATH)

    @staticmethod
    def create_log_path():
        if not os.path.exists(Path.LOG_FOLDER_PATH):
            os.makedirs(Path.LOG_FOLDER_PATH)
