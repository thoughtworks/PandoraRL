# Reinforcement Learning based Virtual Screening

RL based virtual screening
Prerequisite
- `conda`


## Set up the server on your local machine
1. Setup rl-virtual-screening conda environment

    ```
    $ ./scripts/setup.sh
    $ source activate rl-virtual-screening
    ```
    
2. Run Application
 * Build web
    ```
        $ cd web
        $ npm install
        $ npm run build
     ```
 * Running pipeline through FLASK.
   * Go to source folder rl-virtual-screening
    ```
    $ cd ..
    $ FLASK_APP=src/main/app.py flask run
    ```
    * Open the url shown on terminal output, for example: http://127.0.0.1:5000/
    * "RL Based Ligand Pose Prediction" will open in your browser

## Generate Ligand Pose

### Upload a ligand file
- Upload the ligand you want to generate a pose for in .pdbqt or .mol2 format (prepared ligand) or .pdb format (unprepared ligand)
- Alternatively, to enter the SMILE string, click the checkbox "Smiles String" and enter the string in the textbox that appears.

### Upload a protein file 
- Upload the prepared protein file in .pdbqt or .mol2 format

### Submit inputs and generate pose
- Click the "Generate Ligand Pose" button after entering your inputs.
- A green notification will display your job ID.
- An entry for your job will appear under "Jobs"
- You can view the logs for your job in a new window by clicking on the log file link. While the job is running, keep refreshing the log file page to update the logs.
- NOTE: It will take some time for necessary imports and modules to be loaded, meanwhile the log will remain empty for a few seconds initially.
- While the job is running, clicking on the output file will display "The output file is not yet generated. Please check in some time.". Go back to the main page with the "Back to Application" link. 

### Download output file (.pdb)
- When the algorithm has finished, the output file with ligand in generated pose will be downloadable.  
- Click the output file and save it to your local machine. 
