# Reinforcement Learning based Virtual Screening

RL based virtual screening
Prerequisite
- `conda`


In order to use this library
```
https://gitlab.com/lifesciences/rl-virtual-screening.git
cd rl-virtual-screening
```

####SetUp
1. Setup rl-virtual-screening conda environment

    ```
    $ ./scripts/setup.sh
    ```
    
    ```
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
    $ FLASK_APP=src/main/app.py flask run
    ```
