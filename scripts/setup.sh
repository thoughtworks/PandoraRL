#!/usr/bin/env bash
set -e
export PATH="${PATH}:/root/miniconda3/bin"
CONDA_DEVELOPMENT_ENV_NAME=rl-virtual-screening-gpu

echo "*********** ${CONDA_DEVELOPMENT_ENV_NAME} environment cleanup **************"
environment_lists=($(conda env list | cut -d ' ' -f1))
if [[ " ${environment_lists[*]} " == *" $CONDA_DEVELOPMENT_ENV_NAME "* ]]; then
    conda env remove -y -n ${CONDA_DEVELOPMENT_ENV_NAME}
else
    echo "*********** Nothing to cleanup ***********"
fi
echo "*********** ${CONDA_DEVELOPMENT_ENV_NAME} environment cleanup completed **************"

echo "*********** creating ${CONDA_DEVELOPMENT_ENV_NAME} environment **************"
conda env create -f environment-gpu.yml
echo "*********** ${CONDA_DEVELOPMENT_ENV_NAME} environment created **************"
echo "*********** Installing ${CONDA_DEVELOPMENT_ENV_NAME} dependencies **************"
source activate ${CONDA_DEVELOPMENT_ENV_NAME}
conda install -y -c pytorch pytorch=1.9.1
echo "*********** Finished setup **************"

