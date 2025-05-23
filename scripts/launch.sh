#!/bin/bash

export SLURM_JOB_ID=0

INPUT_CONFIG_FILE_ARG="${1}"
INPUT_EXTRA_CONFIG_ARGS="${2}"
INPUT_SBATCH_ARGS="${3}"

THIS_SCRIPT="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

python3.12 -u main.py --config_file ${INPUT_CONFIG_FILE_ARG} ${INPUT_EXTRA_CONFIG_ARGS} COMMAND_CHECK_IF_COMPLETED True MULTIPROCESSING_DISTRIBUTED False
CODE_RET=$?
echo Completion code $CODE_RET 

if [[ $CODE_RET == 58 ]]
then

echo Validation or Test results already exist for this experiment.
echo Experiment : sbatch ${INPUT_SBATCH_ARGS} scripts/launch.sbatch \
    ${INPUT_CONFIG_FILE_ARG} \
    "${INPUT_EXTRA_CONFIG_ARGS}" \
    data/ \
    "${INPUT_SBATCH_ARGS}" \
    "${THIS_SCRIPT}/launch.sh" 

else

echo Exiting...
sbatch ${INPUT_SBATCH_ARGS} scripts/launch.sbatch \
    ${INPUT_CONFIG_FILE_ARG} \
    "${INPUT_EXTRA_CONFIG_ARGS}" \
    data/ \
    "${INPUT_SBATCH_ARGS}" \
    "${THIS_SCRIPT}/launch.sh" 

fi