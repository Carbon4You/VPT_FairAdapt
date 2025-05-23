#!/bin/bash

#SBATCH --job-name=RUN_SBATCH
#SBATCH --qos=m5
#SBATCH --nodes=1
#SBATCH --gpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40gb
#SBATCH --time=1:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --signal=B:USR1@120
#SBATCH --output=./data/slurm_start_logs/slurm_%j.out
#SBATCH --error=./data/slurm_start_logs/slurm_%j.err
#SBATCH --partition=rtx6000,a40

source ~/.bashrc
source ~/startup.sh

echo LISTING INPUT ARGUMENTS STARTED : 
for i in $*; do 
  echo $i 
done
echo LISTING INPUT ARGUMENTS ENDED : 
which pip3
which python
which python3
echo hostname $(hostname)
date
env
pip3 list

##################################################################
#
#
# Pre-Initialization
#
export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
CURRENT_DATETIME=$(date '+%Y_%m_%d_%H_%M_%S')
#
##################################################################
# Input Arguments
##################################################################
#
INPUT_CONFIG_FILE_ARG="${1}"
INPUT_EXTRA_CONFIG_ARGS="${2}"
INPUT_DATA_DIR_ARG="${3}"
LOG_DIR=${INPUT_DATA_DIR_ARG}/slurm_process_logs
mkdir -p ${LOG_DIR}
INPUT_SBATCH_ARGS="${4}"
INPUT_PARENT_SCRIPT="${5}"
echo trap " ${INPUT_PARENT_SCRIPT} \"${INPUT_CONFIG_FILE_ARG}\" \"${INPUT_EXTRA_CONFIG_ARGS}\" \"${INPUT_SBATCH_ARGS}\" " USR1
#
##################################################################
# Configurations
##################################################################
#
CONFIG="--config_file ${INPUT_CONFIG_FILE_ARG} \
  DIST_URL tcp://$MASTER_ADDR:$MASTER_PORT \
  MULTIPROCESSING_DISTRIBUTED True \
  CHECKPOINT_ROOT /checkpoint/${USER}/ \
  CUDA True \
  SLURM_JOB_ID ${SLURM_JOB_ID} \
  ${INPUT_EXTRA_CONFIG_ARGS}"
#
##################################################################
#
##################################################################
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

export WORLD_SIZE=$SLURM_NTASKS
for ((node_index = 0; node_index < SLURM_JOB_NUM_NODES; node_index++)); do
  node_id=${nodes_array[${node_index}]}
  echo "STARTING WORKER ${node_index} at ${node_id}"

  echo "RESTART COMMAND :" >> ${LOG_DIR}/slurm_${SLURM_JOB_ID}_worker_${node_index}.out 2>&1
  echo "squeue_me.sh | grep ${SLURM_JOB_NAME}" >> ${LOG_DIR}/slurm_${SLURM_JOB_ID}_worker_${node_index}.out 2>&1
  echo "${INPUT_PARENT_SCRIPT} \"${INPUT_CONFIG_FILE_ARG}\" \"${INPUT_EXTRA_CONFIG_ARGS}\" \"${INPUT_SBATCH_ARGS}\"" \
  >> ${LOG_DIR}/slurm_${SLURM_JOB_ID}_worker_${node_index}.out 2>&1

  srun -lN ${node_index} --mem=${SLURM_MEM_PER_NODE}M --gres=gpu:${SLURM_GPUS_ON_NODE} -c ${SLURM_CPUS_ON_NODE} -N 1 -n 1 -r ${node_index} \
    bash -c "python3.12 -u main.py ${CONFIG} RANK ${node_index} WORLD_SIZE ${SLURM_JOB_NUM_NODES}" \
    >> ${LOG_DIR}/slurm_${SLURM_JOB_ID}_worker_${node_index}.out 2>&1 &

  PID=$!

  sleep 1
done

trap " ${INPUT_PARENT_SCRIPT} \"${INPUT_CONFIG_FILE_ARG}\" \"${INPUT_EXTRA_CONFIG_ARGS}\" \"${INPUT_SBATCH_ARGS}\" " USR1

wait
