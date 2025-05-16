#!/bin/bash

# useful for debugging/determinism
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

# SLURM_PROCID is evaluated within this script, this is why this script exists instead of doing 
# everything in launch.slurm 
if [[ "$DISTRIBUTED_CONFIG" != "none" ]]
then
    export LAUNCHER="accelerate launch \
        --config_file /leonardo_work/IscrC_LLM-REC/ale/configs/${DISTRIBUTED_CONFIG}_config.yaml \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --num_processes $(( $NTASKS_PER_NODE * $COUNT_NODE)) \
        --num_machines $COUNT_NODE \
        --machine_rank $SLURM_PROCID \
        "
else
    if [[ "$LAUNCHER_PAR" == "torchrun" ]]
    then
        export LAUNCHER="torchrun --nproc_per_node=$NTASKS_PER_NODE --master_port 25678"
    fi
    if [[ "$LAUNCHER_PAR" == "deepspeed" ]]
    then
        export LAUNCHER="deepspeed --num_gpus $GPU_PER_NODE --master_port 25678"
    else
        # https://stackoverflow.com/questions/76007288/how-to-check-if-system-is-installed-with-latest-python-version-python3-10
        PYTHON_VERSION=`ls /usr/bin/python* | grep -o '[0-9]\+\.[0-9]\+' | tr -d '.' | sort -n | tail -n 1`
        PYTHON_VERSION="${PYTHON_VERSION:0:1}"."${PYTHON_VERSION:1:3}"
        export LAUNCHER="python${PYTHON_VERSION}"
    fi
fi

CMD="$LAUNCHER $SCRIPT_TO_RUN"
$CMD
