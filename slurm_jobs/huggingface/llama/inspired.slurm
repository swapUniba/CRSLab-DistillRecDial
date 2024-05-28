#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --job-name=CRSLab-LLAMA-3-CCD-v0
#SBATCH --output=slurm_jobs/outputs/llama/log.out
#SBATCH --error=slurm_jobs/outputs/llama/log.err
#SBATCH --mail-user=eirsteir@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

module purge
module load Python/3.9.6-GCCcore-11.2.0

python -m pip install --upgrade pip
pip install virtualenv
virtualenv env
source env/bin/activate

pip install torch torchvision torchaudio torch_geometric protobuf~=3.20

pip install -e .

echo "Running Evaluation"

HF_TOKEN=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

nvidia-smi

srun python run_crslab.py --config config/crs/huggingface/llama/ccd.yaml --gpu 0 --tensorboard