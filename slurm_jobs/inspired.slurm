#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50G
#SBATCH --job-name="CRSLab-INSPIRED-CCD-v0"
#SBATCH --output=slurm_jobs/outputs/inspired/log.out
#SBATCH --error=slurm_jobs/outputs/inspired/log.err
#SBATCH --mail-user=eirsteir@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

module purge
module load  Python/3.9.6-GCCcore-11.2.0

python -m pip install --upgrade pip
pip install virtualenv
virtualenv env
source env/bin/activate

pip install torch torchvision torchaudio torch_geometric protobuf~=3.20

pip install -e .

echo "Running Evaluation"

nvidia-smi

srun python run_crslab.py --config config/crs/inspired/ccd.yaml --gpu 0 --tensorboard