#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name="CRSLab-ReDial-CCD-v0"
#SBATCH --output=slurm_jobs/outputs/redial/log.out
#SBATCH --error=slurm_jobs/outputs/redial/log.err
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

srun python run_crslab.py --config config/crs/redial/ccd.yaml --tensorboard