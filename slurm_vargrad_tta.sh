#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=CoxaAI_vargrad   # sensible name for the job
#SBATCH --mem=32G                 # Default memory per CPU is 3GB.
#SBATCH --partition=gpu # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=ngoc.huynh.bao@nmbu.no # Email me when job is done.
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/vargrad-%A.out
#SBATCH --error=outputs/vargrad-%A.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

## Code
# If data files aren't copied, do so
#!/bin/bash
if [ $# -lt 1 ];
    then
    printf "Not enough arguments - %d\n" $#
    exit 0
    fi


echo "Finished seting up files."

# Hack to ensure that the GPUs work
nvidia-modprobe -u -c=0

# Run experiment
export MAX_SAVE_STEP_GB=0
export ITER_PER_EPOCH=128
export NUM_CPUS=4
export TMPDIR=/home/work
export RAY_ROOT=$TMPDIR/$USER/ray
singularity exec --nv deoxys_2024.sif python -u interpretability_vargrad_v2.py $PROJECTS/ngoc/CoxaAI/perf/transfer_v2/$1 --temp_folder $SCRATCH_PROJECTS/ceheads/CoxaAI/transfer_v2/$1 --analysis_folder $SCRATCH_PROJECTS/ngoc/CoxaAI/perf/$1 ${@:2}
