#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=CoxaAI_interpretability   # sensible name for the job
#SBATCH --mem=32G                 # Default memory per CPU is 3GB.
#SBATCH --partition=gpu # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --gres=gpu:1
#SBATCH --mail-user=khanh.phuong.le@nmbu.no # Email me when job is done.
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/interpretability-%A.out
#SBATCH --error=outputs/interpretability-%A.out

# Load necessary modules
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
# export ITER_PER_EPOCH=200
# export NUM_CPUS=4
export RAY_ROOT=$TMPDIR/ray
# export MAX_SAVE_STEP_GB=0
# rm -rf $TMPDIR/ray/*
singularity exec --nv deoxys.sif python interpretability_gradcam.py $PROJECTS/ngoc/CoxaAI/perf/pretrain/$1 --temp_folder $SCRATCH_PROJECTS/ngoc/CoxaAI/perf/$1 --analysis_folder $SCRATCH_PROJECTS/ngoc/CoxaAI/perf/$1 ${@:2}
