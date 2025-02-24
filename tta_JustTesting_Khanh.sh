#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=tta_uncertainty   # sensible name for the job
#SBATCH --mem=32G                 # Default memory per CPU is 3GB.
#SBATCH --partition=gpu # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --gres=gpu:1
#SBATCH --mail-user=khanh.phuong.le@nmbu.no # Email me when job is done.
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/interpret-%A.out
#SBATCH --error=outputs/interpret-%A.out

# Load necessary modules
module load singularity

## Code
# If data files aren't copied, do so
#!/bin/bash
if [ $# -lt 2 ];
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
# singularity exec --nv deoxys.sif python tta_JustTesing_Khanh.py $1 $Projects/ngoc/CoxaAI/perf/$2 --temp_folder $SCRATCH_PROJECTS/ngoc/CoxaAI/perf/$2 --analysis_folder $SCRATCH_PROJECTS/ngoc/CoxaAI/perf/$2 ${@:3}



singularity exec --nv deoxys.sif python tta_JustTesting_Khanh.py \$1 $PROJECTS/ngoc/CoxaAI/perf/$2 --temp_folder $SCRATCH_PROJECTS/ngoc/CoxaAI/perf/$2 --analysis_folder $SCRATCH_PROJECTS/ngoc/CoxaAI/perf/$2 \${@:3}
 
