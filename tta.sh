#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=mice_interpret   # sensible name for the job
#SBATCH --mem=32G                 # Default memory per CPU is 3GB.
#SBATCH --partition=gpu # Use the verysmallmem-partition for jobs requiring < 10 GB RAM.
#SBATCH --gres=gpu:1
#SBATCH --mail-user=ngochuyn@nmbu.no # Email me when job is done.
#SBATCH --mail-type=FAIL
#SBATCH --output=outputs/interpret-%A.out
#SBATCH --error=outputs/interpret-%A.out

# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

## Code
# If data files aren't copied, do so
#!/bin/bash
if [ $# -lt 2 ];
    then
    printf "Not enough arguments - %d\n" $#
    exit 0
    fi

# if [ ! -d "$TMPDIR/$USER/hn_delin" ]
#     then
#     echo "Didn't find dataset folder. Copying files..."
#     mkdir --parents $TMPDIR/$USER/hn_delin
#     fi

# for f in $(ls $PROJECTS/ngoc/datasets/headneck/*)
#     do
#     FILENAME=`echo $f | awk -F/ '{print $NF}'`
#     echo $FILENAME
#     if [ ! -f "$TMPDIR/$USER/hn_delin/$FILENAME" ]
#         then
#         echo "copying $f"
#         cp -r $PROJECTS/ngoc/datasets/headneck/$FILENAME $TMPDIR/$USER/hn_delin/
#         fi
#     done


echo "Finished seting up files."

# Hack to ensure that the GPUs work
nvidia-modprobe -u -c=0

# Run experiment
# export ITER_PER_EPOCH=200
# export NUM_CPUS=4
export RAY_ROOT=$TMPDIR/ray
# export MAX_SAVE_STEP_GB=0
# rm -rf $TMPDIR/ray/*
# singularity exec --nv deoxys.sif python interpretability.py $1 $PROJECTS/KBT/mice/perf/$2 --temp_folder $SCRATCH_PROJECTS/KBT/mice/perf/$2 --analysis_folder $SCRATCH_PROJECTS/KBT/mice/perf/$2 ${@:3}

fold_list=$3
for fold in ${fold_list//,/ }
do
    singularity exec --nv deoxys.sif python interpretability.py \
    "$1_$fold.json $PROJECTS/KBT/mice/perf/$2_$fold" \ 
    "--temp_folder $SCRATCH_PROJECTS/KBT/perf/$2_$fold" \ 
    "--analysis_folder $SCRATCH_PROJECTS/KBT/perf/$2_$fold" \
    ${@:3}

    
done

# test_fold_idx=${fold: -1}

# singularity exec --nv deoxys.sif python -u ensemble_results.py $PROJECTS/KBT/mice/perf/$2_ $3 --merge_name test_fold$test_fold_idx