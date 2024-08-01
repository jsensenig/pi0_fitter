#!/bin/bash

RUN_FILE=/home/jon/work/protodune/analysis/pi0_reco/code/pi0_fitter/run_pi0_fitter.py
IN_FILE=/home/jon/work/protodune/analysis/pi0_reco/data/1gev_ana_files/subset
OUT_DIR=/home/jon/work/protodune/analysis/pi0_reco/batch/1gev_run_revised_fiducial_and_histq

# Check output dir exists, if not create it
[ ! -d ${OUT_DIR} ] && mkdir -p ${OUT_DIR}

#for i in {0..9}
#do
#    echo "Submitting job ${i} "
#    srun -p gpu --nodelist gpu2 python -u ${RUN_FILE} ${IN_FILE}${i} ${OUT_DIR}/fit_result_${i} &
#
#done


#SBATCH --job-name=1gev_pi0_reco           	# Job name
#SBATCH --ntasks=1                		# Run on a single CPU
#SBATCH --mem=8gb                 		# Job memory request
#SBATCH --partition=gpu                 	# Job partition
#SBATCH --gres=gpu                 	        # GPU node
#SBATCH --output=1gev_pi0_reco_%j.log   	# Standard output and error log
#SBATCH --array=1-10%10                         # Submit 10 jobs, running 10 at a time

python -u ${RUN_FILE} ${IN_FILE}${SLURM_ARRAY_TASK_ID} ${OUT_DIR}/fit_result_${SLURM_ARRAY_TASK_ID} &

