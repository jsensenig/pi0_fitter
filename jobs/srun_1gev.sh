#!/bin/bash

RUN_FILE=/home/jon/work/protodune/analysis/pi0_reco/code/pi0_fitter/run_pi0_fitter.py
IN_FILE=/home/jon/work/protodune/analysis/pi0_reco/data/1gev_ana_files/subset
OUT_DIR=/home/jon/work/protodune/analysis/pi0_reco/batch/1gev_run_normal_dnda_old_fv

# Check output dir exists, if not create it
[ ! -d ${OUT_DIR} ] && mkdir -p ${OUT_DIR}

for i in {0..9}
do
    echo "Submitting job ${i} "
    srun -p gpu --gres=gpu python -u ${RUN_FILE} ${IN_FILE}${i} ${OUT_DIR}/fit_result_${i} &
done


