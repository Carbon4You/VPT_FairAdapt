for lr in 1.0 0.5 0.1; do
    for wd in 0.001 0.0001 0.00001; do
        for subset in ASIAN BLACK; do
            bash scripts/launch.sh config/slurm/Full_MAE/chestxray_CheXpert_MIMIC_${subset}.yaml \
                "SOLVER.LEARNING_RATE ${lr} \
                SOLVER.WEIGHT_DECAY ${wd}" \
                "--job-name job_name"
        done
    done
done
