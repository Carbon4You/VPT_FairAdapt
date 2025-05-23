for lr in 0.5 0.1; do
    for wd in 0.001 0.00001; do
        for pt in 10 25 50 100 150 200; do
            for subset in ASIAN BLACK; do
                bash scripts/launch.sh config/slurm/VPT_ViT/imagenet_CheXpert_MIMIC_${subset}.yaml \
                    "MODEL.PROMPT.NUM_TOKENS ${pt} \
                    SOLVER.LEARNING_RATE ${lr} \
                    SOLVER.WEIGHT_DECAY ${wd}" \
                    "--job-name job_name"
            done
        done
    done
done
