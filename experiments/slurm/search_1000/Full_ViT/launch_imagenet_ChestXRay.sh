for lr in 0.01 0.05 0.1 0.5 1.0 2.5 5.0 10.0; do
    for wd in 0.1 0.01 0.001 0.0001 0.00001 0.0; do
        for subset in ALL WHITE ASIAN BLACK; do
            bash scripts/launch.sh config/slurm_1k/Full_ViT/imagenet_CheXpert_MIMIC.yaml \
                "OUTPUT_DIR "./data/search_1k/" MODEL.SAVE_CHECKPOINT False CHECKPOINT_LINKING False DATASET.TEST False DATASET.SUBSET ${subset} \
                SOLVER.LEARNING_RATE ${lr} \
                SOLVER.WEIGHT_DECAY ${wd}" \
                "--job-name job_name"
        done
    done
done