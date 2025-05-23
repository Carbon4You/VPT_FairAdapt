for lr in 1.0 0.5 0.1; do
    for wd in 0.001 0.0001 0.00001; do
        for subset in Male FEMALE; do
            bash scripts/launch.sh config/slurm/Full_ViT/imagenet_CelebA_${subset}.yaml \
                "SOLVER.LEARNING_RATE ${lr} \
                SOLVER.WEIGHT_DECAY ${wd} \
                CHECKPOINT_LINKING False \
                DATASET.VALIDATE True \
                DATASET.TEST True \
                DATASET.VALIDATION_FREQUENCY 99 \
                DATASET.TEST_FREQUENCY 99 \
                MODEL.SAVE_FREQUENCY 5" \
                "--job-name job_name"
        done
    done
done

