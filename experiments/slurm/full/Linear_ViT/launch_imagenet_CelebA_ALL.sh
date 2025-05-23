for lr in 1.0 0.5 0.1; do
    for wd in 0.001 0.0001 0.00001; do
        for subset in ALL; do
            bash scripts/launch.sh config/slurm/Linear_ViT/imagenet_CelebA_${subset}.yaml \
                "SOLVER.LEARNING_RATE ${lr} \
                        SOLVER.WEIGHT_DECAY ${wd}" \
                "--job-name job_name"
        done
    done
done
