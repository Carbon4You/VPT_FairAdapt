for lr in 0.1; do
    for wd in 0.00001; do
        for pt in 10; do
            # for subset in ALL MALE FEMALE; do
            for subset in ALL; do
                bash scripts/launch.sh config/slurm/VPT_ViT/imagenet_CelebA_${subset}.yaml \
                    "MODEL.PROMPT.NUM_TOKENS ${pt} \
                    SOLVER.LEARNING_RATE ${lr} \
                    SOLVER.WEIGHT_DECAY ${wd}" \
                    "--job-name job_name"
            done
        done
    done
done
