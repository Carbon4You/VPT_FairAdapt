for lr in 5.0 2.5 1.0 0.5 0.1 0.05 0.01; do
    for wd in 0.001 0.0001 0.00001 0.0; do
        for pt in 10 25 50 100 150; do
            for gi in 5 10; do
                for subset in ALL MALE FEMALE; do
                    bash scripts/launch.sh config/slurm/GVPT_ViT/imagenet_CelebA_ALL.yaml \
                        "DATASET.BATCH_SIZE 64 \
                        OUTPUT_DIR "./data/search_1k/" MODEL.SAVE_CHECKPOINT False CHECKPOINT_LINKING False DATASET.TEST False DATASET.SUBSET ${subset} \
                        MODEL.PROMPT.NUM_TOKENS ${pt} \
                        MODEL.GATED.PROMPT.NUM_TOKENS ${pt} \
                        MODEL.GATED.PROMPT.GATE_INIT ${gi} \
                        SOLVER.LEARNING_RATE ${lr} \
                        SOLVER.WEIGHT_DECAY ${wd}" \
                        "--job-name job_name"
                done
            done
        done
    done
done
