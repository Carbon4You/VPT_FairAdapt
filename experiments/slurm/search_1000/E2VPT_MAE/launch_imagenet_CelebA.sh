for lr in 0.05 0.1 0.5 1.0 2.5; do
    for wd in 0.001 0.0001 0.00001 0.0; do
        for pt in 5 10 25 50 100 150; do
            for a_pt in 5 10 25 50 100 150; do
                for subset in ALL MALE FEMALE; do
                    bash scripts/launch.sh config/slurm/E2VPT_MAE/imagenet_CelebA_ALL.yaml \
                        "MODEL.E2VPT.KV_PROMPT.NUM_TOKENS_P ${pt} \
                        MODEL.E2VPT.KV_PROMPT.NUM_TOKENS ${a_pt} \
                        DATASET.BATCH_SIZE 64 \
                        OUTPUT_DIR "./data/search_1k/" MODEL.SAVE_CHECKPOINT False CHECKPOINT_LINKING False DATASET.TEST False DATASET.SUBSET ${subset} \
                        SOLVER.LEARNING_RATE ${lr} \
                        SOLVER.WEIGHT_DECAY ${wd}" \
                        "--job-name job_name"
                done
            done
        done
    done
done
