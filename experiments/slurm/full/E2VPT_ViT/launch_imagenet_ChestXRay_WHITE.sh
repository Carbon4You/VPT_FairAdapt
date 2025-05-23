for lr in 0.5 0.1 0.05; do
    for wd in 0.001 0.00001; do
        for pt in 10 25 50 100 150 200; do
            for a_pt in 10 25 50 100; do
                for subset in WHITE; do
                    bash scripts/launch.sh config/slurm/E2VPT_ViT/imagenet_CheXpert_MIMIC_${subset}.yaml \
                        "MODEL.E2VPT.KV_PROMPT.NUM_TOKENS_P ${pt} \
                        MODEL.E2VPT.KV_PROMPT.NUM_TOKENS ${a_pt} \
                        SOLVER.LEARNING_RATE ${lr} \
                        SOLVER.WEIGHT_DECAY ${wd}" \
                        "--job-name job_name"
                done
            done
        done
    done
done
