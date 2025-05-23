for lr in 0.5 0.1; do
    for wd in 0.00001; do
        for pt in 10 25 50 100 150 200; do
            for gi in 1 5 10; do
                for subset in ALL; do
                    bash scripts/launch.sh config/slurm/GVPT_MoCoV3/chestxray_CheXpert_MIMIC_${subset}.yaml \
                        "MODEL.PROMPT.NUM_TOKENS ${pt} MODEL.GATED.PROMPT.NUM_TOKENS ${pt} MODEL.GATED.PROMPT.GATE_INIT ${gi} \
                        SOLVER.LEARNING_RATE ${lr} \
                        SOLVER.WEIGHT_DECAY ${wd}" \
                        "--job-name job_name"
                done
            done
        done
    done
done
