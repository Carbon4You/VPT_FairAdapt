>📋  A template README.md for code accompanying a Machine Learning paper

# Visual Prompt Tuning under Demographic Imbalance: A Fairness and Utility Analysis

## Requirements

### Dataset
* The created splits can be found under "data/datasets/".
* CelebA: Once downloaded can be used by pointing the configuration paths to the dataset.
* CheXpert & MIMIC-CXR: The datasets are protected, therefore we provide a redacted version that only includes paths and splits which then can be merged with official dataset to obtain labels, and demographics. 

### Foundation Models
* ImageNet models can be found under code references.
* Chest X-Ray models we trained will be made public at a later point.

### To install requirements:

```setup
pip install -r requirements.txt
```

## Training

The experiments were done using slurm workload manager. Once the dataset and models are configured they can be started using scripts under experiments directory.

# Code References
Code was taken in parts from below repos.
* [VPT](https://github.com/KMnP/vpt) 
* [GVPT](https://github.com/ryongithub/GatedPromptTuning)
* [E2VPT](https://github.com/ChengHan111/E2VPT/tree/new_branch)
* [MoCoV3](https://github.com/facebookresearch/moco-v3)
* [Medical MAE](https://github.com/lambert-x/medical_mae)