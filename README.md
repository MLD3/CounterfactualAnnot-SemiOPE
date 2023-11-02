# Counterfactual-Augmented Importance Sampling for Semi-Offline Policy Evaluation

This repository contains the source code for replicating all experiments in the NeurIPS 2023 paper, "Counterfactual-Augmented Importance Sampling for Semi-Offline Policy Evaluation".

Repository content:

- `synthetic/` contains code for the experiments on the toy bandit problems. 
- `sepsisSim/` contains code for the experiments on the sepsis simulator.

If you use this code in your research, please cite the following publication:
```
@inproceedings{tang2023counterfactual,
    title={Counterfactual-Augmented Importance Sampling for Semi-Offline Policy Evaluation},
    author={Tang, Shengpu and Wiens, Jenna},
    booktitle={Advances in Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=dsH244r9fA}
}
```

## Synthetic domains - Bandits (Sec 5.1 & Appx E.1)
- `bandit_compare-2state.ipynb`: Table 1, Appx E Table 3
- `bandit_compare-1state.ipynb`: Appx E Table 4
- `bandit_sweepW.ipynb`: Appx E Fig 7 (varying weights)
- `bandit_sweepPcannot.ipynb`: Appx E Fig 8 (varying percent annotated and imputation)

## Healthcare domain - Sepsis simulator (Sec 5.2 & Appx E.2)
- Simulator based on publicly available code at https://github.com/clinicalml/gumbel-max-scm/tree/sim-v2
- Experiment code structure are inspired by https://github.com/MLD3/OfflineRL_ModelSelection and https://github.com/MLD3/OfflineRL_FactoredActions
- The preparation steps are in `data-prep/`, which include the simulator source code as well as several notebooks for dataset generation. The output is saved to `data/` (ground-truth MDP parameters, ground-truth optimal policy, and optimal value functions) and `datagen/` (offline datasets). 
- The code for the main experiments is in `experiments/`. 
    - `0-prepopulate_annotations.ipynb` preprocesses the data so that all counterfactual annotations are pre-populated. This step is run only once (instead of repeated for each experiment) as it takes a long time. The annotations are later replaced with different values or removed depending on the experiment. The output is saved to `results/vaso_eps_0_1-annotOpt_df_seed2_aug_step.pkl`.
    - Use `commands.sh` to run the experiments.
    - Run the following notebooks to generate tables and figures used in the paper:
        - `results.ipynb`: Table 2, Appx E Table 5 top
        - `results-OIS-WIS.ipynb`: Appx E Table 5 bottom
        - `plots-analyses.ipynb`: Fig 5 left, Appx E Fig 9, Appx E Fig 10
        - `plots-noisy[-v2].ipynb`: Fig 5 center, Appx E Fig 11
        - `plots-missing-[v2].ipynb`: Fig 5 right, Appx E Fig 12
        - `plots--legend.ipynb`: figure legend used in Fig 5, Fig 11, and Fig 12
    - `fig/` contains the main figures used in the paper.
