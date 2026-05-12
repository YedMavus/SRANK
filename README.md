# SRank: Towards Semantic-Aware Ranking-Based Evaluation for Continual Learning of Vision-Language Models

This is the official GitHub repository for the implementation of the SRANK metric from the paper "SRANK: TOWARDS SEMANTIC-AWARE RANKING-BASED EVALUATION FOR CONTINUAL LEARNING OF VISION-LANGUAGE MODELS" accepted to ICIP 2026.
Authors: [@YedMavus](https://github.com/YedMavus), [@debarshigit](https://github.com/debarshigit) and Prof. Soma Biswas

This repository contains three independent codebases corresponding
to three different papers in which SRank evaluation was done:

- DIKI
- LADA
- RAIL

Each folder is self-contained and includes its own implementation
of the evaluation metric.


The code for the SRANK metric has been added to utils.py or in line for different methods.
After setting up the datasets as described in LADA, run label_map.py to obtain a .csv file containing all labels present in all datasets altogether. Next, run semantic_eval.py to obtain the rankings based on these classnames.


To Do:

- [ ] Instructions and code to adapt SRANK to custom benchmarks

# Acknowledgements

We thank the authors of [DIKI](https://github.com/lloongx/DIKI) and [XTAIL](https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning) for releasing their codes, and [LADA](https://github.com/maolinluo/LADA) for releasing their code as well as making the datasets available.
