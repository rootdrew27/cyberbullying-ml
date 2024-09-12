# Experiment 1

This experiment tests the ability of the xgboost model to generalize to new data.
**Specifically** when trained upon the dataset from [1], which was curated using a Dyanmic Query Expansion (DQE) method, as detailed in [1]. Importantly, an unkown fraciton of this dataset is curated with DQE.

Training and Validation is done using Dataset 1. Testing is done using Dataset 2.

There are 2 versions of this experiment: one in which training is relabeled such that the specific cyberbullying type is ignored, and the other keeps the specific cyberbullying label whilst relabeling for the purpose of testing.

### Steps:


[1] https://people.cs.vt.edu/ctlu/Publication/2020/IEEE-BD-SOSNet-Wang.pdf
[2] https://arxiv.org/pdf/1703.04009


