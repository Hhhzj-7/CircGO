# CircGO
CircGO: Predicting circRNA Functions through Self-Supervised Learning of Heterogeneous Networks

## Overview
we present a computational method named CircGO for circRNA GO function prediction.  We first integrate circRNA co-expression, circRNA-protein associations, and protein-protein interactions to construct the circRNA-protein heterogeneous network. We employ the self-supervised pre-training that combines walking, aggregation, and clustering to comprehensively extract potential information from the circRNA-protein heterogeneous network. Utilizing Hin2vec and Label Propagation Algorithm (LPA), we initialize node features and pseudo-labels. A heterogeneous graph attention network and LPA with an attention mechanism are employed for self-supervised pre-training on the circRNA-protein heterogeneous graph. Finally, the initialized and pre-trained node features are utilized to train a GO term predictor, enabling the generalization of protein functions to circRNA nodes in the heterogeneous graph.

## Environment
`You can create a conda environment for CircGO by ‘conda env create -f environment.yml‘.`

## Data
Because the file size exceeds the limit, the data can be downloaded from [data](https://drive.google.com/file/d/1vwsrsj9DghGTUy_poHATS5N9Jh0KQzYV/view?usp=drive_link).

## Pretrain and Finetune
You can pretrain CircGO and save the model parameters by ‘python main.py‘.
You can load the model parameters, predict CircRNA function and then evaluate CircGO by ‘python predict.py‘.

## Contact
If you have any issues or questions about this paper or need assistance with reproducing the results, please contact me.

Zhijian Huang

School of Computer Science and Engineering,

Central South University

Email: zhijianhuang@csu.edu.cn
