# Introduction

This repository contains scripts to create transformer-based deep learning models for the prepositional phrase attachment problem. 

# Building the data

There are two data prepositional phrase attachments sets used by this work:
* Binary classification
   * This is the data set from [(Ratnaparkhi et al, 1994)](https://www.aclweb.org/anthology/H94-1048.pdf). 
   * It is located [here](./data/RRR1994).
* Multi-way classification
   * This data set is derived from the scripts and data in [(Belinkov et al, 2014)](https://www.mitpressjournals.org/doi/pdfplus/10.1162/tacl_a_00203). 
   * The scripts to build it are [here](./scripts/data_prep/README.md)

# Setup
```
conda create -n ppa python=3.8

conda activate ppa

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch # from pytorch.org

pip install transformers==4.2.1

pip install datasets
```
# Training and evaluating the model

* Binary classification
```
cd scripts/binary_classification
./run.sh
# Evaluation stats should be printed to console
```
    
* Multiple choice classification
   * Follow instructions in previous section to build the data, before trying this section
```
cd scripts/multiple_choice
./run.sh
# Evaluation stats should be printed to console
```
   
# Results

* Binary classification
   * Development set: 89.5%
   * Test set: 88.7%
    
* Multiple choice
   * Triple only: 83.1%
   * Triple + heads: 91.5%
   * Triple + full sentence: 94.5%
   
# References

[1] Yonatan Belinkov, Tao Lei, Regina Barzilay, and Amir Globerson. 2014. Exploring compositional architectures and word vector representations for prepositional phrase attachment. Transactions of the Asso- ciation for Computational Linguistics, 2:561–572.

[2] Adwait Ratnaparkhi, Jeff Reynar, and Salim Roukos. 1994. A maximum entropy model for prepositional phrase attachment. In HUMAN LANGUAGE TECHNOLOGY: Proceedings of a Workshop held at Plainsboro, New Jersey, March 8-11, 1994.


