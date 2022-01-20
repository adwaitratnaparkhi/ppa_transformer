# Introduction

This repository contains scripts to create transformer-based deep learning models for the prepositional phrase attachment problem, as described in:

Adwait Ratnaparkhi and Atul Kumar. (2021). Resolving Prepositional Phrase Attachment Ambiguities with Contextualized Word Embeddings. In Proceedings of 
ICON-2021: 18th International Conference on Natural Language Processing. National Institute of Technology, Silchar, India. December 16-19, 2021


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

# from pytorch.org
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 

pip install transformers==4.2.1

pip install datasets
```
# Training and evaluating the models

The following shell script will reproduce the experiments in the above paper. It will call the scripts in various sub-directories. Make sure you have built the BLBG data before calling this script!
```
cd scripts
./run_all.sh
```
Results will be stored in a directory named `results_XXXXXXX` in the sub-directory, in a file named `test_results.txt`

# References

[1] Yonatan Belinkov, Tao Lei, Regina Barzilay, and Amir Globerson. (2014). Exploring compositional architectures and word vector representations for prepositional phrase attachment. Transactions of the Association for Computational Linguistics, 2:561–572.

[2] Adwait Ratnaparkhi, Jeff Reynar, and Salim Roukos. (1994). A maximum entropy model for prepositional phrase attachment. In HUMAN LANGUAGE TECHNOLOGY: Proceedings of a Workshop held at Plainsboro, New Jersey, March 8-11, 1994.


