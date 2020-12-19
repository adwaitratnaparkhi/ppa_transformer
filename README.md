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

# Training and evaluating the model


# References

[1] Yonatan Belinkov, Tao Lei, Regina Barzilay, and Amir Globerson. 2014. Exploring compositional architectures and word vector representations for prepositional phrase attachment. Transactions of the Asso- ciation for Computational Linguistics, 2:561–572.

[2] Adwait Ratnaparkhi, Jeff Reynar, and Salim Roukos. 1994. A maximum entropy model for prepositional phrase attachment. In HUMAN LANGUAGE TECHNOLOGY: Proceedings of a Workshop held at Plainsboro, New Jersey, March 8-11, 1994.


