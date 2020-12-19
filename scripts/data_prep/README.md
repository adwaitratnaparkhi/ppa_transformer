# Introduction

The scripts in this directory re-build a superset of the data from (Belinkov et al, 2014), extract additional information from the raw sentence, and re-join that information with the original examples. 

The `extract_pp_attach_for_matlab.py` script, originally from (Belinkov et al, 2014), was modified in the following ways:
* Ignore the word vector filter when extracting heads
* Print the full sentence context in a separate file
* Print the index of the heads in a separate file

As this script does not have the original word vector file used in (Belinkov et al, 2014), it cannot produce the exact same training and test instances. 
It produces more since it does not use a filter.

The `join_sentence_from_rebuilt_data.py` will match the full sentence context from the rebuilt data to the original examples. We use the original examples and joined data for experiments. 

[`pennconverter.jar`](http://nlp.cs.lth.se/software/treebank_converter) is used by `tb3_to_dependency` to create the Penn Treebank 3 dependency format.


# Prerequisites

* Obtain the Penn Treebank Release 3 from the [Linguistic Data Consortium](www.ldc.com)
* Install the data from (Belinkov et al, 2014)
   * `git clone https://github.com/boknilev/pp-attachment`

# Creating the data

To create training and test data:
* Edit the `PENN_TREEBANK_DIR` and `BLBG_DIR` environment variables in `mk_data.sh` to reflect the installation paths of the Penn Treebank Release 3, and the code/data repository of (Belinkov, 2014).
* Run `mk_data.sh`
   * It will store temporary files in the current directory
 

