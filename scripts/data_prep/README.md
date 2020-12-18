This directory contains scripts that can re-build the data from (Belinkov et al, 2014)

`extract_pp_attach_for_matlab.py`, `utils.py`, and `sentence.py` are from the scripts used by (Belinkov et al, 2014): https://github.com/boknilev/pp-attachment/blob/master/scripts

The `extract_pp_attach_for_matlab.py` script was modified to ignore the word vector filter when extracting heads, to print the full sentence context in another file, and to read and write from local file paths.
As this script does not have the original word vector file used in (Belinkov et al, 2014), it cannot produce the exact same training and test instances. 
It will produce more since it does not use a filter. 

`pennconverter.jar` has been copied from http://nlp.cs.lth.se/software/treebank_converter/, and is used by `tb3_to_dependency` to create the Penn Treebank 3 dependency format.


To create training and test data:
* Edit the path variable in `tb3_to_dependency.sh` to reflect your Penn Treebank 3 installation. 
* Confirm the output path in `extract_pp_attach_for_matlab.py`
* Run `mk_data.sh`
   * It will store temporary files in the current directory
 

