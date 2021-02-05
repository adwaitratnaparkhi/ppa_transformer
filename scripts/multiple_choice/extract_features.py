# use this for the BLBG2014 data

import sys
import json
from read_pp_data import read_pp_data

# experiment modes
# 1: Triple only
# 2: Triple  + all heads
# 3: Triple  + sentence
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("args: [BLBG data prefix] [experiment mode = 1,2,3]")
        sys.exit(1)

    ppdata = read_pp_data(sys.argv[1], True)
    exp_mode = int(sys.argv[2])

    if exp_mode < 1 or exp_mode > 3:
        print("unknown experiment mode")
        sys.exit(1)

    max_choices = 8

    for (h, p, ch, l, sent, hi, pi, chi) in ppdata:
        example = {}
        example["label"] = int(l) - 1   # move to 0-based label

        heads = h.split()

        # common to all modes
        for index, head in enumerate(heads):
            triple = head + " " + p + " " + ch
            example["attach" + str(index)] = triple

        # pad the rest of choices
        for index in range(len(heads), max_choices):
            example["attach" + str(index)] = "<pad>"

        if exp_mode == 1:
            example["sent1"] = ""
        if exp_mode == 2:
            example["sent1"] = h
        elif exp_mode == 3:
            example["sent1"] = sent

        example["sent2"] = ""

        print(json.dumps(example))
