# use this for the RRR1994 data

import sys
import json

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("args: [RRR data file]")
        sys.exit(1)

    ppdataFile = open(sys.argv[1], 'r')

    ppdata = ppdataFile.readlines()

    for ppinstance  in ppdata:
        example = {}

        (num, v, n, p, n2, label) = ppinstance.rstrip().split()

        #
        # V --> label 0
        # N --> label 1
        #
        assert(label == "N" or label == "V")
        example["label"] = 0 if label == "V" else 1

        example["attach0"] = f"{n} {p} {n2}"
        example["attach1"] = f"{v} {p} {n2}"

        example["sent1"] = f"{v} {n}"
        example["sent2"] = ""

        print(json.dumps(example))
