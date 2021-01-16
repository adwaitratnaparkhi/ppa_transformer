# use this for the RRR1994 data

import sys
import json

for line in sys.stdin:
    tokens = line.rstrip().split(" ")

    example = {}

    example["heads"] = " ".join(tokens[1:5])
    example["label"] = tokens[5]

    print(json.dumps(example))

