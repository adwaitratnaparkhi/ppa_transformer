# use this for the BLBG2014 data

import sys
import json
import os
import argparse

class EXP_MODE:
    BINARY = "binary"
    MULTI_CHOICE_FULL = "multichoice_full_sentence"
    MULTI_CHOICE_HEADS = "multichoice_head_only"
mode_array = [EXP_MODE.BINARY,
              EXP_MODE.MULTI_CHOICE_FULL,
              EXP_MODE.MULTI_CHOICE_HEADS]
def read_file(filename, is_numeric=False, is_split=False):
    with open(filename) as f:
        data = f.readlines()
    data = [t.strip() for t in data]
    if is_split:
        data = [t.split() for t in data]
    if is_numeric:
        data = [[int(l) for l in t] if is_split else int(t) for t in data]
    return data

def read_multichoice_data(prefix, use_sentences: bool):
    def get_filename(prefix, suffix):
        return f'{prefix}.{suffix}'

    labels = read_file(get_filename(prefix, 'labels'), is_numeric=True)

    if use_sentences:
        sentences = read_file(get_filename(prefix, "sentences"), is_numeric=False, is_split=True)
        hids = read_file(get_filename(prefix, "id.heads.words"), is_numeric=True, is_split=True)
        pids = read_file(get_filename(prefix, "id.preps.words"), is_numeric=True)
        chids = read_file(get_filename(prefix, "id.children.words"), is_numeric=True)
    else:
        heads = read_file(get_filename(prefix, "heads.words"), is_numeric=False, is_split=True)
        preps = read_file(get_filename(prefix, "preps.words"))
        children = read_file(get_filename(prefix, "children.words"))

    pp_data = []
    for i, l in enumerate(labels):
        example = {}
        if use_sentences:
            example['sentence'] = sentences[i]
            example['heads_index'] = hids[i]
            example['pp_index'] = pids[i]
            example['children_index'] = chids[i]
        else:
            head_words_str = ' '.join(heads[i])
            head_idx = list(range(len(heads[i])))
            full_sentence = f'{head_words_str} {preps[i]} {children[i]}'
            example['sentence'] = full_sentence.split()
            example['heads_index'] = head_idx
            example['pp_index'] = head_idx[-1] + 1
            example['children_index'] = head_idx[-1] + 2
        example['label'] = labels[i] - 1

        pp_data.append(example)
    return pp_data

def read_binary_data(filepath):
    data = read_file(filepath, is_numeric=False, is_split=True)
    pp_data = []
    for (_, v, n, p, n2, label) in data:
        example = {}
        full_sentence = f"{n} {v} {p} {n2}"
        example['sentence'] = full_sentence.split()
        example['heads_index'] = [0, 1]
        example['pp_index'] = 2
        example['children_index'] = 3
        example['label'] = 0 if label == "N" else 1
        pp_data.append(example)
    return pp_data

# experiment modes
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parsing experiment files')
    parser.add_argument("-i", dest="infile", required=True,
                        help="input file path")
    parser.add_argument("-o", dest="outfile", required=True,
                        help="output file path")
    parser.add_argument('-m', dest="mode", required=True,
                        choices=[1, 2, 3],
                        type = int,
                        help='Experiment mode')

    args = parser.parse_args()

    filepath = os.path.expanduser(args.infile)
    output_file = os.path.expanduser(args.outfile)
    exp_mode = mode_array[args.mode-1]
    if exp_mode == EXP_MODE.BINARY:
        pp_data = read_binary_data(filepath)
    else:
        pp_data = read_multichoice_data(filepath, exp_mode == EXP_MODE.MULTI_CHOICE_FULL)

    with open(output_file, 'w') as outfile:
        outfile.write('\n'.join([json.dumps(d) for d in pp_data]) + '\n')
