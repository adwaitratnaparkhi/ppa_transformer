import sys

def read_pp_data(prefix, use_sentences: bool):
    headsFile = open(prefix + ".heads.words", "r")
    heads = headsFile.readlines()

    prepFile = open(prefix + ".preps.words", "r")
    preps = prepFile.readlines()

    chFile = open(prefix + ".children.words", "r")
    children = chFile.readlines()

    labelFile = open(prefix + ".labels", "r")
    labels = labelFile.readlines()



    if use_sentences:
        sentFile = open(prefix + ".sentences", "r")
        sentences = sentFile.readlines()

        hids = open(prefix + '.id.heads.words', 'r').readlines()
        pids = open(prefix + '.id.preps.words', 'r').readlines()
        chids = open(prefix + '.id.children.words', 'r').readlines()

    pp_data = []

    if use_sentences:
        for (h, p, ch, l, sent, hi, pi, chi) in zip(heads, preps, children, labels, sentences, hids, pids, chids):
            pp_data.append((h, p, ch, l, sent, hi, pi, chi))
    else:
        for (h, p, ch, l) in zip(heads, preps, children, labels):
            pp_data.append((h, p, ch, l))

    return pp_data


# remove newlines and add tabs for easier printing
def make_key(h, p, ch, l):
    return h.rstrip() + "\t" + p.rstrip() + "\t" + ch.rstrip() + "\t" + l.rstrip()

def create_dict(pp_data):

    dict = {}

    for (h, p, ch, l, s, hi, pi, chi) in pp_data:
        key = make_key(h, p, ch, l)

        if key in dict:
            dict[key].append( (s, hi, pi, chi ))
        else:
            dict[key] = [ (s,  hi, pi, chi) ]

    return dict





if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("args: [Original PP data prefix] [Rebuilt PP data prefix]")
        sys.exit(1)

    orig_pp_data = read_pp_data(sys.argv[1], False)
    rebuilt_pp_data = read_pp_data(sys.argv[2], True)

    outFileName = sys.argv[1] + ".sentences"
    outFile = open(outFileName, "w")

    hi_outfile = open(sys.argv[1] + ".id.heads.words", "w")
    pi_outfile = open(sys.argv[1] + ".id.preps.words", "w")
    chi_outfile = open(sys.argv[1] + ".id.children.words", "w")

    dict = create_dict(rebuilt_pp_data)

    numWritten = 0
    # loop through original, join with sentence in dict
    # for repeats, keep track of which sentence was used
    index_used = {}
    for (h, p, ch, l) in orig_pp_data:
        key = make_key(h, p, ch, l)

        assert(key in dict)

        sentList = dict[key]
        index = 0
        if key in index_used:
            index_used[key] += 1
        else:
            index_used[key] = 0

        index = index_used[key]
        assert(index < len(sentList))
        sentence, hi, pi, chi = sentList[index]

        #for debugging
        #print(key + "\t" + sentence.rstrip())

        print(sentence.rstrip(), file=outFile)
        print(hi.rstrip(), file=hi_outfile)
        print(pi.rstrip(), file=pi_outfile)
        print(chi.rstrip(), file=chi_outfile)

        numWritten += 1

    outFile.close()
    print(f"Wrote {numWritten} lines to {outFileName}")









