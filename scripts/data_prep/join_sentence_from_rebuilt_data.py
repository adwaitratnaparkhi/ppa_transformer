import sys

from read_pp_data import *


# remove newlines and add tabs for easier printing
def make_key(h, p, ch, l):
    return h + "\t" + p + "\t" + ch + "\t" + l

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
        #print(key + "\t" + sentence())

        print(sentence, file=outFile)
        print(hi, file=hi_outfile)
        print(pi, file=pi_outfile)
        print(chi, file=chi_outfile)

        numWritten += 1

    outFile.close()
    print(f"Wrote {numWritten} lines to {outFileName}")









