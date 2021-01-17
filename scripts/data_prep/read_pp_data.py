#
# Read the BLBG PP attachment data
#
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
            pp_data.append((h.rstrip(), p.rstrip(), ch.rstrip(), l.rstrip(), sent.rstrip(), hi.rstrip(), pi.rstrip(), chi.rstrip()))
    else:
        for (h, p, ch, l) in zip(heads, preps, children, labels):
            pp_data.append((h.rstrip(), p.rstrip(), ch.rstrip(), l.rstrip()))

    return pp_data