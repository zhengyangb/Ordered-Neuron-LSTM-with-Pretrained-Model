import os
import pickle as pkl
import numpy as np






def load_fasttext_embd(fname, corpus, words_to_load=100000, reload=False):

    def get_pretrain_emb(pretrained, token, notPretrained):
        if token == '<pad>':
            notPretrained.append(0)
            return [0] * 300
        if token in pretrained:
            notPretrained.append(0)
            return pretrained[token]
        else:
            notPretrained.append(1)
            return [0] * 300

    label = 'fasttext_emb'
    #     print(label)
    if os.path.exists(label + ".pkl") and (not reload):
        data = pkl.load(open(label + ".pkl", "rb"))
        print("found existing embeddings pickles.." + fname[:-4])
    else:
        print("loading embeddings.." + fname[:-4])
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        fin.readline()
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            if tokens[0] in corpus.dictionary.idx2word:
                data[tokens[0]] = list(map(float, tokens[1:]))

        fin.close()
        pkl.dump(data, open(label + ".pkl", "wb"))
    notPretrained = []
    embeddings = [get_pretrain_emb(data, token, notPretrained) for token in corpus.dictionary.idx2word]

    print("There are {} not pretrained words out of {} total words.".format(sum(notPretrained), len(notPretrained)))
    return embeddings, np.array(notPretrained)
