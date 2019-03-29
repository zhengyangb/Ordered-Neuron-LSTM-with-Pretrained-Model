import numpy as np
import os
import pickle as pkl
import torch

EOS_token = 0
PAD_IDX = 1
UNK_IDX = 2


def pkl_dumper(obj, fname):
    with open(fname+'.p', 'wb') as f:
        pkl.dump(obj, f, protocol=None)


def pkl_loader(fname):
    with open(fname+'.p', 'rb') as f:
        obj = pkl.load(f)
    return obj


def load_wvec(type, max_vocab=50000):
    if type == 'glove':
        emb_size = 300
        output_dir = 'data/wordvec/glove'
        print('generate {} Glove embeddings'.format(max_vocab))
        with open(os.path.join('data', 'glove.840B.300d.txt')) as f:
            word_vecs = np.zeros((max_vocab + 3, emb_size))
            word_vecs[UNK_IDX] = np.random.normal(scale=0.6, size=(emb_size,))
            word_vecs[EOS_token] = np.random.normal(scale=0.6, size=(emb_size,))

            words2idx = {'<pad>': PAD_IDX,
                        '<unk>': UNK_IDX,
                        '<eos>': EOS_token,
                        }
            idx2words = {PAD_IDX: '<pad>', UNK_IDX: '<unk>', EOS_token: '<eos>'}
            count = 0
            for i, line in enumerate(f):
                if i == 0:
                    continue
                if len(idx2words) >= max_vocab + 3:
                    break
                s = line.split()
                if np.asarray(s[1:]).size != emb_size:
                    print(i, np.asarray(s[1:]).size)
                    continue
                word_vecs[count + 3, :] = np.asarray(s[1:])
                words2idx[s[0]] = count + 3
                idx2words[count + 3] = s[0]
                count += 1
        word_vecs = torch.FloatTensor(word_vecs)
        pkl_dumper(word_vecs, os.path.join(output_dir, 'word_vecs'))
        pkl_dumper(words2idx, os.path.join(output_dir, 'words2idx'))
        pkl_dumper(idx2words, os.path.join(output_dir, 'idx2words'))


def indexesFromSentence(tokens, word2index):
    indexes = [word2index[token] if token in word2index else UNK_IDX for token in tokens]
    return indexes

