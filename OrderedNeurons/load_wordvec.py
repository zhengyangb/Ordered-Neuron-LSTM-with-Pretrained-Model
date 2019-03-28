import numpy as np
import os
import pickle as pkl
import torch

EOS_token = 0
PAD_IDX = 1
UNK_IDX = 2

VOCAB_SIZE = 80000
data_dir = 'data/'


def pkl_dumper(obj, file_name):
    with open(file_name+'.p', 'wb') as f:
        pkl.dump(obj, f, protocol=None)


if __name__ == '__main__':
    print('generate Glove embeddings')
    with open(os.path.join(data_dir, 'glove.840B.300d.txt')) as f:
        word_vecs = np.zeros((VOCAB_SIZE + 4, 300))
        word_vecs[UNK_IDX] = np.random.normal(scale=0.6, size=(300,))
        word_vecs[EOS_token] = np.random.normal(scale=0.6, size=(300,))

        words_ft = {'<pad>': PAD_IDX,
                    '<unk>': UNK_IDX,
                    '<eos>': EOS_token,
                    }
        idx2words_ft = {PAD_IDX: '<pad>', UNK_IDX: '<unk>', EOS_token: '<eos>'}
        count = 0
        for i, line in enumerate(f):
            if i == 0:
                continue
            if len(idx2words_ft) >= VOCAB_SIZE + 3:
                break
            s = line.split()
            if np.asarray(s[1:]).size != 300:
                print(i, np.asarray(s[1:]).size)
                continue
            word_vecs[count + 3, :] = np.asarray(s[1:])
            words_ft[s[0]] = count + 3
            idx2words_ft[count + 3] = s[0]
            count += 1
    word_vecs = torch.FloatTensor(word_vecs)
    pkl_dumper(word_vecs, os.path.join(data_dir, '_word_vecs'))
    pkl_dumper(words_ft, os.path.join(data_dir, '_words_ft'))
    pkl_dumper(idx2words_ft, os.path.join(data_dir, '_idx2words_ft'))