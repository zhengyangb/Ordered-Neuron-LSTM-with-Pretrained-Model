import os
import torch

from collections import Counter
import tools


class Dictionary(object):
    def __init__(self, wvec=None, word2idx=None, idx2word=None):
        if wvec:
            self.word2idx = word2idx
            self.idx2word = idx2word
            self.counter = Counter()
            self.total = len(self.word2idx)
        else:
            self.word2idx = {}
            self.idx2word = []
            self.counter = Counter()
            self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, wvec=None, word2idx=None, idx2word=None):
        self.dictionary = Dictionary()
        self.wvec = wvec
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        if self.wvec:
            self.dictionary = Dictionary(self.wvec, self.word2idx, self.idx2word)
            ids = []
            with open(path, 'r') as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    ids += tools.indexesFromSentence(words, self.word2idx)
            ids = torch.tensor(ids, dtype=torch.long)

        else:
            with open(path, 'r') as f:
                tokens = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1

        return ids
