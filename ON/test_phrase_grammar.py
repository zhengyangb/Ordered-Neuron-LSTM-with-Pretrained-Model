import argparse
import re
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import data_ptb
from utils import batchify, get_batch, repackage_hidden, evalb

from parse_comparison import corpus_stats_labeled, corpus_average_depth
from data_ptb import word_tags
import tools

criterion = nn.CrossEntropyLoss()

def evaluate(data_source, batch_size=1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        output = model.decoder(output)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def corpus2idx(sentence):
    arr = np.array([data.dictionary.word2idx[c] for c in sentence.split()], dtype=np.int32)
    return torch.from_numpy(arr[:, None]).long()


# Test model
def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = np.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


# def build_tree(depth, sen):
#     assert len(depth) == len(sen)
#     assert len(depth) >= 0
#
#     if len(depth) == 1:
#         parse_tree = sen[0]
#     else:
#         idx_max = numpy.argmax(depth[1:]) + 1
#         parse_tree = []
#         if len(sen[:idx_max]) > 0:
#             tree0 = build_tree(depth[:idx_max], sen[:idx_max])
#             parse_tree.append(tree0)
#         if len(sen[idx_max:]) > 0:
#             tree1 = build_tree(depth[idx_max:], sen[idx_max:])
#             parse_tree.append(tree1)
#     return parse_tree


def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1

def MRG(tr):
    if isinstance(tr, str):
        #return '(' + tr + ')'
        return tr + ' '
    else:
        s = '( '
        for subtr in tr:
            s += MRG(subtr)
        s += ') '
        return s

def MRG_labeled(tr):
    if isinstance(tr, nltk.Tree):
        if tr.label() in word_tags:
            return tr.leaves()[0] + ' '
        else:
            s = '(%s ' % (re.split(r'[-=]', tr.label())[0])
            for subtr in tr:
                s += MRG_labeled(subtr)
            s += ') '
            return s
    else:
        return ''

def mean(x):
    return sum(x) / len(x)


def test(model, corpus, cuda, prt=False):
    model.eval()

    prec_list = []
    reca_list = []
    f1_list = []

    pred_tree_list = []
    targ_tree_list = []

    nsens = 0
    word2idx = corpus.dictionary.word2idx
    if args.wsj10:
        dataset = zip(corpus.train_sens, corpus.train_trees, corpus.train_nltktrees)
    else:
        dataset = zip(corpus.test_sens, corpus.test_trees, corpus.test_nltktrees)

    corpus_sys = {}
    corpus_ref = {}
    for sen, sen_tree, sen_nltktree in dataset:
        if args.wsj10 and len(sen) > 12:
            continue
        x = np.array([word2idx[w] if w in word2idx else word2idx['<unk>'] for w in sen])
        input = Variable(torch.LongTensor(x[:, None]))
        if cuda:
            input = input.cuda()

        hidden = model.init_hidden(1)
        _, hidden = model(input, hidden)

        distance = model.distance[0].squeeze().data.cpu().numpy()
        distance_in = model.distance[1].squeeze().data.cpu().numpy()

        nsens += 1
        if prt and nsens % 100 == 0:
            for i in range(len(sen)):
                tools.print_log(args.save, '%15s\t%s\t%s' % (sen[i], str(distance[:, i]), str(distance_in[:, i])))
            tools.print_log(args.save, 'Standard output:{}'.format(sen_tree))

        sen_cut = sen[1:-1]
        # gates = distance.mean(axis=0)
        for gates in [
            # distance[0],
            distance[1],
            # distance[2],
            # distance.mean(axis=0)
        ]:
            depth = gates[1:-1]
            parse_tree = build_tree(depth, sen_cut)

            corpus_sys[nsens] = MRG(parse_tree)
            corpus_ref[nsens] = MRG_labeled(sen_nltktree)

            pred_tree_list.append(parse_tree)
            targ_tree_list.append(sen_tree)

            model_out, _ = get_brackets(parse_tree)
            std_out, _ = get_brackets(sen_tree)
            overlap = model_out.intersection(std_out)

            prec = float(len(overlap)) / (len(model_out) + 1e-8)
            reca = float(len(overlap)) / (len(std_out) + 1e-8)
            if len(std_out) == 0:
                reca = 1.
                if len(model_out) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            prec_list.append(prec)
            reca_list.append(reca)
            f1_list.append(f1)

            if prt and nsens % 100 == 0:
                tools.print_log(args.save, 'Model output:{}'.format(parse_tree))
                tools.print_log(args.save, 'Prec: %f, Reca: %f, F1: %f' % (prec, reca, f1))

        if prt and nsens % 100 == 0:
            tools.print_log(args.save + '.log', '-' * 80)

            f, axarr = plt.subplots(3, sharex=True, figsize=(distance.shape[1] // 2, 6))
            axarr[0].bar(np.arange(distance.shape[1])-0.2, distance[0], width=0.4)
            axarr[0].bar(np.arange(distance_in.shape[1])+0.2, distance_in[0], width=0.4)
            axarr[0].set_ylim([0., 1.])
            axarr[0].set_ylabel('1st layer')
            axarr[1].bar(np.arange(distance.shape[1]) - 0.2, distance[1], width=0.4)
            axarr[1].bar(np.arange(distance_in.shape[1]) + 0.2, distance_in[1], width=0.4)
            axarr[1].set_ylim([0., 1.])
            axarr[1].set_ylabel('2nd layer')
            axarr[2].bar(np.arange(distance.shape[1]) - 0.2, distance[2], width=0.4)
            axarr[2].bar(np.arange(distance_in.shape[1]) + 0.2, distance_in[2], width=0.4)
            axarr[2].set_ylim([0., 1.])
            axarr[2].set_ylabel('3rd layer')
            plt.sca(axarr[2])
            plt.xlim(left=-0.5, right=distance.shape[1] - 0.5)
            plt.xticks(np.arange(distance.shape[1]), sen, fontsize=10, rotation=45)

            figure_dir = model_dir + '/figure/'
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)
            plt.savefig(figure_dir+'%d.png' % (nsens))
            plt.close()

    prec_list, reca_list, f1_list \
        = np.array(prec_list).reshape((-1,1)), np.array(reca_list).reshape((-1,1)), np.array(f1_list).reshape((-1,1))
    if prt:
        tools.print_log(args.save, '-' * 80)
        np.set_printoptions(precision=1)
        tools.print_log(args.save,
                        'Mean Prec:{:.1f}, Mean Reca:{:.1f}, Mean F1:{:.1f}'.format(100.0 * prec_list.mean(axis=0)[0],
                                                                                    100.0 * reca_list.mean(axis=0)[0],
                                                                                    100.0 * f1_list.mean(axis=0)[0]))
        tools.print_log(args.save, 'Number of sentence: %i' % nsens)

        correct, total = corpus_stats_labeled(corpus_sys, corpus_ref)
        tools.print_log(args.save, correct)
        tools.print_log(args.save, total)
        tools.print_log(args.save, 'ADJP:{}, {}, {:.1f}'.format(correct['ADJP'], total['ADJP'],
                                                                100 * correct['ADJP'] / total['ADJP']))
        tools.print_log(args.save,
                        'NP::{}, {}, {:.1f}'.format(correct['NP'], total['NP'], 100 * correct['NP'] / total['NP']))
        tools.print_log(args.save,
                        'PP::{}, {}, {:.1f}'.format(correct['PP'], total['PP'], 100 * correct['PP'] / total['PP']))
        tools.print_log(args.save, 'INTJ::{}, {}, {:.1f}'.format(correct['INTJ'], total['INTJ'],
                                                                 100 * correct['INTJ'] / total['INTJ']))
        tools.print_log(args.save, 'Averaged Depth {:.1f}'.format(corpus_average_depth(corpus_sys)))
        evalb(pred_tree_list, targ_tree_list)
        tools.print_log(args.save,
                        '{:.1f},' ',{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}'.format(100 * prec_list.mean(axis=0)[0],
                                                                               corpus_average_depth(corpus_sys),
                                                                               100 * correct['ADJP'] / total['ADJP'],
                                                                               100 * correct['NP'] / total['NP'],
                                                                               100 * correct['PP'] / total['PP'],
                                                                               100 * correct['INTJ'] / total['INTJ']))
    return f1_list.mean(axis=0)


if __name__ == '__main__':
    marks = [' ', '-', '=']

    np.set_printoptions(precision=2, suppress=True, linewidth=5000)

    parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

    # Model parameters.
    parser.add_argument('--data', type=str, default='data/ptb',
                        help='location of the data corpus')
    parser.add_argument('--data_train', type=str, default='data/penn',
                        help='location of the data corpus')
    parser.add_argument('--check_dir', type=str, default='PTB',
                        help='model checkpoint to use')
    parser.add_argument('--output', metavar='SAVE DIR',
                        default='./res/',
                        help='path to save result')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--wsj10', action='store_true',
                        help='use WSJ10')
    parser.add_argument('--wvec', type=str, default='',
                        help='load pretrained word vector')
    args = parser.parse_args()
    args.bptt = 70
    model_dir = args.output + args.check_dir
    args.checkpoint = os.path.join(model_dir, args.check_dir + '.pt')
    args.save = model_dir + '/test'

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    # Load model
    with open(args.checkpoint, 'rb') as f:
        model, _, _ = torch.load(f)
        torch.cuda.manual_seed(args.seed)
        model.cpu()
        if args.cuda:
            model.cuda()
            torch.cuda.synchronize()

    # Load data
    import hashlib

    # fn = 'corpus.{}.data'.format(hashlib.md5('data/penn/'.encode()).hexdigest())
    fn = 'corpus.{}.data'.format(hashlib.md5((args.data_train + args.wvec).encode()).hexdigest())
    # fn = 'corpus.{}.data'.format(hashlib.md5((args.data + args.wvec).encode()).hexdigest())
    tools.print_log(args.save, 'Loading cached dataset...')
    corpus = torch.load(fn)
    dictionary = corpus.dictionary

    # test_batch_size = 1
    # test_data = batchify(corpus.test, test_batch_size, args)
    # test_loss = evaluate(test_data, test_batch_size)
    # print('=' * 89)
    # print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    #     test_loss, math.exp(test_loss), test_loss / math.log(2)))
    # print('=' * 89)

    tools.print_log(args.save, 'Loading PTB dataset...')
    if args.wvec:
        word2idx = tools.pkl_loader(os.path.join('data/wordvec', args.wvec, 'words2idx'))
        idx2word = tools.pkl_loader(os.path.join('data/wordvec', args.wvec, 'idx2words'))
        corpus = data_ptb.Corpus(args.data, args.wvec, word2idx, idx2word)
    else:
        corpus = data_ptb.Corpus(args.data)
    corpus.dictionary = dictionary

    test(model, corpus, args.cuda, prt=True)
