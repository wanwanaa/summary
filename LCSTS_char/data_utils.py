import torch
import torch.utils.data as data_util
import numpy as np
import pickle
import os
from scipy.stats import truncnorm


# check whether the given 'filename' exists
# raise a FileNotFoundError when file not found
def file_check(filename):
    for name in filename:
        if os.path.isfile(name) is False:
            raise FileNotFoundError('No such file or directory: {}'.format(name))


# train
def get_datasets_train(filename):
    text = []
    summary = []
    group = []
    i = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            group.append(line.strip())
            i += 1
            if i % 8 == 0:
                summary.append(group[2])
                text.append(group[5])
                group = []
                i = 0
    return text, summary


# vaild, test(human label)
def get_datasets(filename):
    text = []
    summary = []
    group = []
    i = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            group.append(line.strip())
            i += 1
            if i % 9 == 0:
                label = int(list(group[1].split('<')[1])[-1])
                if label >= 3:
                    summary.append(group[3])
                    text.append(group[6])
                group = []
                i = 0
    return text, summary


def get_vocab(train_text):
    vocab = {}
    for line in train_text:
        line = list(line)
        for v in line:
            flag = vocab.get(v)
            if flag is None:
                vocab[v] = 0
            else:
                vocab[v] += 1
    vocab = sorted(vocab.items(), key=lambda x:x[1], reverse=True)
    return vocab


def word2index(vocab, vocab_size):
    word2idx = {}
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1
    word2idx['<bos>'] = 2
    word2idx['<eos>'] = 3
    for i in range(vocab_size-4):
        word2idx[vocab[i][0]] = i + 4
    return word2idx


def index2word(vocab, vocab_size):
    idx2word = ['<pad>', '<unk>', '<bos>', '<eos>']
    for i in range(vocab_size-4):
        idx2word.append(vocab[i][0])
    return idx2word


# save word2idx idx2word
def write_vocab(filename, vocab):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)
    print('vocab saved at: DATA/result')


# save npy
def get_trimmed_datasets(filename, datasets, word2idx, max_length):
    data = np.zeros([len(datasets), max_length])
    k = 0
    for line in datasets:
        line = list(line)
        sen = np.zeros(max_length, dtype=np.int32)
        for i in range(max_length):
            if i == len(line):
                sen[i] = word2idx['<eos>']
                break
            else:
                flag = word2idx.get(line[i])
                if flag is None:
                    sen[i] = word2idx['<unk>']
                else:
                    sen[i] = word2idx[line[i]]
        data[k] = sen
        k += 1
    data = torch.from_numpy(data)
    torch.save(data, filename)
    print('datasets saved at: DATA/result')


def index2sentence(index, idx2word):
    sen = []
    for i in range(len(index)):
        if idx2word[index[i]] == '<eos>':
            break
        else:
            sen.append(idx2word[index[i]])
        # sen.append(idx2word[index[i]])
    return sen


def get_embeddings(filename_glove, filename, word2idx, vocab_size, dim):
    embeddings = np.zeros((vocab_size, dim))
    flag = list(np.arange(0, 4000))
    with open(filename_glove, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            if word in word2idx.keys():
                flag.remove(word2idx[word])
                embedding = [float(x) for x in line[1:]]
                embeddings[word2idx[word]] = embedding
    for i in flag:
        np.random.seed(i)
        embedding = truncnorm.rvs(-2, 2, size=dim)
        embeddings[i] = embedding
    embeddings = torch.from_numpy(embeddings)
    torch.save(embeddings, filename)
    print('embeddings save at: DATA/result')


def write_gold_summaries(datasets, filename):
    gold_summary = []
    for line in datasets:
        line = list(line)
        gold_summary.append(' '.join(line))
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(gold_summary))


def load_data(filename_text, filename_summary, batch_size, shuffle, num_works):
    text = torch.load(filename_text).type(torch.LongTensor)
    summary = torch.load(filename_summary).type(torch.LongTensor)
    data = data_util.TensorDataset(text, summary)
    data_loader = data_util.DataLoader(data, batch_size, shuffle=shuffle, num_workers=num_works)
    return data_loader


def load_embeddings(filename):
    embeddings = torch.load(filename).type(torch.FloatTensor)
    return embeddings


# GloVe train.txt
def write_train(text, summary):
    result = []
    for line in text:
        result.append(' '.join(list(line)))
    with open('DATA/data/train.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))
    print('train.txt save at DATA/data')