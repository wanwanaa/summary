import argparse
from LCSTS_char.config import Config
from LCSTS_char.data_utils import file_check, get_datasets_train, get_datasets, get_vocab, word2index, index2word, \
    write_vocab, get_trimmed_datasets, get_embeddings, write_gold_summaries, write_train


def main():
    config = Config()

    # input
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', metavar='filename', type=str, nargs='+',
                        help='filenames must be given in order [filename_train, filename_valid, filename_test]\
                                 (default:*), if only a few files are to be given, default ones must be replaced by "*"')
    parser.add_argument('--max_length', '-m', metavar='NUM', type=int, help='display max_length')
    parser.add_argument('--summary_length', '-s', metavar='NUM', type=int, help='display summary_length')

    args = parser.parse_args()
    if args.max_length:
        config.max_length = args.max_length
    if args.summary_length:
        config.summary_length = args.summary_length

    if args.filename:
        name = ['config.filename_train', 'config.filename_valid', 'config.filename_test']
        for i in range(len(args.filename)):
            if args.filename[i] == '*':
                continue
            else:
                a = name[i] + '=' + '\'' + args.filename[i] + '\''
                exec(a)

    file_check([config.filename_train, config.filename_valid, config.filename_test])

    # Generate datasets
    train, train_summary = get_datasets_train(config.filename_train)
    valid, valid_summary = get_datasets(config.filename_valid)
    test, test_summary = get_datasets(config.filename_test)

    # bulid word vocab
    vocab = get_vocab(train)
    word2idx = word2index(vocab, config.vocab_size)
    idx2word = index2word(vocab, config.vocab_size)

    # save vocab
    write_vocab(config.filename_words, word2idx)
    write_vocab(config.filename_index, idx2word)

    # save npz
    # glove embedding
    get_embeddings(config.filename_glove, config.filename_embeddings, word2idx, config.vocab_size, config.dim)

    # train
    get_trimmed_datasets(config.filename_trimmed_train_text, train, word2idx, config.seq_len)
    get_trimmed_datasets(config.filename_trimmed_train_summary, train_summary, word2idx, config.summary_len)

    # valid
    get_trimmed_datasets(config.filename_trimmed_valid_text, valid, word2idx, config.seq_len)
    get_trimmed_datasets(config.filename_trimmed_valid_summary, valid_summary, word2idx, config.summary_len)

    # test
    get_trimmed_datasets(config.filename_trimmed_test_text, test, word2idx, config.seq_len)
    get_trimmed_datasets(config.filename_trimmed_test_summary, test_summary, word2idx, config.summary_len)

    # gold summaries
    write_gold_summaries(test_summary, config.gold_summaries)


if __name__ == '__main__':
    # print('build_data...')
    # main()
    # print('Done!')

    config = Config()
    # _, datasets = get_datasets(config.filename_test)
    # write_gold_summaries(datasets, config.gold_summaries)

    # # search word in GloVe
    # train, train_summary = get_datasets_train(config.filename_train)
    # vocab = get_vocab(train)
    # word2idx = word2index(vocab, config.vocab_size)
    # get_embeddings(config.filename_glove, config.filename_embeddings, word2idx, config.vocab_size, config.dim)

    # # get train.txt for training GloVe
    # text, summary = get_datasets_train(config.filename_train)
    # write_train(text, summary)