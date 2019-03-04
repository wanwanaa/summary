import argparse
from LCSTS_char.config import Config
from LSTM.save_load import load_model
from LSTM.train_util import beam_test
from LCSTS_char.data_utils import load_embeddings

# result
filename_result = 'result/summary/'
# rouge
filename_rouge = 'result/summary/ROUGE.txt'
# initalization
open(filename_rouge, 'w')

if __name__ == '__main__':
    config = Config()
    # input
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size for train')
    parser.add_argument('--hidden_size', '-s', type=int, default=512, help='dimension of  code')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='number of training epochs')
    parser.add_argument('--num_layers', '-n', type=int, default=2, help='number of gru layers')
    parser.add_argument('--beam_size', '-r', type=int, default=2, help='number of beam size')
    parser.add_argument('--bidirectional', '-d', action='store_true', default=False, help="whether to use bi-gru")
    parser.add_argument('--attention', '-a', action='store_true', default=False, help="whether to use attention")
    parser.add_argument('--pre_train', '-p', action='store_true', default=False, help="load pre-train embedding")
    args = parser.parse_args()

    ########test######## #
    args.batch_size = 1
    args.beam_size = 3
    args.attention = True
    args.bidirectional = True
    ########test######## #

    # embeddings
    if args.pre_train:
        filename = config.filename_embeddings
        embeddings = load_embeddings(filename)
    else:
        embeddings = None

    # test
    for epoch in range(args.epoch):
        model = load_model(embeddings, epoch, config, args)
        beam_test(config, epoch, model, args, filename_result, filename_rouge)
