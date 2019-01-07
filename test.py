import pickle
import torch
import argparse
import numpy as np
from LCSTS_char.config import Config
from LSTM.ROUGE import rouge_score, write_rouge
from LSTM.save_load import load_model
from LCSTS_char.data_utils import index2sentence, load_data, load_embeddings


# filename
# result
filename_result = 'result/summary/'
# rouge
filename_rouge = 'result/summary/ROUGE.txt'
# initalization
open(filename_rouge, 'w')


def test(config, epoch, model, args):
    # batch, dropout
    model = model.eval()

    # filename
    filename_test_text = config.filename_trimmed_test_text
    filename_test_summary = config.filename_trimmed_test_summary
    filename_idx2word = config.filename_index

    # data
    test = load_data(filename_test_text, filename_test_summary, args.batch_size, shuffle=False, num_works=2)

    # idx2word
    f = open(filename_idx2word, 'rb')
    idx2word = pickle.load(f)

    bos = config.bos
    s_len = config.summary_len
    r = []
    for batch in test:
        x, _ = batch
        if torch.cuda.is_available():
            x = x.cuda()
            # y = y.cuda()
        # model
        # attention
        if args.attention is True:
            h, encoder_outputs = model.encoder(x)
            out = (torch.ones(x.size(0)) * bos)
            result = []
            for i in range(s_len):
                out = out.type(torch.LongTensor)
                out, h = model.decoder(out, h, encoder_outputs)
                out = torch.squeeze(model.output_layer(out))
                out = torch.nn.functional.softmax(out, dim=1)
                out = torch.argmax(out, dim=1)
                result.append(out.numpy())
            result = np.transpose(np.array(result))

        # seq2seq
        else:
            h, _ = model.encoder(x)
            out = (torch.ones(x.size(0)) * bos)
            result = []
            for i in range(s_len):
                out = out.type(torch.LongTensor).view(-1, 1)
                out, h = model.decoder(out, h)
                out = torch.squeeze(model.output_layer(out))
                out = torch.nn.functional.softmax(out, dim=1)
                out = torch.argmax(out, dim=1)
                result.append(out.numpy())
            result = np.transpose(np.array(result))

        for i in range(result.shape[0]):
            # sen1 = index2sentence(list(x[i]), idx2word)
            sen = index2sentence(list(result[i]), idx2word)
            r.append(' '.join(sen))

    # write result
    filename_data = filename_result + 'summary_' + str(epoch) + '.txt'
    with open(filename_data, 'w', encoding='utf-8') as f:
        f.write('\n'.join(r))

    # ROUGE
    score = rouge_score(config.gold_summaries, filename_result)

    # write rouge
    write_rouge(filename_rouge, score)

    # print rouge
    print('epoch:', epoch, '|ROUGE-1 f: %.4f' % score['rouge-1']['f'],
          ' p: %.4f' % score['rouge-1']['p'],
          ' r: %.4f' % score['rouge-1']['r'])
    print('epoch:', epoch, '|ROUGE-2 f: %.4f' % score['rouge-2']['f'],
          ' p: %.4f' % score['rouge-2']['p'],
          ' r: %.4f' % score['rouge-2']['r'])
    print('epoch:', epoch, '|ROUGE-L f: %.4f' % score['rouge-l']['f'],
          ' p: %.4f' % score['rouge-l']['p'],
          ' r: %.4f' % score['rouge-l']['r'])


if __name__ == '__main__':
    config = Config()
    # input
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size for train')
    parser.add_argument('--hidden_size', '-s', type=int, default=512, help='dimension of  code')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='number of training epochs')
    parser.add_argument('--num_layers', '-n', type=int, default=2, help='number of gru layers')
    parser.add_argument('--pre_train', '-p', action='store_true', default=False, help="load pre-train embedding")
    parser.add_argument('--attention', '-a', action='store_true', default=False, help="whether to use attention")
    # parser.add_argument('--devices', '-d', type=int, default=2, help='specify a gpu')
    args = parser.parse_args()

    # embeddings
    if args.pre_train is True:
        filename = config.filename_embeddings
        embeddings = load_embeddings(filename)
    else:
        embeddings = None

    # test
    for epoch in range(args.epoch):
        model = load_model(embeddings, epoch, config, args)
        test(config, epoch, model, args)
