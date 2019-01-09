import pickle
import math
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


# beam search
def maximum(data):
    # position
    p = torch.argmax(data).item()
    # value
    v = data[p].item()
    data[p] = torch.tensor(0.0)
    return p, v, data


def max_prob(candidate):
    v = 0
    p = 0 # position of max
    for i in range(len(candidate)):
        if candidate[i][1] > v:
            v = candidate[i][1]
            p = i
    return p
# beam search


def test(config, epoch, model, args):
    # batch, dropout
    model = model.eval()

    # filename
    filename_test_text = config.filename_trimmed_test_text
    filename_test_summary = config.filename_trimmed_test_summary
    filename_idx2word = config.filename_index

    # data
    test = load_data(filename_test_text, filename_test_summary, 1, shuffle=False, num_works=2)

    # idx2word
    f = open(filename_idx2word, 'rb')
    idx2word = pickle.load(f)

    bos = config.bos
    s_len = config.summary_len
    # r = []
    result = []
    for batch in test:
        x, _ = batch
        if torch.cuda.is_available():
            x = x.cuda()
            # y = y.cuda()
        # model
        # attention
        if args.attention is True:
            h, encoder_outputs = model.encoder(x)
            sequence = [[[bos], 0.0]]
            # result = []
            for i in range(s_len):
                candidate = []
                for j in range(len(sequence)):
                    out = torch.tensor(sequence[j][0][-1]).type(torch.LongTensor).unsqueeze(0)
                    # print(out)
                    # print(out.size())
                    out, h = model.decoder(out, h, encoder_outputs)
                    out = torch.squeeze(model.output_layer(out))
                    out = torch.nn.functional.softmax(out, dim=0)
                    pre_path = sequence[j][0]
                    pre_prob = sequence[j][1]
                    for k in range(args.beam_size):
                        p, v, out = maximum(out)
                        path = pre_path.copy()
                        path.append(p)
                        prob = math.log(v) + pre_prob
                        candidate.append([path, prob])
                sequence = []
                for w in range(args.beam_size):
                    sequence.append(candidate.pop(max_prob(candidate)))
            sequence[max_prob(sequence)][0].pop(0)
            sen = index2sentence(sequence[max_prob(sequence)][0], idx2word)
            result.append(' '.join(sen))

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

        # for i in range(result.shape[0]):
        #     # sen1 = index2sentence(list(x[i]), idx2word)
        #     sen = index2sentence(list(result[i]), idx2word)
        #     r.append(' '.join(sen))

    # write result
    filename_data = filename_result + 'summary_' + str(epoch) + '.txt'
    with open(filename_data, 'w', encoding='utf-8') as f:
        f.write('\n'.join(result))

    # ROUGE
    score = rouge_score(config.gold_summaries, filename_data)

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
    parser.add_argument('--beam_size', '-z', type=int, default=2, help='number of beam search (using argmax when beam size=1)')
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
