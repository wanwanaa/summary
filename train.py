import time
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from LCSTS_char.config import Config
from LSTM.model import Encoder, Decoder, Seq2Seq, Attention, AttnDecoder, AttnSeq2Seq
from LSTM.save_load import save_model
from LSTM.ROUGE import rouge_score, write_rouge
from LCSTS_char.data_utils import index2sentence, load_data, load_embeddings


# filename
# save model
filename_model = 'models/summary/'
# result
filename_result = 'result/summary/'
# rouge
filename_rouge = 'result/summary/ROUGE.txt'
# initalization
open(filename_rouge, 'w')
# checkpoint
filename_checkpoint = '../models/summary/'

# plot
train_loss = []
valid_loss = []
test_rouge = []


def plot_result(train, valid, test):
    x = np.linspace(0, len(train), len(train))
    plt.plot(x, train, 'r', label='train loss')
    plt.plot(x, valid, 'b', label='valid loss')
    plt.plot(x, test, 'g',  label='ROUGE-1')

    plt.show()


def valid(config, epoch, model):
    # filename
    filename_valid_text = config.filename_trimmed_valid_text
    filename_valid_summary = config.filename_trimmed_valid_summary

    # data
    valid = load_data(filename_valid_text, filename_valid_summary, args.batch_size, shuffle=True, num_works=2)

    # loss
    loss_func = nn.CrossEntropyLoss()

    all_loss = 0
    num = 0
    for batch in enumerate(valid):
        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        result = model(x, y)
        result = result.contiguous().view(-1, 4000)
        y = y.view(-1)
        loss = loss_func(result, y)
        all_loss += loss.item()
        num += 1
    print('epoch:', epoch, '|valid_loss: %.4f' % (all_loss / num))
    valid_loss.append(all_loss / num)


def test(config, epoch, model, args):
    model.eval()
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
                out = F.softmax(out, dim=1)
                out = torch.argmax(out, dim=1)
                result.append(out.numpy())
            result = np.transpose(np.array(result))

        for i in range(result.shape[0]):
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

        test_rouge.append(score['rouge-1']['f'])


def train(args, config, model):
    start = time.time()
    # filename
    filename_train_text = config.filename_trimmed_train_text
    filename_train_summary = config.filename_trimmed_train_summary

    # data
    train = load_data(filename_train_text, filename_train_summary, args.batch_size, shuffle=True, num_works=2)

    # # idx2word # display the result
    # filename_idx2word = config.filename_index
    # f = open(filename_idx2word, 'rb')
    # idx2word = pickle.load(f)

    # loss
    optim = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_func = nn.CrossEntropyLoss()

    start_epoch = 0
    # checkpoint
    if args.checkpoint != 0:
        start_epoch = args.checkpoint
        filename = filename_checkpoint + 'model_' + args.checkpoint + '.pkl'
        model.load_state_dict(torch.load(filename))

    for e in range(start_epoch, args.epoch):
        all_loss = 0
        num = 0
        for step, batch in enumerate(train):
            x, y = batch
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            result = model(x, y)

            # # display the result
            # a = y[0]
            # b = result[0]

            result = result.contiguous().view(-1, 4000)
            y = y.view(-1)
            loss = loss_func(result, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            all_loss += loss.item()
            num += 1
            if step % 200 == 0:
                print('epoch:', e, '|step:', step, '|train_loss: %.4f' % loss.item())
                # # display the result
                # if torch.cuda.is_available():
                #     a = list(a.cpu().numpy())
                #     b = list(torch.argmax(b, dim=1).cpu().numpy())
                # else:
                #     a = list(a.numpy())
                #     b = list(torch.argmax(b, dim=1).numpy())
                # a = index2sentence(a, idx2word)
                # b = index2sentence(b, idx2word)
                # # display the result
                # print(''.join(a))
                # print(''.join(b))

        # train loss
        print('epoch:', e, '|train_loss: %.4f' % (all_loss / num))
        train_loss.append(all_loss / num)

        # save model
        if args.save_model is True:
            filename = filename_model + 'model_' + str(e) + '.pkl'
            save_model(model, filename)

        # valid
        valid(config, e, model)

        # test
        test(config, e, model, args)

        # end time
        end = time.time()
        print('time: ', (end - start))


if __name__ == '__main__':
    # input
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=512, help='batch size for train')
    parser.add_argument('--hidden_size', '-l', type=int, default=512, help='dimension of  code')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='number of training epochs')
    parser.add_argument('--num_layers', '-n', type=int, default=2, help='number of gru layers')
    parser.add_argument('--checkpoint', '-c', type=int, default=0, help='number of checking model')
    parser.add_argument('--pre_train', '-p', action='store_true', default=False, help="load pre-train embedding")
    parser.add_argument('--attention', '-a', action='store_true', default=False, help="whether to use attention")
    parser.add_argument('--save_model', '-m', action='store_true', default=False, help="whether to save model")
    # parser.add_argument('--devices', '-d', type=int, default=2, help='specify a gpu')
    # parser.add_argument('--beam_size', '-s', type=int, default=2, help='size of beam search')
    args = parser.parse_args()

    # config
    config = Config()

    # embedding
    if args.pre_train is True:
        filename = config.filename_embeddings
        embeddings = load_embeddings(filename)
    else:
        embeddings = None

    # model
    if args.attention is True:
        # attention model
        if torch.cuda.is_available():
            encoder = Encoder(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers).cuda()
            # v1(model)
            attention = Attention(args.hidden_size).cuda()
            # # v2(model_attn)
            # attention = Attention(args.hidden_size, config.EMBEDDING_SIZE, config.seq_len).cuda()

            decoder = AttnDecoder(attention, embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE,
                                  args.hidden_size, config.summary_len, args.num_layers).cuda()
            seq2seq = AttnSeq2Seq(encoder, decoder, config.VOCAB_SIZE, args.hidden_size, config.summary_len, config.bos).cuda()
        else:
            encoder = Encoder(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
            # v1
            attention = Attention(args.hidden_size).cuda()
            # # v2
            # attention = Attention(args.hidden_size, config.EMBEDDING_SIZE, config.seq_len)
            decoder = AttnDecoder(attention, embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE,
                                  args.hidden_size, config.summary_len, args.num_layers)
            seq2seq = AttnSeq2Seq(encoder, decoder, config.VOCAB_SIZE, args.hidden_size, config.summary_len, config.bos)
    else:
        # seq2seq model
        if torch.cuda.is_available():
            encoder = Encoder(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers).cuda()
            decoder = Decoder(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers).cuda()
            seq2seq = Seq2Seq(encoder, decoder, config.VOCAB_SIZE, args.hidden_size, config.bos).cuda()
        else:
            encoder = Encoder(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
            decoder = Decoder(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
            seq2seq = Seq2Seq(encoder, decoder, config.VOCAB_SIZE, args.hidden_size, config.bos)

    train(args, config, seq2seq)
