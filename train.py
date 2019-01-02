import time
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LCSTS_char.config import Config
from LSTM.model import Encoder, Decoder, Seq2Seq, Attention, AttnDecoder, AttnSeq2Seq
# from LSTM.ROUGE import rouge_score, write_rouge
from LCSTS_char.data_utils import index2sentence, load_data, load_embeddings

LR = 0.001

# embeddings
filename = 'DATA/data/glove_embeddings_300d.pt'
embeddings = load_embeddings(filename)


def save_model(model, epoch):
    filename = 'LSTM/models/summary/seq2seq/model_' + str(epoch) + '.pkl'
    torch.save(model.state_dict(), filename)
    print('model save at ', filename)


def test(config, epoch, model):
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
        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        h = model.encoder(x)
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
            # sen1 = index2sentence(list(x[i]), idx2word)
            sen = index2sentence(list(result[i]), idx2word)
            r.append(' '.join(sen))
    # write result
    filename_result = 'DATA/result/summary/summary/summary_result_' + str(epoch) + '.txt'
    with open(filename_result, 'w', encoding='utf-8') as f:
        f.write('\n'.join(r))

    # # ROUGE
    # score = rouge_score(config.gold_summaries, filename_result)
    #
    # # write rouge
    # filename_rouge = 'DATA/result/summary/ROUGE_' + str(epoch) + '.txt'
    # write_rouge(filename_rouge, score)
    #
    # # print rouge
    # print('epoch:', epoch, '|ROUGE-1 f: %.4f' % score['rouge-1']['f'],
    #       ' p: %.4f' % score['rouge-1']['p'],
    #       ' r: %.4f' % score['rouge-1']['r'])
    # print('epoch:', epoch, '|ROUGE-2 f: %.4f' % score['rouge-2']['f'],
    #       ' p: %.4f' % score['rouge-2']['p'],
    #       ' r: %.4f' % score['rouge-2']['r'])
    # print('epoch:', epoch, '|ROUGE-L f: %.4f' % score['rouge-l']['f'],
    #       ' p: %.4f' % score['rouge-l']['p'],
    #       ' r: %.4f' % score['rouge-l']['r'])


def train(args, config, model):
    start = time.time()
    # filename
    filename_train_text = config.filename_trimmed_train_text
    filename_train_summary = config.filename_trimmed_train_summary
    filename_idx2word = config.filename_index

    # data
    train = load_data(filename_train_text, filename_train_summary, args.batch_size, shuffle=True, num_works=2)

    # idx2word
    f = open(filename_idx2word, 'rb')
    idx2word = pickle.load(f)

    # loss
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for e in range(args.epoch):
        all_loss = 0
        num = 0
        for step, batch in enumerate(train):
            x, y = batch
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            result = model(x, y)
            a = y[0]
            b = result[0]
            result = result.contiguous().view(-1, 4000)
            y = y.view(-1)
            loss = loss_func(result, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            all_loss += loss.item()
            num += 1
            if step % 20 == 0:
                if torch.cuda.is_available():
                    a = list(a.cpu().numpy())
                    b = list(torch.argmax(b, dim=1).cpu().numpy())
                else:
                    a = list(a.numpy())
                    b = list(torch.argmax(b, dim=1).numpy())
                a = index2sentence(a, idx2word)
                b = index2sentence(b, idx2word)
                print('epoch:', e, '|step:', step, '|train_loss: %.4f' % loss.item())
                print(''.join(a))
                print(''.join(b))
        # train loss
        print('epoch:', e, '|train_loss: %.4f' % (all_loss / num))
        save_model(model, e)
        end = time.time()
        print('time: ', (end-start))
        # test
        # test(config, e, model)


if __name__ == '__main__':
    # Hyper Parameters
    VOCAB_SIZE = 4000
    EMBEDDING_SIZE = 300

    # input
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size for train')
    parser.add_argument('--hidden_size', '-s', type=int, default=512, help='dimension of  code')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='number of training epochs')
    parser.add_argument('--num_layers', '-n', type=int, default=2, help='number of gru layers')
    # parser.add_argument('--attention', '-a', type=) # bool
    # parser.add_argument('--devices', '-d', type=int, default=2, help='specify a gpu')
    args = parser.parse_args()

    # config
    config = Config()

    # # model
    # if torch.cuda.is_available():
    #     encoder = Encoder(embeddings, VOCAB_SIZE, EMBEDDING_SIZE, args.hidden_size, args.num_layers).cuda()
    #     decoder = Decoder(embeddings, VOCAB_SIZE, EMBEDDING_SIZE, args.hidden_size, args.num_layers).cuda()
    #     seq2seq = Seq2Seq(encoder, decoder, VOCAB_SIZE, args.hidden_size, config.bos).cuda()
    # else:
    #     encoder = Encoder(embeddings, VOCAB_SIZE, EMBEDDING_SIZE, args.hidden_size, args.num_layers)
    #     decoder = Decoder(embeddings, VOCAB_SIZE, EMBEDDING_SIZE, args.hidden_size, args.num_layers)
    #     seq2seq = Seq2Seq(encoder, decoder, VOCAB_SIZE, args.hidden_size, config.bos)

    # attention model
    if torch.cuda.is_available():
        encoder = Encoder(embeddings, VOCAB_SIZE, EMBEDDING_SIZE, args.hidden_size, args.num_layers).cuda()
        attention = Attention(args.hidden_size).cuda()
        decoder = AttnDecoder(attention, embeddings, VOCAB_SIZE, EMBEDDING_SIZE,
                              args.hidden_size, config.summary_len, args.num_layers).cuda()
        seq2seq = AttnSeq2Seq(encoder, decoder, VOCAB_SIZE, args.hidden_size, config.summary_len, config.bos).cuda()
    else:
        encoder = Encoder(embeddings, VOCAB_SIZE, EMBEDDING_SIZE, args.hidden_size, args.num_layers)
        attention = Attention(args.hidden_size)
        decoder = AttnDecoder(attention, embeddings, VOCAB_SIZE, EMBEDDING_SIZE,
                              args.hidden_size, config.summary_len, args.num_layers)
        seq2seq = AttnSeq2Seq(encoder, decoder, VOCAB_SIZE, args.hidden_size, config.summary_len, config.bos)

    train(args, config, seq2seq)
