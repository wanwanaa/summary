import time
import argparse
import torch
import torch.nn as nn
from LCSTS_char.config import Config
from GRU.save_load import save_model
from LCSTS_char.data_utils import load_data, load_embeddings
from GRU.train_util import valid, test, save_plot


# filename
# save model
filename_model = 'models/summary/'
# result
filename_result = 'result/summary/'
# rouge
filename_rouge = 'result/summary/ROUGE.txt'
# initalization
open(filename_rouge, 'w')
# load_model
filename_load_model = 'models/summary/glove/'


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

    # loss result
    train_loss = []
    valid_loss = []
    test_loss = []
    rouge = []

    # loss
    optim = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_func = nn.CrossEntropyLoss()

    start_epoch = 0
    # # loading model
    # if args.load_model != 0:
    #     start_epoch = args.load_model
    #     filename = filename_load_model + 'model_' + args.load_model + '.pkl'
    #     model.load_state_dict(torch.load(filename))

    for e in range(start_epoch, args.epoch):
        model.train()
        all_loss = 0
        num = 0
        for step, batch in enumerate(train):
            num += 1
            x, y = batch
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            result = model(x, y)

            result = result.contiguous().view(-1, 4000)
            y = y.view(-1)
            loss = loss_func(result, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            all_loss += loss.item()
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
        loss = all_loss / num
        print('epoch:', e, '|train_loss: %.4f' % loss)
        train_loss.append(loss)

        # valid
        loss = valid(config, e, model, args, loss_func)
        valid_loss.append(loss)

        # test
        loss, test_rouge = test(config, e, model, args, filename_result, filename_rouge, loss_func)
        test_loss.append(loss)
        rouge.append(test_rouge)

        # save model
        if args.save_model:
            filename = filename_model + 'model_' + str(e) + '.pkl'
            save_model(model, filename)

        # end time
        end = time.time()
        print('time: ', (end - start))
    save_plot(train_loss, valid_loss, test_loss, rouge, filename_result)


if __name__ == '__main__':
    # input
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size for train')
    parser.add_argument('--hidden_size', '-l', type=int, default=512, help='dimension of code')
    parser.add_argument('--epoch', '-e', type=int, default=20, help='number of training epochs')
    parser.add_argument('--num_layers', '-n', type=int, default=2, help='number of gru layers')
    parser.add_argument('-seed', '-s', type=int, default=1234, help="Random seed")
    parser.add_argument('--bidirectional', '-d', action='store_true', default=False, help="whether to use bi-gru")
    parser.add_argument('--pre_train', '-p', action='store_true', default=False, help="load pre-train embedding")
    parser.add_argument('--attention', '-a', action='store_true', default=False, help="whether to use attention")
    parser.add_argument('--save_model', '-m', action='store_true', default=False, help="whether to save model")

    args = parser.parse_args()

    # config
    config = Config()

    torch.manual_seed(args.seed)

    # embedding
    if args.pre_train:
        filename = config.filename_embeddings
        embeddings = load_embeddings(filename)
    else:
        embeddings = None

    # ###### test #######
    args.bidirectional = True
    args.attention = True
    # ###### test #######

    # model
    if args.attention:
        # attention model
        if torch.cuda.is_available():
            embeds = LSTM.Embeds(embeddings, config.vocab_size, config.dim).cuda()
            encoder = LSTM.Encoder(embeds, config.dim, args.hidden_size, args.bidirectional, args.num_layers).cuda()
            attention = LSTM.Attention(args.hidden_size, config.seq_len).cuda()
            decoder = LSTM.AttnDecoder(attention, embeds, config.vocab_size, config.dim,
                                  args.hidden_size, config.summary_len, args.num_layers).cuda()

            seq2seq = LSTM.AttnSeq2Seq(encoder, decoder, config.vocab_size, args.hidden_size, config.seq_len,
                                       config.summary_len, config.bos).cuda()
        else:
            embeds = LSTM.Embeds(embeddings, config.vocab_size, config.dim)
            encoder = LSTM.Encoder(embeds, config.dim, args.hidden_size, args.bidirectional, args.num_layers)
            attention = LSTM.Attention(args.hidden_size, config.seq_len)
            decoder = LSTM.AttnDecoder(attention, embeds, config.vocab_size, config.dim,
                                  args.hidden_size, config.summary_len, args.num_layers)
            seq2seq = LSTM.AttnSeq2Seq(encoder, decoder, config.vocab_size, args.hidden_size, config.seq_len,
                                       config.summary_len, config.bos)
    else:
        # seq2seq model
        if torch.cuda.is_available():
            embeds = LSTM.Embeds(embeddings, config.vocab_size, config.dim).cuda()
            encoder = LSTM.Encoder(embeds, config.dim, args.hidden_size, args.bidirectional, args.num_layers).cuda()
            decoder = LSTM.Decoder(embeds, config.vocab_size, config.dim, args.hidden_size, args.num_layers).cuda()
            seq2seq = LSTM.Seq2Seq(encoder, decoder, config.vocab_size, args.hidden_size, config.bos).cuda()
        else:
            embeds = LSTM.Embeds(embeddings, config.vocab_size, config.dim)
            encoder = LSTM.Encoder(embeds, config.dim, args.hidden_size, args.bidirectional, args.num_layers)
            decoder = LSTM.Decoder(embeds, config.vocab_size, config.dim, args.hidden_size, args.num_layers)
            seq2seq = LSTM.Seq2Seq(encoder, decoder, config.vocab_size, args.hidden_size, config.bos)

    # model
    train(args, config, seq2seq)