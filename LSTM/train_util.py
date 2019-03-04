import pickle
import torch
import numpy as np
from LSTM.ROUGE import rouge_score, write_rouge
from LSTM.beam import beam_search
from LCSTS_char.data_utils import index2sentence, load_data


def save_plot(train_loss, valid_loss, test_loss, test_rouge, filename_result):
    result = [train_loss, valid_loss, test_loss, test_rouge]
    filename = filename_result + 'loss.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(result, f)


def valid(config, epoch, model, args, loss_func):
    # filename
    filename_valid_text = config.filename_trimmed_valid_text
    filename_valid_summary = config.filename_trimmed_valid_summary

    # data
    valid = load_data(filename_valid_text, filename_valid_summary, args.batch_size, shuffle=False, num_works=2)

    all_loss = 0
    num = 0

    for batch in valid:
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
    loss = all_loss / num
    print('epoch:', epoch, '|valid_loss: %.4f' % loss)
    return loss


def test(config, epoch, model, args, filename_result, filename_rouge, loss_func):
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
    all_loss = 0
    num = 0
    for batch in test:
        num += 1
        x, y = batch
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        # model
        h, encoder_outputs = model.encoder(x)
        out = (torch.ones(x.size(0)) * bos)
        result = []
        idx = []
        for i in range(s_len):
            if torch.cuda.is_available():
                out = out.type(torch.cuda.LongTensor)
            else:
                out = out.type(torch.LongTensor)

            # seq2seq
            if args.attention is False:
                out = out.view(-1, 1)
            out, h = model.decoder(out, h, encoder_outputs)
            gen = model.output_layer(out).squeeze()
            result.append(gen)
            final = torch.nn.functional.softmax(gen, dim=1)
            out = torch.argmax(final, dim=1)
            idx.append(out.cpu().numpy())
        result = torch.transpose(torch.stack(result), 0, 1)
        idx = np.transpose(np.array(idx))

        # loss
        result = result.contiguous().view(-1, 4000)
        y = y.view(-1)
        loss = loss_func(result, y)
        all_loss += loss.item()

        for i in range(idx.shape[0]):
            # sen1 = index2sentence(list(x[i]), idx2word)
            sen = index2sentence(list(idx[i]), idx2word)
            r.append(' '.join(sen))

    loss = all_loss / num
    print('epoch:', epoch, '|test_loss: %.4f' % loss)

    # write result
    filename_data = filename_result + 'summary_' + str(epoch) + '.txt'
    with open(filename_data, 'w', encoding='utf-8') as f:
        f.write('\n'.join(r))

    # ROUGE
    score = rouge_score(config.gold_summaries, filename_data)

    # write rouge
    write_rouge(filename_rouge, score, epoch)

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

    return loss, score


def beam_test(config, epoch, model, args, filename_result, filename_rouge):
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

    result = []
    r = []
    for batch in test:
        x, _ = batch
        if torch.cuda.is_available():
            x = x.cuda()
        # model
        h, encoder_outputs = model.encoder(x)
        path = beam_search(model, h, encoder_outputs, s_len, bos, args.beam_size, args.attention)
        result.append(path)

    for i in range(len(result)):
        # sen1 = index2sentence(list(x[i]), idx2word)
        sen = index2sentence(result[i], idx2word)
        r.append(' '.join(sen))

    # write result
    filename_data = filename_result + 'summary_' + str(epoch) + '.txt'
    with open(filename_data, 'w', encoding='utf-8') as f:
        f.write('\n'.join(r))

    # ROUGE
    score = rouge_score(config.gold_summaries, filename_data)

    # write rouge
    write_rouge(filename_rouge, score, epoch)

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
