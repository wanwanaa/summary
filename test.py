import pickle
import torch
import numpy as np
from LCSTS_char.config import Config
from LSTM.model import Encoder, Decoder, Seq2Seq, Attention, AttnDecoder, AttnSeq2Seq
from LSTM.ROUGE import rouge_score, write_rouge
from LCSTS_char.data_utils import index2sentence, load_data, load_embeddings
# embeddings
filename = 'DATA/data/glove_embeddings.pt'
embeddings = load_embeddings(filename)
VOCAB_SIZE = 4000
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 512
BATCH_SIZE = 128
EPOCH = 20


def load_model(epoch):
    filename = 'LSTM/models/summary/attention/model_' + str(epoch) + '.pkl'
    # Seq2Seq
    encoder = Encoder(embeddings, VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, 2)
    decoder = Decoder(embeddings, VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, 2)
    model = Seq2Seq(encoder, decoder, VOCAB_SIZE, HIDDEN_SIZE, 2)

    # # attention model
    # encoder = Encoder(embeddings, VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, 2)
    # attention = Attention(HIDDEN_SIZE)
    # decoder = AttnDecoder(attention, embeddings, VOCAB_SIZE, EMBEDDING_SIZE,
    #                       HIDDEN_SIZE, config.summary_len, 2)
    # model = AttnSeq2Seq(encoder, decoder, VOCAB_SIZE, HIDDEN_SIZE, config.summary_len, config.bos)
    # model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


def test(config, epoch, model):
    # test
    # model = model.eval()

    # filename
    filename_test_text = config.filename_trimmed_test_text
    filename_test_summary = config.filename_trimmed_test_summary
    filename_idx2word = config.filename_index

    # data
    test = load_data(filename_test_text, filename_test_summary, BATCH_SIZE, shuffle=False, num_works=2)

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
        # # seq2seq
        # h, _ = model.encoder(x)
        # attention
        h, encoder_outputs = model.encoder(x)

        out = (torch.ones(x.size(0)) * bos)
        result = []
        for i in range(s_len):
            # seq2seq
            out = out.type(torch.LongTensor).view(-1, 1)
            out, h = model.decoder(out, h)

            # # attention
            # out = out.type(torch.LongTensor)
            # out, h = model.decoder(out, h, encoder_outputs)

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
    filename_result = 'DATA/result/summary/attention/summary_' + str(epoch) + '.txt'
    with open(filename_result, 'w', encoding='utf-8') as f:
        f.write('\n'.join(r))

    # ROUGE
    score = rouge_score(config.gold_summaries, filename_result)

    # write rouge
    filename_rouge = 'DATA/result/summary/attention/ROUGE_' + str(epoch) + '.txt'
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
    for epoch in range(EPOCH):
        model = load_model(epoch)
        test(config, epoch, model)
