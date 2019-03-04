import torch
import LSTM


def load_model(embeddings, epoch, config, args):
    if args.attention:
        # attention model
        embeds = LSTM.Embeds(embeddings, config.vocab_size, config.dim)
        encoder = LSTM.Encoder(embeds, config.dim, args.hidden_size, args.bidirectional, args.num_layers)
        attention = LSTM.Attention(args.hidden_size, config.seq_len)
        decoder = LSTM.AttnDecoder(attention, embeds, config.vocab_size, config.dim,
                              args.hidden_size, config.summary_len, args.num_layers)
        model = LSTM.AttnSeq2Seq(encoder, decoder, config.vocab_size, args.hidden_size, config.seq_len,
                                   config.summary_len, config.bos)
    else:
        # seq2seq model
        embeds = LSTM.Embeds(embeddings, config.vocab_size, config.dim)
        encoder = LSTM.Encoder(embeds, config.dim, args.hidden_size, args.bidirectional, args.num_layers)
        decoder = LSTM.Decoder(embeds, config.vocab_size, config.dim, args.hidden_size, args.num_layers)
        model = LSTM.Seq2Seq(encoder, decoder, config.vocab_size, args.hidden_size, config.bos)

    filename = 'models/summary/model_' + str(epoch) + '.pkl'
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print('model save at ', filename)