import torch
import LSTM


def load_model(embeddings, epoch, config, args):
    if args.attention is True:
        # attention model
        embeds = LSTM.Embeds(embeddings, config.vocab_size, config.dim)
        encoder = LSTM.Encoder(embeds, config.dim, args.hidden_size, args.num_layers)
        # attetntion_v1
        attention = LSTM.Attention(args.hidden_size)
        # # attention_v2
        # attention = Attention(args.hidden_size, config.EMBEDDING_SIZE, config.seq_len).cuda()
        decoder = LSTM.AttnDecoder(attention, embeds, config.vocab_size, config.dim,
                              args.hidden_size, config.summary_len, args.num_layers)
        # pointer
        if args.point:
            pointer = LSTM.Pointer(embeds, config.dim, args.hidden_size)
        else:
            pointer = None
        model = LSTM.AttnSeq2Seq(encoder, decoder, config.vocab_size, args.hidden_size,
                            config.summary_len, config.bos, pointer)
    else:
        # Seq2Seq
        embeds = LSTM.Embeds(embeddings, config.vocab_size, config.dim)
        encoder = LSTM.Encoder(embeds, config.dim, args.hidden_size, args.num_layers)
        decoder = LSTM.Decoder(embeds, config.vocab_size, config.dim, args.hidden_size, args.num_layers)
        model = LSTM.Seq2Seq(encoder, decoder, config.vocab_size, args.hidden_size, args.num_layers)
    filename = 'models/summary/model_' + str(epoch) + '.pkl'
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


# def load_model(embeddings, epoch, config, args):
#     embeds = Embeds(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE)
#     # attention model
#     encoder = Encoder(embeds, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
#     # attetntion_v1
#     attention = Attention(args.hidden_size)
#     # attention_v2
#     # attention = Attention(args.hidden_size, config.EMBEDDING_SIZE, config.seq_len).cuda()
#     decoder = AttnDecoder(attention, embeds, config.VOCAB_SIZE, config.EMBEDDING_SIZE,
#                           args.hidden_size, config.summary_len, args.num_layers)
#     model = AttnSeq2Seq(encoder, decoder, config.VOCAB_SIZE, args.hidden_size, config.summary_len, config.bos)
#     filename = 'models/summary/attentionmodel_' + str(epoch) + '.pkl'
#     model.load_state_dict(torch.load(filename, map_location='cpu'))
#     return model


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print('model save at ', filename)