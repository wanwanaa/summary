import torch
from LSTM.model import Encoder, Decoder, Seq2Seq, Attention, AttnDecoder, AttnSeq2Seq, Embeds
# from LSTM.model import Encoder, Decoder, Seq2Seq, Attention, AttnDecoder, AttnSeq2Seq, get_embeds


def load_model(embeddings, epoch, config, args):
    if args.attention is True:
        # attention model
        embeds = Embeds(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE)
        encoder = Encoder(embeds, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
        # attetntion_v1
        attention = Attention(args.hidden_size)
        # # attention_v2
        # attention = Attention(args.hidden_size, config.EMBEDDING_SIZE, config.seq_len).cuda()
        decoder = AttnDecoder(attention, embeds, config.VOCAB_SIZE, config.EMBEDDING_SIZE,
                              args.hidden_size, config.summary_len, args.num_layers)
        model = AttnSeq2Seq(encoder, decoder, config.VOCAB_SIZE, args.hidden_size, config.summary_len, config.bos)
    else:
        # Seq2Seq
        embeds = Embeds(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE)
        encoder = Encoder(embeds, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
        decoder = Decoder(embeds, config.VOCAB_SIZE, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
        model = Seq2Seq(encoder, decoder, config.VOCAB_SIZE, args.hidden_size, args.num_layers)
    filename = 'models/summary/model_' + str(epoch) + '.pkl'
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


# def load_model(embeddings, epoch, config, args):
#     embeds = get_embeds(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE)
#     if args.attention is True:
#         # attention model
#         encoder = Encoder(embeds, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
#         # attetntion_v1
#         attention = Attention(args.hidden_size)
#         # attention_v2
#         # attention = Attention(args.hidden_size, config.EMBEDDING_SIZE, config.seq_len).cuda()
#         decoder = AttnDecoder(attention, embeds, config.VOCAB_SIZE, config.EMBEDDING_SIZE,
#                               args.hidden_size, config.summary_len, args.num_layers)
#         model = AttnSeq2Seq(encoder, decoder, config.VOCAB_SIZE, args.hidden_size, config.summary_len, config.bos)
#     else:
#         # Seq2Seq
#         encoder = Encoder(embeds, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
#         decoder = Decoder(embeds, config.VOCAB_SIZE, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
#         model = Seq2Seq(encoder, decoder, config.VOCAB_SIZE, args.hidden_size, args.num_layers)
#     filename = 'models/summary/model_' + str(epoch) + '.pkl'
#     model.load_state_dict(torch.load(filename, map_location='cpu'))
#     return model


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print('model save at ', filename)