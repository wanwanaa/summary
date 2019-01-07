import torch
from LSTM.model_attn import Encoder, Decoder, Seq2Seq, Attention, AttnDecoder, AttnSeq2Seq


def load_model(embeddings, epoch, config, args):
    if args.attention is True:
        # attention model
        encoder = Encoder(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
        # attetntion_v1
        attention = Attention(config.HIDDEN_SIZE)
        # attention_v2
        # attention = Attention(args.hidden_size, config.EMBEDDING_SIZE, config.seq_len).cuda()
        decoder = AttnDecoder(attention, embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE,
                              args.hidden_size, config.summary_len, args.num_layers)
        model = AttnSeq2Seq(encoder, decoder, config.VOCAB_SIZE, args.hidden_size, config.summary_len, config.bos)
    else:
        # Seq2Seq
        encoder = Encoder(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
        decoder = Decoder(embeddings, config.VOCAB_SIZE, config.EMBEDDING_SIZE, args.hidden_size, args.num_layers)
        model = Seq2Seq(encoder, decoder, config.VOCAB_SIZE, args.hidden_size, args.num_layers)
    filename = 'models/summary/model_' + str(epoch) + '.pkl'
    model.load_state_dict(torch.load(filename, map_location='cpu'))
    return model


def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print('model save at ', filename)