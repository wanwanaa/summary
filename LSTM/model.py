import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_dim, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embeddings = embeddings
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # embedding
        self.embeds = nn.Embedding.from_pretrained(embeddings)

        # encoder
        # input size(batch, time_step, embedding_dim)
        # output size(batch, time_step, hidden_size*2)
        self.encoder_gru = nn.GRU(self.embedding_dim,
                                  self.hidden_size,
                                  batch_first=True,
                                  bidirectional=True,
                                  num_layers=self.num_layers)

    def forward(self, x):
        e = self.embeds(x)
        out, _ = self.encoder_gru(e)
        code = out[:, -1, :self.hidden_size] + out[:, -1, self.hidden_size:]
        return code.view(1, -1, self.hidden_size)


class Decoder(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_dim, hidden_size):
        super(Decoder, self).__init__()
        self.embeddings = embeddings
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # embedding
        self.embeds = nn.Embedding.from_pretrained(embeddings)

        # decoder
        self.decoder_gru = nn.GRU(self.embedding_dim, self.hidden_size, batch_first=True)
        self.decoder_vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, h):
        e = self.embeds(x)
        out, h = self.decoder_gru(e, h)
        return out, h


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, hidden_size, bos):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # <bos>
        self.bos = bos

        self.output_Linear = nn.Linear(self.hidden_size, self.vocab_size)

    def convert(self, x):
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x[:, :-1]

    def output_layer(self, x):
        input = x.contiguous().view(-1, self.hidden_size)
        output = self.output_Linear(input)
        return output.view(-1, x.size(1), self.vocab_size)

    def forward(self, x, y):
        code = self.encoder(x)
        y = self.convert(y)
        out, h = self.decoder(y, code)
        out = self.output_layer(out)
        return out, h
