import torch
import torch.nn as nn
import torch.nn.functional as F


# encoding the sentences
class Encoder(nn.Module):
    def __init__(self, embeds, embedding_dim, hidden_size, bidirectional, num_layers):
        super(Encoder, self).__init__()
        self.embeds = embeds
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.encoder_gru = nn.GRU(self.embedding_dim,
                                  self.hidden_size,
                                  batch_first=True,
                                  bidirectional=self.bidirectional,
                                  num_layers=self.num_layers)

    def forward(self, x):
        """
        :param x: (batch, t_len)
        :return: h(n_layer, batch, hidden_size)
                  out(batch, t_len, hidden_size) hidden state of gru
        """
        e = self.embeds(x)

        # out (batch, time_step, hidden_size*bidirection)
        # h (batch, n_layers*bidirection, hidden_size)
        out, h = self.encoder_gru(e)

        # bidirectional
        if self.bidirectional:
            out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]

        h = h[:self.num_layers]

        # h = h[:self.num_layers] + h[self.num_layers:]
        return h, out


# calculate attention weights and return context vector
class Attention(nn.Module):
    def __init__(self, hidden_size, t_len):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.t_len = t_len

        self.linear_in = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=0.1),
                                       nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=0.1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden, encoder_out):
        """
        :param hidden: (n_layers, batch, hidden_size) decoder hidden state
        :param encoder_out: (batch, time_step, hidden_size) encoder gru hidden state
        :return: attn_weight (batch, 1, time_step)
                  context (batch, 1, hidden_size) attention vector
        """

        hidden = self.linear_in(hidden).view(-1, self.hidden_size, hidden.size(0)) # (n_layer, batch, hidden)
        # print('hidden:', hidden.size())
        weights = torch.bmm(encoder_out, hidden).view(-1, hidden.size(2), self.t_len) # (batch, n_layer, time_step)
        # print('weights:', weights.size())
        weights = self.softmax(weights)

        context = torch.bmm(weights, encoder_out).view(hidden.size(2), -1, self.hidden_size)  # (n_layer, batch, hidden_size)
        # print(context.size())
        return weights, context


class AttnDecoder(nn.Module):
    def __init__(self, attention, embeds, vocab_size, embedding_dim, hidden_size, s_len, num_layers):
        super(AttnDecoder, self).__init__()
        self.attention = attention
        self.embeds = embeds
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.s_len = s_len
        self.num_layers = num_layers

        # decoder
        self.decoder_gru = nn.GRU(self.hidden_size,
                                  self.hidden_size,
                                  batch_first=True,
                                  num_layers=num_layers)
        self.decoder_vocab = nn.Linear(self.hidden_size, self.vocab_size)
        self.linear_out = nn.Sequential(nn.Linear(hidden_size*2, hidden_size), nn.SELU(), nn.Dropout(p=0.1),
                                       nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=0.1))

    def forward(self, x, h, encoder_output):
        """
        :param x: (batch, 1) decoder input
        :param h: (batch, n_layer, hidden_size) code
        :param encoder_output: (batch, t_len, hidden_size) encoder gru hidden state
        :return: attn_weight (batch, 1, time_step)
                  context (batch, 1, hidden_size)
                  out (batch, hidden_size) decoder output
                  h (batch, n_layer, hidden_size) decoder hidden state
        """
        e = self.embeds(x).unsqueeze(1)
        out, h = self.decoder_gru(e, h)
        attn_weight, context = self.attention(h, encoder_output)
        h = torch.cat((h, context), dim=2)
        h = self.linear_out(h)
        return out, h


class AttnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, hidden_size, t_len, s_len, bos):
        super(AttnSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.t_len = t_len
        self.s_len = s_len

        # <bos>
        self.bos = bos

        self.output_Linear = nn.Linear(self.hidden_size, self.vocab_size)

    def output_layer(self, x):
        """
        :param x: (batch, hidden_size) decoder output
        :return: (batch, vocab_size)
        """
        return self.output_Linear(x)

    # add <bos> to sentence
    def convert(self, x):
        """
        :param x:(batch, s_len) (word_1, word_2, ... , word_n)
        :return:(batch, s_len) (<bos>, word_1, ... , word_n-1)
        """
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x[:, :-1]

    def forward(self, x, y):
        """
        :param x: (batch, t_len) encoder input
        :param y: (batch, s_len) decoder input
        :return: loss: coverage loss
                  outputs (batch, t_len, vocabulary)
        """
        h, encoder_out = self.encoder(x)

        # add <bos>
        y = self.convert(y)

        # decoder
        result = []
        for i in range(y.size(1)):
            out, h = self.decoder(y[:, i], h, encoder_out)
            gen = self.output_layer(out).squeeze()
            result.append(gen)

        outputs = torch.stack(result)
        return torch.transpose(outputs, 0, 1)


# embedding
class Embeds(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size):
        super(Embeds, self).__init__()
        if embeddings is None:
            self.embeds = nn.Embedding(vocab_size, embedding_size)
        else:
            self.embeds = nn.Embedding.from_pretrained(embeddings)

    def forward(self, x):
        """
        :param x: (batch, t_len)
        :return: (batch, t_len, embedding_size)
        """
        return self.embeds(x)


class Decoder(nn.Module):
    def __init__(self, embeds, vocab_size, embedding_dim, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embeds = embeds
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # decoder
        self.decoder_gru = nn.GRU(self.embedding_dim,
                                  self.hidden_size,
                                  batch_first=True,
                                  num_layers=num_layers)
        self.decoder_vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, h, flag):
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
        code, _ = self.encoder(x)
        y = self.convert(y)
        out, h = self.decoder(y, code, None)
        out = self.output_layer(out)
        return out
