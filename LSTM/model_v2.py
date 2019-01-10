import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embeds, embedding_dim, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embeds = embeds
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # encoder
        # input size(batch, time_step, embedding_dim)
        # output size(batch, time_step, hidden_size*bidirection)
        self.encoder_gru = nn.GRU(self.embedding_dim,
                                  self.hidden_size,
                                  batch_first=True,
                                  bidirectional=True,
                                  num_layers=self.num_layers,
                                  dropout=0.5)

    def forward(self, x):
        e = self.embeds(x)

        # out (batch, time_step, hidden_size*bidirection)
        # h (n_layers*bidirection, batch, hidden_size)
        out, h = self.encoder_gru(e)

        # out (batch, time_step, hidden_size)
        out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]

        # h (n_layers, time_step, hidden_size)
        h = h[:self.num_layers]
        return h, out


class Attention(nn.Module):
    def __init__(self, hidden_size, embedding_dim, t_len):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.t_len = t_len

        self.attn = nn.Linear(self.hidden_size+self.embedding_dim, self.t_len)

    def forward(self, embedded, hidden, encoder_out):
        # embedded (batch, embedding_dim)
        # encoder_out (batch, time_step, hidden_size)
        # hidden (n_layers, batch, hidden_size)
        # weights (batch, t_len) -> (batch, 1, t_len)
        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), dim=1))).unsqueeze(1)

        # context (batch, 1, hidden_size)
        context = torch.bmm(attn_weights, encoder_out)

        return context


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
        self.attn_combine = nn.Linear(self.hidden_size + self.embedding_dim, self.embedding_dim)
        self.decoder_gru = nn.GRU(self.embedding_dim,
                                  self.hidden_size,
                                  batch_first=True,
                                  num_layers=num_layers,
                                  dropout=0.5)
        self.decoder_vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, h, encoder_output):
        e = self.embeds(x)
        # print('e:', e.size())
        context = self.attention(e, h, encoder_output)
        # print('context:', context.size())
        inputs = self.attn_combine(torch.cat((e.unsqueeze(1), context), dim=2))
        out, h = self.decoder_gru(inputs, h)
        return out, h


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
                                  num_layers=num_layers,
                                  dropout=0.5)
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
        code, _ = self.encoder(x)
        y = self.convert(y)
        out, h = self.decoder(y, code)
        out = self.output_layer(out)
        return out


class AttnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, hidden_size, s_len, bos):
        super(AttnSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.s_len = s_len

        # <bos>
        self.bos = bos

        self.output_Linear = nn.Linear(self.hidden_size, self.vocab_size)

    def output_layer(self, x):
        return self.output_Linear(x)

    def convert(self, x):
        if torch.cuda.is_available():
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.cuda.LongTensor)
        else:
            start = (torch.ones(x.size(0), 1) * self.bos).type(torch.LongTensor)
        x = torch.cat((start, x), dim=1)
        return x[:, :-1]

    def forward(self, x, y):
        h, encoder_out = self.encoder(x)
        y = self.convert(y)
        result = []
        for i in range(self.s_len):
            out, h = self.decoder(y[:, i], h, encoder_out)
            result.append(self.output_layer(out).squeeze())
        outputs = torch.stack(result)
        return torch.transpose(outputs, 0, 1)


class Embeds(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size):
        super(Embeds, self).__init__()
        if embeddings is None:
            self.embeds = nn.Embedding(vocab_size, embedding_size)
        else:
            self.embeds = nn.Embedding.from_pretrained(embeddings)

    def forward(self, x):
        return self.embeds(x)