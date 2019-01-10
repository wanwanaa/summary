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

        # # embedding
        # if embeddings is None:
        #     self.embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        # else:
        #     self.embeds = nn.Embedding.from_pretrained(self.embeddings)

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
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size*2, 1)

    def forward(self, hidden, encoder_out):
        # encoder_out (batch, time_step, hidden_size)
        # hidden (n_layers, batch, hidden_size) -> (1, batch, hidden_size)
        #     -> (batch, 1, hidden_size) -> (batch, time_step, hidden_size)
        hidden = hidden[-1].view(-1, 1, self.hidden_size).repeat(1, encoder_out.size(1), 1)

        # attn_weights (batch, 1, time_step)
        # (batch. time_step, 1) -> (batch, 1, time_step)
        attn_weights = F.softmax(self.attn(torch.cat((hidden, encoder_out), dim=2)).squeeze(2)).unsqueeze(1)

        context = torch.bmm(attn_weights, encoder_out) # (batch, 1, hidden_size)
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

        # # embedding
        # if embeddings is None:
        #     self.embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        # else:
        #     self.embeds = nn.Embedding.from_pretrained(self.embeddings)

        # decoder
        self.decoder_gru = nn.GRU(self.embedding_dim+self.hidden_size,
                                  self.hidden_size,
                                  batch_first=True,
                                  num_layers=num_layers,
                                  dropout=0.5)
        self.decoder_vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, x, h, encoder_output):
        e = self.embeds(x).unsqueeze(1)
        context = self.attention(h, encoder_output)
        # print('e:', e.size())
        # print('context:', context.size())
        inputs = torch.cat((e, context), dim=2)
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

        # # embedding
        # if embeddings is None:
        #     self.embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        # else:
        #     self.embeds = nn.Embedding.from_pretrained(self.embeddings)

        # decoder
        self.decoder_gru = nn.GRU(self.embedding_dim,
                                  self.hidden_size,
                                  batch_first=True,
                                  num_layers=num_layers)
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


# # make encode and decode embedding identically
# def get_embeds(embeddings, vocab_size, embedding_size):
#     if embeddings is None:
#         embeds = nn.Embedding(vocab_size, embedding_size)
#     else:
#         embeds = nn.Embedding.from_pretrained(embeddings)
#     return embeds


class Embeds(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size):
        super(Embeds, self).__init__()
        if embeddings is None:
            self.embeds = nn.Embedding(vocab_size, embedding_size)
        else:
            self.embeds = nn.Embedding.from_pretrained(embeddings)

    def forward(self, x):
        return self.embeds(x)
