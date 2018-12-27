import torch
import torch.nn as nn
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, embeddings, vocab_size, seq_len, embedding_dim, hidden_size):
        super(Autoencoder, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.input_size = embedding_dim
        self.hidden_size = hidden_size

        # embedding
        self.embedding = nn.Embedding(self.vocab_size, self.input_size)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings).type(torch.FloatTensor))

        # encode
        self.encode_gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.encode_Linear = nn.Linear(self.hidden_size*2, self.hidden_size)

        # decode
        self.decode_gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.decode_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.vocab_size),
            # nn.ReLU()
        )

    # def embeds(self, x):
    #     embeds_matrix = self.embedding(x)
    #     return embeds_matrix

    def convert(self, x):
        indices = torch.from_numpy(np.arange(0, self.seq_len)).type(torch.LongTensor)
        if torch.cuda.is_available():
            indices = torch.from_numpy(np.arange(0, self.seq_len)).type(torch.cuda.LongTensor)
            x = torch.cat(((torch.ones(x.size(0), 1)*2).type(torch.cuda.LongTensor), x), dim=1)
        else:
            x = torch.cat(((torch.ones(x.size(0), 1) * 2).type(torch.LongTensor), x), dim=1)
            indices = torch.from_numpy(np.arange(0, self.seq_len)).type(torch.LongTensor)
        # print('x1', x.size())
        x = torch.index_select(x, 1, indices)
        return x

    def encoder(self, x):
        e = self.embedding(x)
        code, _ = self.encode_gru(e)
        code = code[:, -1, :].view(-1, self.hidden_size*2)
        return self.encode_Linear(code).view(1, -1, self.hidden_size)

    def decoder(self, x, h):
        x = self.embedding(x)
        out, h = self.decode_gru(x, h)
        return out, h

    def output_layer(self, x):
        result = []
        for i in range(x.size(1)):
            # print('x', x[i].size())
            output = self.decode_linear(x[:, i, :])
            # print('output', output.size())
            result.append(output)
        result = torch.stack(result, dim=1)
        return result

    def forward(self, x):
        code = self.encoder(x)
        x = self.convert(x)
        out, _ = self.decoder(x, code)
        result = self.output_layer(out)
        # print(result)
        return result
